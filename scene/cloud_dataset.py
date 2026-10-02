"""Cloud JSON adapter: preserve splits, lazily decode pixels, and use known suns."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scene.dataset_readers import (BasicPointCloud, CameraInfo, SceneInfo, fetchPly,
                                   getNerfppNorm)
from utils.data_cache import LazyImage
from utils.graphics_utils import focal2fov, fov2focal


def is_cloud_dataset(path):
    with open(Path(path) / 'transforms_train.json') as f:
        contents = json.load(f)
    return bool(contents['frames']) and 'sun_direction' in contents['frames'][0]


def read_cloud_scene(args):
    root = Path(args.source_path)
    ply = root / args.init_ply
    if not ply.is_file():
        raise FileNotFoundError(f'Cloud training requires an existing initial point cloud: {ply}')
    pcd = fetchPly(str(ply))
    scale = float(args.scene_scale)
    if scale <= 0:
        raise ValueError('scene_scale must be positive')
    pcd = BasicPointCloud(points=(pcd.points * scale).astype(np.float32),
                          colors=pcd.colors, normals=pcd.normals)
    if np.linalg.norm(pcd.points, axis=1).max() >= 1.0:
        raise ValueError('Scale the cloud into the unit sphere for the fixed orthographic shadow map')
    paths = {}

    def read_split(name):
        content = json.loads((root / f'transforms_{name}.json').read_text())
        frames = content['frames']
        if args.less > 0:
            frames = frames[:args.less]
        if name == 'train' and args.max_training_images > 0:
            selected = set(np.random.permutation(len(frames))[:args.max_training_images].tolist())
            frames = [f for i, f in enumerate(frames) if i in selected]
        cameras = []
        paths[name] = [f['file_path'] for f in frames]
        for i, frame in enumerate(frames):
            image_path = root / frame['file_path']
            with Image.open(image_path) as image:
                W, H = image.size
            factor = min(1.0, args.max_reso / max(W, H)) if args.max_reso > 0 else 1.0
            size = (round(W * factor), round(H * factor))
            c2w = np.array(frame['transform_matrix'], dtype=np.float64)
            c2w[:3, 3] *= scale
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            fovx = float(frame.get('camera_angle_x', content['camera_angle_x']))
            fovy = focal2fov(fov2focal(fovx, size[0]), size[1])
            light = np.asarray(frame['sun_direction'], dtype=np.float32)
            light /= np.linalg.norm(light)
            cameras.append(CameraInfo(
                uid=i, R=w2c[:3, :3].T, T=w2c[:3, 3], FovY=fovy, FovX=fovx,
                image=LazyImage(image_path, size, args.white_background),
                image_path=str(image_path), image_name=frame['file_path'],
                width=size[0], height=size[1], full_width=size[0], full_height=size[1],
                pl_pos=light * 3.0, pl_intensity=np.ones(3, dtype=np.float32),
                light_dir=light, source_frame=frame))
        return cameras

    train, test = read_split('train'), read_split('test')
    if set(paths['train']).intersection(paths['test']):
        raise ValueError('Train and test image paths overlap')
    if not args.eval:
        train.extend(test)
        test = []
    metadata = dict(adapter='cloud_directional', scene_scale=scale,
                    world_axes='Dataset OpenGL world axes; only camera axes converted to OpenCV',
                    light_direction='Unit world-space vector toward the sun',
                    light_intensity=1.0, initial_points=len(pcd.points),
                    initial_ply=str(ply), initial_ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),
                    max_initial_radius=float(np.linalg.norm(pcd.points, axis=1).max()),
                    train_count=len(train), test_count=len(test), eval=bool(args.eval))
    out = Path(args.model_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'dataset_adapter.json').write_text(json.dumps(metadata, indent=2))
    print(f'Cloud dataset: {len(train)} train / {len(test)} test, scale {scale:.9f}, '
          f'{len(pcd.points)} existing initial points; lazy RGB/rays')
    return SceneInfo(pcd, train, test, getNerfppNorm(train), str(ply))
