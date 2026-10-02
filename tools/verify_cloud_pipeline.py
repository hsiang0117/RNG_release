"""Check actual cloud pixels, geometry, sun inputs, crop optimizers, and both GPU stages."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from color_mlp import ColorMLP
from depth_mlp import DepthMLP
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.data_cache import ByteLRU, cache_stats
from utils.loss_utils import l1_loss, ssim


def main():
    parser = argparse.ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args()
    torch.manual_seed(0)
    args.source_path = '/myfiles/data/CloudDatasetUniform_envon'
    args.model_path = str(ROOT / 'temporary-build/cloud-pipeline-verification')
    args.eval = args.json = args.color_mlp = args.directional_light = True
    args.data_device = 'cuda'
    args.resolution = 1
    args.max_reso = 1024
    args.scene_scale = 1.0 / (27.463693618774414 * 1.05)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(lp.extract(args), gaussians, shuffle=False)
    assert len(scene.getTrainCameras()) == 1308 and len(scene.getTestCameras()) == 152
    gaussians.training_setup(op.extract(args))
    camera = scene.getTrainCameras()[0]
    expected = torch.from_numpy(np.array(Image.open(Path(args.source_path) / camera.image_name).convert('RGB'))).permute(2, 0, 1).float().cuda() / 255.0
    assert torch.equal(camera.original_image, expected), 'GT pixel conversion changed'
    rays, unnorm = camera.gen_rays_from_image(camera.full_height, camera.full_width, camera.focal, camera.C2W)
    torch.testing.assert_close(camera.camera_rays.cpu(), rays, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(camera.camera_rays_unnorm.cpu(), unnorm, rtol=2e-6, atol=2e-6)
    results = dict(split='1308 train / 152 test', initial_points=len(gaussians.get_xyz),
                   pixels='bit identical to dataset RGB', rays='matches upstream within 2e-6',
                   scene_scale=args.scene_scale)
    # Camera and point normalization preserves projected positions.
    points = scene.gaussians.get_xyz[:1000].detach().cpu().numpy() / args.scene_scale
    original_c2w = np.asarray(camera.source_frame['transform_matrix']).copy()
    original_c2w[:3, 1:3] *= -1
    original = (np.linalg.inv(original_c2w)[:3, :3] @ points.T + np.linalg.inv(original_c2w)[:3, 3:4])
    normalized = camera.W2C[:3, :3] @ (points * args.scene_scale).T + camera.W2C[:3, 3:4]
    np.testing.assert_allclose(original[:2]/original[2:3], normalized[:2]/normalized[2:3], atol=2e-6, rtol=2e-6)
    results['projection'] = 'unchanged by common camera/point scale'
    cache = ByteLRU(12)
    cache.put('a', np.zeros(8, np.uint8)); cache.put('b', np.zeros(8, np.uint8))
    assert cache.bytes == 8 and cache.get('a') is None
    results['cache_eviction'] = 'passed'
    pipe = pp.extract(args)
    background = torch.zeros(3, device='cuda')
    for stage in ('forward', 'deferred'):
        if stage == 'deferred':
            pipe.defer_shading = pipe.shadow_map = pipe.shadow_grad = pipe.depth_mlp = True
            pipe.encoding_levels_each, pipe.encoding_levels_shadow = 2, 8
            gaussians.crop_pc(1.0)
        channels = pipe.in_channels + 7 + pipe.encoding_levels_each * 12
        if pipe.shadow_map:
            channels += 1 + pipe.encoding_levels_shadow * 2
        color = ColorMLP(channels).cuda()
        depth = DepthMLP(3).cuda() if stage == 'deferred' else None
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        losses = []
        for j in range(2):
            view = scene.getTrainCameras()[j]
            res = render(view, gaussians, pipe, background, color_mlp=color, depth_mlp=depth,
                         iteration=30000+j)
            image, gt = res['render'], view.original_image
            assert image.shape == (3, 1024, 1024) and torch.isfinite(image).all()
            loss = 0.8*l1_loss(image, gt) + 0.2*(1-ssim(image, gt))
            loss.backward()
            params = [p for group in gaussians.optimizer.param_groups for p in group['params']]
            params += list(color.parameters()) + (list(depth.parameters()) if depth else [])
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            assert gaussians._xyz.grad.abs().sum() > 0
            for optimizer in [gaussians.optimizer, color.optimizer] + ([depth.optimizer] if depth else []):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            del res, loss, image, gt
        torch.cuda.synchronize()
        results[stage] = dict(steps=2, loss=losses, finite_geometry_and_mlp_gradients=True,
                              elapsed_seconds=time.monotonic()-start,
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with torch.no_grad():
            a = render(scene.getTrainCameras()[0], gaussians, pipe, background, color_mlp=color, depth_mlp=depth, iteration=30010)['render']
            b = render(scene.getTrainCameras()[1], gaussians, pipe, background, color_mlp=color, depth_mlp=depth, iteration=30010)['render']
            assert not a.requires_grad and not b.requires_grad
            assert (a-b).abs().mean() > 0, 'Changing the sun has no effect'
            results[stage]['same_camera_sun_change_mae'] = float((a-b).abs().mean())
        del color, depth, a, b
    # Force actual pruning and verify that parameter identity stays connected to Adam.
    small = GaussianModel(args.sh_degree)
    small.create_from_pcd(type('PCD', (), dict(points=np.array([[0.,0.,0.],[.2,0.,0.],[0.,.2,0.],[1.2,0.,0.]],np.float32), colors=np.ones((4,3),np.float32))), 1.0)
    small.training_setup(op.extract(args)); small.crop_pc(1.0)
    names = dict(xyz='_xyz', f_dc='_features_dc', f_rest='_features_rest', opacity='_opacity', scaling='_scaling', rotation='_rotation')
    assert len(small.get_xyz) == 3
    assert all(g['params'][0] is getattr(small,names[g['name']]) for g in small.optimizer.param_groups)
    results['crop_optimizer'] = 'pruned parameters remain connected to Adam'
    results['caches'] = cache_stats()
    path = ROOT / 'temporary-build/verify_cloud_pipeline.json'
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
