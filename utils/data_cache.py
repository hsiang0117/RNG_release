"""Bounded caches for compact host images, CUDA frames/rays, and host shadows."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image


class ByteLRU:
    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        self.items = OrderedDict()
        self.sizes = {}
        self.bytes = self.hits = self.misses = 0

    def get(self, key):
        value = self.items.get(key)
        if value is None:
            self.misses += 1
        else:
            self.items.move_to_end(key)
            self.hits += 1
        return value

    def put(self, key, value):
        size = sum(x.nbytes if isinstance(x, np.ndarray) else x.numel() * x.element_size()
                   for x in (value if isinstance(value, tuple) else (value,)))
        if key in self.items:
            self.items.pop(key)
            self.bytes -= self.sizes.pop(key)
        if size > self.max_bytes:
            return value
        while self.items and self.bytes + size > self.max_bytes:
            old, _ = self.items.popitem(last=False)
            self.bytes -= self.sizes.pop(old)
        self.items[key] = value
        self.sizes[key] = size
        self.bytes += size
        return value

    def stats(self):
        return dict(bytes=self.bytes, entries=len(self.items), hits=self.hits,
                    misses=self.misses, max_bytes=self.max_bytes)


HOST_IMAGES = ByteLRU(6 * 1024**3)
CUDA_IMAGES = ByteLRU(32 * 1024**2)
CUDA_RAYS = ByteLRU(96 * 1024**2)
HOST_SHADOWS = ByteLRU(12 * 1024**3)
_workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix='image-prefetch')
_pending = {}


def decode_image(path, size, white_background):
    with Image.open(path) as source:
        rgba = np.array(source.convert('RGBA'), dtype=np.uint8)
    if np.all(rgba[..., 3] == 255):
        rgb = rgba[..., :3].copy()
    else:
        values = rgba.astype(np.float64) / 255.0
        rgb = np.asarray((values[..., :3] * values[..., 3:4] +
                         float(white_background) * (1.0 - values[..., 3:4])) * 255.0,
                         dtype=np.uint8)
    if (rgb.shape[1], rgb.shape[0]) != tuple(size):
        rgb = np.array(Image.fromarray(rgb).resize(size), dtype=np.uint8)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


class LazyImage:
    def __init__(self, path, size, white_background=False):
        self.path, self.size = str(path), tuple(size)
        self.white_background = bool(white_background)
        self.key = (self.path, self.size, self.white_background)

    def prefetch(self):
        if self.key not in HOST_IMAGES.items and self.key not in _pending:
            _pending[self.key] = _workers.submit(
                decode_image, self.path, self.size, self.white_background)

    def host(self):
        data = HOST_IMAGES.get(self.key)
        if data is None:
            future = _pending.pop(self.key, None)
            data = future.result() if future else decode_image(
                self.path, self.size, self.white_background)
            HOST_IMAGES.put(self.key, data)
        return data

    def tensor(self, device):
        device = torch.device(device)
        if device.type != 'cuda':
            return torch.from_numpy(self.host()).to(device=device, dtype=torch.float32) / 255.0
        key = (self.key, str(device))
        value = CUDA_IMAGES.get(key)
        if value is None:
            value = torch.from_numpy(self.host()).to(device=device, dtype=torch.float32) / 255.0
            CUDA_IMAGES.put(key, value)
        return value


def camera_rays(camera):
    rotation = np.ascontiguousarray(camera.C2W[:3, :3], dtype=np.float32)
    key = (camera.full_height, camera.full_width, float(camera.focal), rotation.tobytes())
    rays = CUDA_RAYS.get(key)
    if rays is None:
        device = camera.camera_center.device
        H, W, focal = camera.full_height, camera.full_width, camera.focal
        u, v = torch.meshgrid(torch.arange(W, device=device, dtype=torch.float32),
                              torch.arange(H, device=device, dtype=torch.float32), indexing='xy')
        local = torch.stack(((u + 0.5 - W/2) / focal,
                             (v + 0.5 - H/2) / focal, torch.ones_like(u)), dim=0)
        world = (torch.as_tensor(rotation, device=device) @ local.reshape(3, -1)).reshape(3, H, W)
        rays = (world / torch.linalg.vector_norm(world, dim=0, keepdim=True), world)
        CUDA_RAYS.put(key, rays)
    return rays


def cache_stats():
    return {name: cache.stats() for name, cache in
            [('host_images', HOST_IMAGES), ('cuda_images', CUDA_IMAGES),
             ('cuda_rays', CUDA_RAYS), ('host_shadows', HOST_SHADOWS)]}
