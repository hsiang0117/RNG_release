"""Training sampling, fixed validation previews, and small resource/progress records."""
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

import torch
import torchvision

from utils.data_cache import cache_stats
from utils.image_utils import psnr


def shuffled_cameras(cameras):
    """Keep the upstream pop(randint) order, with two host decodes in flight."""
    pending, stack = deque(), []
    while True:
        while len(pending) < 3:
            if not stack:
                stack = cameras.copy()
            camera = stack.pop(random.randint(0, len(stack)-1))
            camera.prefetch_image()
            pending.append(camera)
        yield pending.popleft()


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    return 0


class Monitor:
    def __init__(self, output, stage, cameras):
        self.root = Path(output)
        self.stage, self.start = stage, time.monotonic()
        self.rss_peak = 0
        self.preview_dir = self.root / 'rendertest'
        self.preview_dir.mkdir(exist_ok=True)
        ordered = sorted(cameras, key=lambda c: c.image_name)
        chosen = []
        for camera_index in (1, 10):
            for sun_index in (7, 37):
                match = next((c for c in ordered if c.source_frame is not None and
                              c.source_frame.get('camera_index') == camera_index and
                              c.source_frame.get('time_index') == sun_index), None)
                if match is not None:
                    chosen.append(match)
        self.views = chosen or ordered[:4]
        manifest = [dict(file_path=c.image_name,
                         sun_direction=c.light_dir.detach().cpu().tolist() if c.light_dir is not None else None,
                         source_frame=c.source_frame) for c in self.views]
        (self.preview_dir / 'views.json').write_text(json.dumps(manifest, indent=2))
        if self.views:
            torchvision.utils.save_image(torch.stack([c.original_image.cpu() for c in self.views]),
                                         self.preview_dir / 'ground_truth.png', nrow=2)

    def record(self, iteration, loss, image, gt, gaussians, finite_grads=None):
        self.rss_peak = max(self.rss_peak, rss_bytes())
        result = dict(stage=self.stage, iteration=iteration,
                      loss=float(loss.detach()), psnr=float(psnr(image.clamp(0, 1)[None], gt[None]).mean()),
                      gaussian_count=len(gaussians.get_xyz), elapsed_seconds=time.monotonic()-self.start,
                      timestamp=datetime.now().astimezone().isoformat(), rss_bytes=rss_bytes(),
                      peak_rss_bytes=self.rss_peak, cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), caches=cache_stats())
        if finite_grads is not None:
            result['finite_gradients'] = finite_grads
        with (self.root / f'training_stats_{self.stage}.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        temp = self.root / 'training_progress.json.tmp'
        temp.write_text(json.dumps(result, indent=2))
        os.replace(temp, self.root / 'training_progress.json')

    @torch.no_grad()
    def preview(self, iteration, gaussians, render, pipe, background, color_mlp, depth_mlp):
        images = []
        for c in self.views:
            image = render(c, gaussians, pipe, background, color_mlp=color_mlp,
                           depth_mlp=depth_mlp, iteration=iteration)['render'].clamp(0, 1)
            if not torch.isfinite(image).all():
                raise FloatingPointError(f'Non-finite validation preview at {iteration}: {c.image_name}')
            images.append(image.cpu())
        if images:
            torchvision.utils.save_image(torch.stack(images), self.preview_dir / f'{iteration:07d}.png', nrow=2)
        print(f'[ITER {iteration}] Saved fixed camera/sun preview')

    def finish(self, first_iter, final_iter, gaussians):
        result = dict(stage=self.stage, first_iteration=first_iter, final_iteration=final_iter,
                      training_steps=final_iter-first_iter, elapsed_seconds=time.monotonic()-self.start,
                      gaussian_count=len(gaussians.get_xyz), peak_rss_bytes=self.rss_peak,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), caches=cache_stats())
        (self.root / f'training_summary_{self.stage}.json').write_text(json.dumps(result, indent=2))
