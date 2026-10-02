"""Original 3DGS PSNR/SSIM and LPIPS-VGG, one saved test pair at a time.

Run with an environment containing lpips (e.g. the original 3DGS venv).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lpips
import numpy as np
from PIL import Image
import torch
from utils.image_utils import psnr
from utils.loss_utils import ssim


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--iteration', type=int, default=100000)
    args = parser.parse_args()
    directory = args.model_path / 'test' / f'ours_{args.iteration}'
    rows = json.loads((directory / 'views.json').read_text())
    network = lpips.LPIPS(net='vgg').cuda().eval()
    def load(path):
        with Image.open(path) as image:
            pixels = np.array(image.convert('RGB'), dtype=np.uint8)
        return torch.from_numpy(pixels).permute(2, 0, 1).cuda().float() / 255.0
    for i, row in enumerate(rows):
        image, gt = load(directory / 'renders' / row['filename']), load(directory / 'gt' / row['filename'])
        row.update(psnr=float(psnr(image[None], gt[None]).mean()), ssim=float(ssim(image[None], gt[None])),
                   lpips_vgg=float(network(image[None]*2-1, gt[None]*2-1).item()))
        if (i+1) % 10 == 0:
            print(f'Evaluated {i+1}/{len(rows)}', flush=True)
    def mean(selected):
        if not selected:
            return dict(count=0)
        return dict(count=len(selected), **{k:sum(r[k] for r in selected)/len(selected)
                                           for k in ('psnr', 'ssim', 'lpips_vgg')})
    held_out_suns = {7, 22, 37, 52}
    result = dict(metric_protocol='Original 3DGS PSNR/SSIM, lpips.LPIPS(net=vgg), mean per image',
                  precision='uint8 saved PNG / 255', all=mean(rows),
                  heldout_suns=mean([r for r in rows if r['sun_index'] in held_out_suns]),
                  seen_sun_new_combinations=mean([r for r in rows if r['sun_index'] not in held_out_suns]))
    (args.model_path / 'per_view_metrics.json').write_text(json.dumps(rows, indent=2))
    (args.model_path / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
