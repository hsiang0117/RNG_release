"""Stream the exact held-out cloud split using saved stage configuration."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchvision
from tqdm import tqdm
from arguments import ModelParams, PipelineParams
from color_mlp import ColorMLP
from depth_mlp import DepthMLP
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--stage', default='deferred', choices=['forward', 'deferred'])
    parser.add_argument('--iteration', type=int, default=100000)
    parser.add_argument('--limit', type=int, default=0, help='Validation only; 0 renders the complete test split')
    cli = parser.parse_args()
    saved = json.loads((cli.model_path / f'training_args_{cli.stage}.json').read_text())
    saved['model_path'] = str(cli.model_path.resolve())
    saved['eval'] = True
    args = argparse.Namespace(**saved)
    groups = argparse.ArgumentParser()
    model, pipeline = ModelParams(groups), PipelineParams(groups)
    pipe = pipeline.extract(args)
    pipe.shadow_grad = False
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=cli.iteration, shuffle=False)
    n = pipe.in_channels + 7 + pipe.encoding_levels_each*12
    if pipe.shadow_map:
        n += 1 + pipe.encoding_levels_shadow*2
    color = ColorMLP(n, checkpoint=str(cli.model_path / f'color_mlp_chkpnt{cli.iteration}.pth')).cuda().eval()
    depth = DepthMLP(3, checkpoint=str(cli.model_path / f'depth_mlp_chkpnt{cli.iteration}.pth'),
                     depth_mlp_modifier=pipe.depth_mlp_modifier).cuda().eval() if pipe.depth_mlp else None
    output = cli.model_path / 'test' / f'ours_{cli.iteration}'
    for name in ('renders', 'gt'):
        (output / name).mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.monotonic()
    views = scene.getTestCameras()
    if cli.limit > 0:
        views = views[:cli.limit]
    for i, view in enumerate(tqdm(views, desc='Render held-out split')):
        image = render(view, gaussians, pipe, torch.zeros(3, device='cuda'), color_mlp=color,
                       depth_mlp=depth, iteration=cli.iteration)['render'].clamp(0, 1)
        gt = view.original_image.cuda()
        if not torch.isfinite(image).all():
            raise FloatingPointError(f'Non-finite test render: {view.image_name}')
        filename = f'{i:05d}.png'
        torchvision.utils.save_image(image, output / 'renders' / filename)
        torchvision.utils.save_image(gt, output / 'gt' / filename)
        row = dict(index=i, filename=filename, file_path=view.image_name,
                   camera_index=view.source_frame['camera_index'], sun_index=view.source_frame['time_index'],
                   sun_direction=view.source_frame['sun_direction'],
                   psnr_float=float(psnr(image[None], gt[None]).mean()), ssim_float=float(ssim(image[None], gt[None])))
        rows.append(row)
        if (i+1) % 10 == 0:
            (cli.model_path / 'evaluation_progress.json').write_text(json.dumps(dict(rendered=i+1, total=len(scene.getTestCameras()))))
    (output / 'views.json').write_text(json.dumps(rows, indent=2))
    summary = dict(count=len(rows), precision='clamped float32 renderer output',
                   psnr=sum(r['psnr_float'] for r in rows)/len(rows),
                   ssim=sum(r['ssim_float'] for r in rows)/len(rows),
                   elapsed_seconds=time.monotonic()-start)
    (output / 'metrics_float.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
