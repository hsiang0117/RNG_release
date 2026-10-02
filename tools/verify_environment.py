"""Exercise RNG's installed CUDA extensions and networks without a dataset."""
import importlib.metadata
import json
import math
from pathlib import Path

import torch

from color_mlp import ColorMLP
from depth_mlp import DepthMLP
from simple_knn._C import distCUDA2
from utils.graphics_utils import getProjectionMatrix


def finite(x):
    assert torch.isfinite(x).all(), "Non-finite tensor"


def render_check(module, orthographic=False):
    device = "cuda"
    n, channels = 32, 16
    xyz = torch.randn(n, 3, device=device) * 0.15
    xyz[:, 2] += 2.0
    xyz.requires_grad_()
    screen = torch.zeros_like(xyz, requires_grad=True)
    colors = torch.rand(n, channels, device=device, requires_grad=True)
    opacity = torch.full((n, 1), 0.3, device=device, requires_grad=True)
    scales = torch.full((n, 3), 0.07, device=device, requires_grad=True)
    rotations = torch.zeros(n, 4, device=device)
    rotations[:, 0] = 1.0
    rotations.requires_grad_()
    settings = dict(
        image_height=64, image_width=64, bg=torch.zeros(channels, device=device),
        scale_modifier=1.0, viewmatrix=torch.eye(4, device=device),
        projmatrix=getProjectionMatrix(0.01, 100.0, math.pi / 2, math.pi / 2).T.cuda(),
        sh_degree=0, campos=torch.zeros(3, device=device),
        prefiltered=False, debug=False, low_pass_filter_radius=0.3,
    )
    if orthographic:
        settings.update(
            light_dir=torch.tensor([0.0, 0.0, 1.0], device=device),
            cam_center=torch.zeros(3, device=device),
            cam_right=torch.tensor([1.0, 0.0, 0.0], device=device),
            cam_up=torch.tensor([0.0, 1.0, 0.0], device=device),
        )
    else:
        settings.update(tanfovx=1.0, tanfovy=1.0)
    raster = module.GaussianRasterizer(module.GaussianRasterizationSettings(**settings))
    image, radii, depth, alpha = raster(
        means3D=xyz, means2D=screen, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=rotations,
    )
    for x in (image, depth, alpha):
        finite(x)
    assert image.shape == (channels, 64, 64)
    assert (radii > 0).any() and alpha.max() > 0
    result = dict(image_shape=list(image.shape), visible=int((radii > 0).sum()),
                  max_alpha=float(alpha.detach().max()), forward="passed")
    loss = image.square().mean()
    if orthographic:
        loss = loss + depth.square().mean() + alpha.square().mean()
    loss.backward()
    for name, tensor in dict(xyz=xyz, screen=screen, colors=colors,
                             opacity=opacity, scales=scales, rotations=rotations).items():
        assert tensor.grad is not None, name
        finite(tensor.grad)
    assert colors.grad.abs().sum() > 0 and xyz.grad.abs().sum() > 0
    result["feature_backward"] = "passed"
    if orthographic:
        result["depth_alpha_backward"] = "passed"
    return result


def main():
    torch.manual_seed(20261001)
    assert torch.cuda.is_available()
    import diff_gaussian_rasterization as perspective
    import diff_gaussian_rasterization_orthographic as orthographic

    points = torch.randn(257, 3, device="cuda")
    result = distCUDA2(points)
    reference = torch.cdist(points.double(), points.double()).square()
    reference.fill_diagonal_(float("inf"))
    reference = reference.topk(3, largest=False).values.mean(dim=1).float()
    torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
    stats = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "simple_knn": {"status": "passed", "max_abs_error": float((result-reference).abs().max())},
        "perspective": render_check(perspective),
        "orthographic": render_check(orthographic, True),
    }
    for name, model, dim in [("color_mlp", ColorMLP(23).cuda(), 23),
                             ("depth_mlp", DepthMLP(3).cuda(), 3)]:
        for _ in range(5):
            x = torch.rand(64, dim, device="cuda")
            model.optimizer.zero_grad(set_to_none=True)
            output = model(x)
            finite(output)
            output.square().mean().backward()
            for parameter in model.parameters():
                assert parameter.grad is not None
                finite(parameter.grad)
            model.optimizer.step()
        stats[name] = "five CUDA optimization steps passed"
    stats["packages"] = {name: importlib.metadata.version(name) for name in
                         ["torchvision", "numpy", "torchmetrics", "OpenEXR", "setuptools", "ninja"]}
    torch.cuda.synchronize()
    Path("temporary-build/verify_environment.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
