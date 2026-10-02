"""Compare the orthographic CUDA adjoint to a separate float64 PyTorch renderer."""
import argparse
import json
from pathlib import Path

import torch
import diff_gaussian_rasterization_orthographic as native

torch.manual_seed(17744)
DEVICE = 'cuda'
H, W, C = 40, 48, 16
light = torch.tensor([0.3, -0.4, 0.8660254], device=DEVICE)
light = light / light.norm()
right = torch.linalg.cross(torch.tensor([0., 1., 0.], device=DEVICE), light)
right = right / right.norm()
up = torch.linalg.cross(light, right)
center = torch.tensor([0.12, -0.08, 0.2], device=DEVICE)
bg = torch.linspace(.01, .15, C, device=DEVICE)


def covariance(scales, rotations, modifier):
    r, x, y, z = rotations.unbind(1)
    R = torch.stack([
        1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
        2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
        2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y),
    ], dim=1).reshape(-1, 3, 3)
    return R @ torch.diag_embed((modifier*scales).square()) @ R.transpose(1, 2)


def inputs():
    xyz = center + torch.tensor([1.4, 1.8, 2.3], device=DEVICE)[:, None]*light
    xyz = xyz + torch.tensor([-.11, .05, .12], device=DEVICE)[:, None]*right
    xyz = xyz + torch.tensor([.08, -.07, .02], device=DEVICE)[:, None]*up
    rotation = torch.tensor([[.95,.15,-.20,.18], [.91,-.23,.12,.29], [.86,.24,.32,-.20]], device=DEVICE)
    rotation = rotation/rotation.norm(dim=1, keepdim=True)
    return dict(xyz=xyz, colors=torch.rand(3,C,device=DEVICE)*.7+.1,
                opacity=torch.tensor([[.24],[.36],[.31]],device=DEVICE),
                scales=torch.tensor([[.15,.12,.09],[.13,.16,.11],[.14,.10,.17]],device=DEVICE),
                rotations=rotation)


def cuda_render(p, modifier=1.):
    settings = native.GaussianRasterizationSettings(
        image_height=H, image_width=W, light_dir=light, cam_center=center,
        cam_right=right, cam_up=up, bg=bg, scale_modifier=modifier,
        viewmatrix=torch.eye(4,device=DEVICE), projmatrix=torch.eye(4,device=DEVICE),
        sh_degree=0, campos=center, prefiltered=False, debug=False, low_pass_filter_radius=.3)
    screen=torch.zeros_like(p['xyz'], requires_grad=True)
    args=dict(means3D=p['xyz'], means2D=screen, colors_precomp=p['colors'], opacities=p['opacity'])
    if 'cov' in p: args['cov3D_precomp']=p['cov']
    else: args.update(scales=p['scales'], rotations=p['rotations'])
    image, radii, depth, alpha=native.GaussianRasterizer(settings)(**args)
    return (image, depth, alpha), screen


def reference(p, modifier=1.):
    # No native projection/compositing helper is used here.
    dtype=p['xyz'].dtype
    L, X, Y, O, B=[v.to(dtype) for v in (light,right,up,center,bg)]
    delta=p['xyz']-O
    depths=delta@L
    plane=delta-depths[:,None]*L
    px=((plane@X+1)*(W-1)/2+.5).clamp_max(W-.5)
    py=((plane@Y+1)*(H-1)/2+.5).clamp_max(H-.5)
    if 'cov' in p:
        a,b,c,d,e,f=p['cov'].unbind(1)
        sigma=torch.stack([a,b,c,b,d,e,c,e,f],dim=1).reshape(-1,3,3)
    else: sigma=covariance(p['scales'],p['rotations'],modifier)
    J=torch.stack([X*(W/2),Y*(H/2)])
    projected=J[None]@sigma@J.T[None]+.3*torch.eye(2,device=DEVICE,dtype=dtype)
    inverse=projected.inverse()
    yy,xx=torch.meshgrid(torch.arange(H,device=DEVICE,dtype=dtype),torch.arange(W,device=DEVICE,dtype=dtype),indexing='ij')
    transmittance=torch.ones(H,W,device=DEVICE,dtype=dtype)
    image=torch.zeros(C,H,W,device=DEVICE,dtype=dtype)
    depth=torch.zeros(H,W,device=DEVICE,dtype=dtype)
    alpha_sum=torch.zeros_like(depth)
    done=torch.zeros(H,W,device=DEVICE,dtype=torch.bool)
    for i in depths.detach().argsort().tolist():
        dx,dy=px[i]-xx,py[i]-yy
        power=-.5*(inverse[i,0,0]*dx.square()+inverse[i,1,1]*dy.square())-inverse[i,0,1]*dx*dy
        alpha=(p['opacity'][i,0]*power.exp()).clamp_max(.99)
        valid=(power<=0)&(alpha>=1/255)&~done
        done=done | (valid & ((transmittance*(1-alpha)).detach()<.0001))
        alpha=torch.where(valid & ~done,alpha,0.)
        weight=alpha*transmittance
        image=image+p['colors'][i,:,None,None]*weight
        depth=depth+depths[i]*weight
        alpha_sum=alpha_sum+weight
        transmittance=transmittance*(1-alpha)
    return image+B[:,None,None]*transmittance,depth[None],alpha_sum[None]


def objective(outputs, mode, edge=False):
    image,depth,alpha=outputs
    # Keep finite differences away from thresholded footprint boundaries.
    columns=slice(45,48) if edge else slice(22,27)
    image=image[:,18:23,columns]; depth=depth[:,18:23,columns]; alpha=alpha[:,18:23,columns]
    if mode=='feature': return image.square().mean()
    if mode=='depth': return depth.square().mean()
    if mode=='alpha': return alpha.square().mean()
    if mode=='normalized_depth': return (depth/alpha.clamp_min(.01)).square().mean()
    return image.square().mean()+.3*depth.square().mean()+.2*alpha.square().mean()


def run_case(name, mode, modifier=1., precomputed=False, saturated=False, clamped_mean=False):
    original=inputs()
    if saturated: original['opacity']=torch.tensor([[1.2],[.36],[.31]],device=DEVICE)
    if clamped_mean: original['xyz'][0]+=right*1.2
    if precomputed:
        sigma=covariance(original.pop('scales'),original.pop('rotations'),modifier)
        original['cov']=sigma[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]
    p={k:v.clone().requires_grad_() for k,v in original.items()}
    q={k:v.double().detach().requires_grad_() for k,v in original.items()}
    out,screen=cuda_render(p,modifier)
    ref=reference(q,modifier)
    for actual,expected in zip(out,ref):torch.testing.assert_close(actual.double(),expected,rtol=2e-5,atol=2e-6)
    objective(out,mode,clamped_mean).backward();objective(ref,mode,clamped_mean).backward()
    errors={}
    for key in p:
        # Mathematically unused inputs may receive None in the reference.
        expected=q[key].grad if q[key].grad is not None else torch.zeros_like(q[key])
        assert p[key].grad is not None and torch.isfinite(p[key].grad).all(),key
        torch.testing.assert_close(p[key].grad.double(),expected,rtol=5e-4,atol=2e-5,msg=lambda msg:key+': '+msg)
        errors[key]=float((p[key].grad.double()-expected).abs().max())
    assert screen.grad is not None and torch.isfinite(screen.grad).all()
    fd_errors={}
    if not clamped_mean:
        for key,index in [('xyz',(1,2)),('opacity',(1,0)),('colors',(2,7)),
                          ('cov' if precomputed else 'scales',(0,1))]:
            eps=1e-3
            plus={k:v.detach().clone() for k,v in p.items()}; minus={k:v.detach().clone() for k,v in p.items()}
            plus[key][index]+=eps;minus[key][index]-=eps
            with torch.no_grad():
                fd=(objective(cuda_render(plus,modifier)[0],mode)-objective(cuda_render(minus,modifier)[0],mode))/(2*eps)
            torch.testing.assert_close(p[key].grad[index],fd,rtol=.025,atol=.0004)
            fd_errors[key]=float((p[key].grad[index]-fd).abs())
    return dict(name=name,mode=mode,max_absolute_gradient_errors=errors,
                finite_difference_absolute_errors=fd_errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record-forward',action='store_true');args=parser.parse_args()
    fixture=Path('temporary-build/orthographic-forward-before.pt')
    if args.record_forward:
        p=inputs();out,_=cuda_render(p)
        torch.save(dict(inputs={k:v.cpu() for k,v in p.items()},outputs=[x.detach().cpu() for x in out]),fixture)
        print('Saved pre-fix forward outputs');return
    unchanged=None
    if fixture.exists():
        saved=torch.load(fixture,weights_only=True)
        after,_=cuda_render({k:v.cuda() for k,v in saved['inputs'].items()})
        for a,b in zip(after,saved['outputs']):torch.testing.assert_close(a.cpu(),b,rtol=0,atol=0)
        unchanged=True
    results=[run_case(mode,mode) for mode in ['feature','depth','alpha','combined','normalized_depth']]
    results += [run_case('covariance_input','combined',precomputed=True),
                run_case('scale_modifier_0.65','combined',modifier=.65),
                run_case('saturated_alpha','combined',saturated=True),
                run_case('upper_mean_clamp','combined',clamped_mean=True)]
    # An actual depth-only optimization must move depth and lower its loss.
    p=inputs();p={k:v.detach().requires_grad_(k=='xyz') for k,v in p.items()}
    optimizer=torch.optim.SGD([p['xyz']],lr=.02)
    losses=[]
    for _ in range(10):
        optimizer.zero_grad();out,_=cuda_render(p);loss=objective(out,'depth');loss.backward()
        assert p['xyz'].grad.norm()>0 and torch.isfinite(p['xyz'].grad).all()
        losses.append(float(loss.detach()));optimizer.step()
    assert losses[-1]<losses[0],losses
    report=dict(forward_bitwise_unchanged=unchanged,cases=results,depth_optimization_losses=losses)
    Path('temporary-build/orthographic-gradient-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
