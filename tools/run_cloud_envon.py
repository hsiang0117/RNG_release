"""Complete forward/deferred training, followed by streamed held-out evaluation."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('/myfiles/data/CloudDatasetUniform_envon'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metric_python', type=Path,
                        default=Path('/myfiles/projects/gaussian-splatting/.venv/bin/python'))
    args = parser.parse_args()
    output = args.output or ROOT / 'output' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_status.json').exists():
        raise FileExistsError(f'A run already exists at {output}')
    from scene.dataset_readers import fetchPly
    import numpy as np
    points = fetchPly(str(args.dataset / 'points3d.ply')).points
    scale = 1.0 / (float(np.linalg.norm(points, axis=1).max()) * 1.05)
    common = ['-s', str(args.dataset), '-m', str(output), '--eval', '--json',
              '--color_mlp', '--directional_light', '--in_channels', '16',
              '--max_training_images', '0', '--max_reso', '1024', '--resolution', '1',
              '--scene_scale', str(scale), '--loss', 'l1', '--rendertest_interval', '1000']
    commands = [
        ('forward', [sys.executable, '-u', 'train.py', *common, '--iterations', '30000',
                     '--save_iterations', '7000', '15000', '30000', '--densify_until_iter', '30000',
                     '--run_stage', 'forward']),
        ('deferred', [sys.executable, '-u', 'train.py', *common, '--iterations', '100000',
                      '--save_iterations', '40000', '60000', '80000', '100000',
                      '--load_pc', str(output / 'chkpnt30000.pth'), '--defer_shading',
                      '--shadow_map', '--shadow_grad', '--depth_mlp', '--depth_mlp_modifier', '1.0',
                      '--encoding_levels_each', '2', '--encoding_levels_shadow', '8',
                      '--crop_pc', '1.0', '--run_stage', 'deferred']),
        ('render_test', [sys.executable, '-u', 'tools/render_cloud_test.py', str(output)]),
        ('evaluate_test', [str(args.metric_python), '-u', 'tools/evaluate_saved_images.py', str(output)]),
    ]
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    setup = dict(dataset=str(args.dataset), image_resolution=[1024,1024], train_frames=1308,
                 test_frames=152, initial_points=len(points), scene_scale=scale,
                 initial_ply_sha256=sha(args.dataset/'points3d.ply'),
                 transforms_train_sha256=sha(args.dataset/'transforms_train.json'),
                 transforms_test_sha256=sha(args.dataset/'transforms_test.json'),
                 stages=dict(forward_steps=30000, deferred_steps=70000, final_global_iteration=100000),
                 adaptation='True directional sun, common camera/point scaling, original RGB and exact split',
                 renderer_changes='Python directional input/cue wiring and safe masked divisions; CUDA unchanged for this task',
                 loader='6GiB host uint8 RGB, 32MiB CUDA image, 96MiB CUDA ray, 12GiB host shadow caches; two decode workers',
                 optimizer_fix='Crop retains optimizer parameter identity and state',
                 preview='Four fixed held-out camera/sun combinations every 1000 global steps',
                 metric_python=str(args.metric_python),
                 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),
                 commands={name: command for name, command in commands})
    (output/'experiment_setup.json').write_text(json.dumps(setup, indent=2))
    # Archive precisely the source tracked by the run's commit.
    with (output/'source_snapshot.tar').open('wb') as f:
        subprocess.run(['git','archive','HEAD'],cwd=ROOT,stdout=f,check=True)
    status = dict(output=str(output), runner_pid=os.getpid(), phase='starting',
                  start_time=datetime.now().astimezone().isoformat(), stages={}, exit_code=None)
    start = time.monotonic()
    def write_status():
        tmp = output/'run_status.json.tmp'
        tmp.write_text(json.dumps(status,indent=2)); os.replace(tmp,output/'run_status.json')
    write_status()
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MPLBACKEND='Agg')
    try:
        for name, command in commands:
            status['phase'] = name
            began = time.monotonic()
            with (output/f'{name}.log').open('w') as log:
                child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT)
                status['active_pid']=child.pid; write_status()
                rc = child.wait()
            status['stages'][name]=dict(exit_code=rc,elapsed_seconds=time.monotonic()-began)
            write_status()
            if rc:
                raise RuntimeError(f'{name} exited with code {rc}; see {output/name}.log')
            if name in ('forward','deferred'):
                shutil.copyfile(output/'cfg_args',output/f'cfg_args_{name}')
        status['phase']='completed'; status['exit_code']=0
    except BaseException as exc:
        status['phase']='failed'; status['exit_code']=1; status['error']=str(exc)
        raise
    finally:
        status['active_pid']=None
        status['end_time']=datetime.now().astimezone().isoformat()
        status['elapsed_seconds']=time.monotonic()-start
        write_status()


if __name__ == '__main__':
    main()
