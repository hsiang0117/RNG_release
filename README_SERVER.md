# RNG 服务器环境

服务器目录：`/myfiles/projects/RNG_release`，独立环境：`.venv`。Python 3.12、PyTorch 2.11.0+cu128、CUDA Toolkit 12.8，RTX 4090 编译架构为 8.9；固定依赖见 `requirements-server-lock.txt`。

```bash
cd /myfiles/projects/RNG_release
source tools/env_server.sh
python train.py --help
```

重建环境：`bash tools/install_server.sh`。只编译扩展：`bash tools/build_server.sh`。

保留的源码修复包括标准 C++ 头文件补齐，以及正交光栅器的位置／协方差反向、深度和 alpha 梯度接口修复；正交前向计算保持原样。GPU 前后向、独立参考、有限差分和短优化此前均已验证。

```bash
source tools/env_server.sh
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python tools/verify_environment.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python tools/verify_orthographic_gradients.py
```

`dependency-cache` 保存包与扩展 wheel；安装脚本还会使用 Vol3DGS／EVER 的包缓存。`environment-backups` 保存修复前源码。`tools` 保存构建和验证脚本。`temporary-build` 仅存放可清理的编译／验证输出，安装或验证时生成。

## Cloud env-on 完整训练

```bash
source tools/env_server.sh
python tools/run_cloud_envon.py
```

默认读取 `/myfiles/data/CloudDatasetUniform_envon` 的原始 JSON 划分，使用全部 1308 张训练图和 152 张测试图，以 1024×1024 原始分辨率训练，不使用 README 示例中的 1000 张子集和 512 分辨率。初始点云必须是数据目录中已有的 `points3d.ply`；不会生成或覆盖随机点云。

训练沿用作者的 forward/deferred 两阶段流程：第一阶段 30000 步并增密至 30000，第二阶段从第一阶段点云检查点继续至全局 100000 步，即另训练 70000 步。第二阶段重新初始化不同输入维度的颜色 MLP 和深度 MLP，保留作者的编码配置、阴影梯度及每 10 个全局步骤重建阴影缓存的条件。方向光输入由帧中的世界坐标 `sun_direction` 给出，方向指向太阳，光照强度输入为常数 1；颜色 MLP 可从太阳方向拟合 env-on 图像。

方向光训练接通已有的正交阴影光栅器，不以远距离点光源近似太阳。点云和相机位置按相同系数缩放，将初始点云放入半径 1/1.05 的球中，适配该光栅器固定的投影范围，透视投影保持一致。缩放系数、初始 PLY 和划分 JSON 的 SHA256 都保存到实验目录。正交阴影提示使用光线方向上的投影距离。此次数据适配没有修改 CUDA；此前安装时完成的正交反向修复仍保留。

图像按需解码，CPU RGB uint8 缓存上限 6 GiB，两个解码线程预取；CUDA 图像和共享相机射线分别限制为 32 MiB 和 96 MiB。CPU 阴影缓存上限 12 GiB，足以保留本数据集全部 1460 张图的两张 float32 阴影图，因而保留原始缓存命中和重建条件。超过容量的其他数据集会按 LRU 淘汰并重算。摄像机对象只常驻小矩阵和元数据。

Python 侧还修正了点云裁剪后 Adam 参数未同步、掩码分支中的零除和 PyTorch 2.6+ 完整训练检查点加载兼容性。损失只反传一次，不再保留已使用的计算图；损失曲线由每 10 步写图改为每 1000 步写图。训练损失、增密、剪枝、学习率、MLP 和网络精度设置保持原配置。

输出为 `output/YYYYMMDD_HHMMSS`。`rendertest/` 每 1000 个全局步骤保存四个固定测试组合的 2×2 图像；两行分别为 camera 01/10，两列为 sun 07/37，全部来自测试集。GT 及来源在 `ground_truth.png` 和 `views.json`。两阶段参数、日志、统计分别保存，`run_status.json` 记录当前阶段、进程和退出码。

训练完成后自动重载最终 PLY、颜色 MLP 和深度 MLP，逐张渲染全部 152 张测试图，再通过原版 3DGS 环境计算 PSNR、SSIM 和 LPIPS-VGG，保存 `metrics.json` 和 `per_view_metrics.json`。这一步只读取测试 GT 计算指标。可用 `--metric_python` 指定其他已安装 `lpips` 的 Python 环境。

```bash
python tools/render_cloud_test.py output/YYYYMMDD_HHMMSS
/myfiles/projects/gaussian-splatting/.venv/bin/python tools/evaluate_saved_images.py output/YYYYMMDD_HHMMSS
```

统一指标取逐图平均，区分 96 张未见太阳方向与 56 张已见太阳方向下的新组合。`metrics.json` 采用保存 PNG 的 uint8 口径；渲染器 float32 PSNR/SSIM 另外保存在 `test/ours_100000/metrics_float.json`。原生 `render.py` 也改为逐图计算，避免整批 GT 和渲染常驻显存；其 LPIPS-Alex 与统一报告的 VGG 不应混用。

检查真实数据、两阶段前后向、缩放投影、GT 像素和裁剪优化器：

```bash
python tools/verify_cloud_pipeline.py
```
