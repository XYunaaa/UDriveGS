# scene2 迁移与三组实验交接

更新：2026-11-25。负责人小云。Git仓库：https://github.com/XYunaaa/UDriveGS.git 。训练当前停止；本文的训练命令仅供在新机器上明确执行。

## 1. 已完成什么

scene2是室内门厅，不是scene1的室外道路。原始输入来自19段视频。固定训练几何来自 `data/scene2_splatfacto/colmap/host_resume/sparse/0`；同一组图和位姿用于本次三组控制，不重新抽帧或重跑COLMAP。

| 划分 | 视频/数量 | 用途 |
|---|---|---|
| train | 其余17段视频，474张 | 拟合模型及可选深度目标 |
| val | `IMG_9765`，41张 | 选择配置 |
| test | `IMG_9780`，30张成功定位 | 最后测试；原始38个候选中8张未成功定位 |

原始模型有2,311,188个Gaussian。四版本在同一41张验证图的完整分辨率原生评测：

| 版本 | 保留Gaussian | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|---:|
| 原模型 | 2,311,188 | 22.1526 | 0.82007 | 0.24299 |
| alpha≥0.01 | 2,067,849 | 22.1499 | 0.82004 | 0.24305 |
| alpha≥0.02 | 1,853,409 | 22.1244 | 0.81985 | 0.24335 |
| alpha≥0.05 | 1,499,026 | 21.9375 | 0.81750 | 0.24545 |

结论：简单不透明度过滤没有提高整体画质，虚拟视角仍有墙面/玻璃门拉伸。0.01只可视为基本保持画质的轻量副本。完整数值、逐图分数和来源签名见 `metadata/scene2/cleanup-validation.json`。

深度准备已在原机器运行：474训练图中460张接受、14张拒绝。深度模型来源/权重SHA已固定，但深度正则训练未完成。**这三组控制也尚无新训练结果，不能宣布最佳配置。**

## 2. 哪些文件需要搬

本轮只需要 `transfer/scene2-rgb-inputs-20261125/`：

- `manifest.json`：与Git内 `metadata/scene2/inputs.json` 相同的权威输入清单。
- `scene2-inputs.tar.part-000`、`scene2-inputs.tar.part-001`：每块不超过512MiB，总输入约1.066GB（十进制）。

还原后包含545张原始PNG字节、`transforms.json`、稀疏初始化 `sparse_pc.ply`、`provenance.json` 和三个冻结COLMAP二进制文件，共551个文件。唯一的元数据迁移改动：provenance里的稀疏模型位置改为项目相对路径；原始provenance哈希另有记录。图片、相机参数和点云数值均不转换。

**不用搬：**源视频、抽帧中间目录、COLMAP数据库、旧optimizer checkpoint、原始大PLY、清理副本、渲染视频、原环境或CUDA编译缓存。Git中只含清单，**clone不会自动得到场景数据**。

从原机器复制到新机器，例如：

```bash
# 在原机器执行；自行替换新机器地址与接收路径。
rsync -avP transfer/scene2-rgb-inputs-20261125/ \
  USER@NEW_HOST:/mnt/transfer/scene2-rgb-inputs-20261125/
```

需要重新打包时，在保留原始provenance的源机器选择一个不存在的输出目录。还原后的provenance已有路径迁移，若从新机器再次打包，需要重新审核并版本化输入清单，不能假定其源文件审计哈希不变。

```bash
python3 code/scene2_transfer.py pack --bundle transfer/scene2-rgb-inputs-copy
python3 code/scene2_transfer.py verify --bundle transfer/scene2-rgb-inputs-copy
```

工具不会覆盖已有包。校验既检查各分块，又逐个检查归档文件SHA256；恢复时先校验全部输入，再检查目标路径。相同文件可复用，有冲突则停止，不覆盖原数据。它拒绝路径逃逸、归档链接和目标符号链接。解包时临时拼接约1.1GB归档，需要额外临时空间。

## 3. 新机器环境

已测环境：Linux x86_64、RTX4090D24GB、驱动550.120、CUDA Toolkit12.4、GCC/G++12、Python3.10、COLMAP3.9.1、PyTorch2.4.0+cu124、Nerfstudio1.1.5、gsplat1.4.0、NumPy1.26.4。完整审计见 `metadata/environment/`。

脚本不再写死本机Conda目录或4090架构。默认由PyTorch识别可见GPU；需要时可自己设置 `TORCH_CUDA_ARCH_LIST`。其他GPU仍需与这版PyTorch/CUDA兼容，尚未实测。新机器建议预留至少30GB可用磁盘；本机环境约16GB，另有数据、新模型、优化器瞬时存储和编译缓存。

```bash
git clone https://github.com/XYunaaa/UDriveGS.git
cd UDriveGS
git config core.hooksPath .githooks

# Debian/Ubuntu；需要系统管理员权限时自行运行。
sudo apt-get update
sudo apt-get install -y gcc-12 g++-12 ffmpeg rsync

# 先安装并初始化Conda，再在没有ROS/CARLA注入的新终端执行。
export CONDA_EXE="$(conda info --base)/bin/conda"
bash code/setup_splatfacto.sh
```

安装脚本创建项目内 `.envs/splatfacto`，安装固定核心版本，执行 `pip check`、真实PyTorch与gsplat的CUDA前后向检查。首次JIT编译可能需要几分钟。不要复制旧环境或旧缓存，也不要通过 `-allow-unsupported-compiler` 绕过GCC兼容性检查。

`pip-snapshot-linux-cu124.txt` 是审计记录，已去掉本地构建URI；安装入口仍是setup脚本与constraints，不能把快照当成跨系统通用锁文件。

## 4. 恢复与预检（不训练）

```bash
python3 code/scene2_transfer.py verify \
  --bundle /mnt/transfer/scene2-rgb-inputs-20261125
python3 code/scene2_transfer.py unpack \
  --bundle /mnt/transfer/scene2-rgb-inputs-20261125

SCENE2_RESULT_DIR="$PWD/results/splatfacto/scene2_rgb_v1" \
  bash code/run_scene2_quality.sh rgb --check-only

PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/code" \
  .envs/splatfacto/bin/python -m unittest discover -s code -p 'test_*.py'
```

预检应报告474/41/30、`training_started: false`。配置从 `configs/scene2/splatfacto_big.yml` 加载，数据和结果路径在当前checkout内解析；目录不必叫UDrive-GS，也不必放在 `/home/luciana`。

## 5. 在新机器明确启动三组控制

```bash
# 可先进入 tmux，避免SSH断开影响训练。
SCENE2_RESULT_DIR="$PWD/results/splatfacto/scene2_rgb_v1" \
  bash code/run_scene2_quality.sh rgb
```

队列串行执行 `pose → scale → prune`，每组从同一初始化重新训练30k、seed42，然后评测相同41张验证图。变更分别只有相机优化、尺度正则、剪枝alpha阈值；完整基础参数沿用真实 `splatfacto-big` 基线。已有原生精度/缓存设置保持一致。

每次训练至少等待12GiB空闲显存；不终止其他任务。GPU等待不代表失败。缺少CUDA会明确退出，不退回CPU训练。输入、代码哈希变化时拒绝复用同一个结果目录。已完成阶段可跳过；存在部分训练目录却没有最终checkpoint时会停下，先检查，或使用新的结果目录，避免误把半程结果当成完成。

输出在指定目录下：

| 文件/目录 | 内容 |
|---|---|
| `host-queue.log` / `status.json` | 运行日志、当前阶段 |
| `manifest.json` / `rgb_contract.json` | 输入和执行代码指纹 |
| `runs/<pose或scale或prune>/...` | 各自配置、TensorBoard、最终checkpoint |
| `exports/<实验>/splat.ply` | 原生Gaussian导出和校验manifest |
| `evaluation/<实验>/val.json` / `val/` | 完整分辨率指标、固定视图和虚拟视角 |
| `rgb_comparison.json` | 三组验证结果汇总；仍需视觉复核 |

仅保存新训练的最终checkpoint；完成后移除optimizer/scheduler/scaler状态，逐个验证pipeline张量不变。这些新checkpoint用于评测和导出，**不能用于恢复优化器继续训练**。失败的部分训练不会自动删除。

虚拟视角偏移±0.03/±0.06是归一化SfM单位，不是米；没有真值，不计算其PSNR。对比时重点查看墙面/玻璃、显示屏边缘、细杆、浮点与孔洞。RGB队列不自动使用测试集选配置，也不自动宣布视觉最佳。原模型验证指标在Git内可供参考，不能替代新机器上对原模型重新评测。

## 6. 后续深度与完整队列

`rgb`模式不克隆DepthAnything、不下载其权重，也不启动深度训练。以后明确要做深度时，`scene2_depth.py` 会校验 `metadata/third-party.json` 记录的官方代码版本与权重SHA，只使用训练图和对应COLMAP观测拟合目标。

深度代码首次获取时必须处于记录的提交，不能随意用上游最新版本代替。例如在项目根目录执行 `git clone https://github.com/DepthAnything/Depth-Anything-V2 code/third_party/Depth-Anything-V2`，然后在该目录 `git checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`；权重由代码从官方地址获取并校验。首次LPIPS模型初始化也可能需要联网下载其预训练网络权重。

`run`模式是原先“清理 → 三组RGB → 深度 → 验证集选择 → 最终测试”的完整队列，另需原模型的最终checkpoint和原始Gaussian PLY，放在原有项目相对路径，并使用新的结果目录；**本次最小输入包没有这些大文件，不能直接运行run模式**。原机已生成的深度缓存带有绝对路径与实现指纹，不直接搬来假装缓存有效；必要时在新机器重新准备。

## 7. 已知问题与验证边界

原机第一组新训练在导入PyTorch时退出，还没创建训练目录，报错在标准库 `posixpath.join`：`startswith ... NoneType`。本次配置比对时，主进程确认配置一致，但后台TorchInductor子进程再次出现 `sre_parse.py` 的 `Tokenizer` 类型异常并退出。独立复核还在 `test_export_camera_paths` 导入SciPy时遇到 `inspect.Signature` 类型异常；另一轮47项CPU测试全部通过，但最终复跑仍在 `test_scene2_depth` 导入TorchInductor时失败（报告40项、1项导入错误，已设置编译线程为1）。完整末次日志见 `metadata/environment/latest-cpu-suite.txt`，当前全量检查不能视为通过。原因未确定；没有用改系统库或盲目升级依赖来掩盖问题。本次迁移保留这一记录，**没有证明三组完整训练已经跑通**。

新机器先通过setup的CUDA检查和CPU套件，再开始30k。若导入阶段再次失败，保留 `host-queue.log`，并在同一干净环境采集：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  .envs/splatfacto/bin/python -X faulthandler -c \
  'import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())'
```

本次交接验证覆盖：CPU测试、配置数值与原基线一致、分块/逐文件哈希、异地目录还原、输入预检、Git忽略和大小边界。没有在当前机器重新安装整个环境，也没有启动GPU训练。

## 8. 之后怎样保存工作

- 代码/参数变更提交Git；指标和小型运行说明人工整理到 `metadata/`，不要把整份results复制进去。
- 数据原件独立备份。迁移时只按输入清单打包，不执行删除源数据的同步命令。
- 模型、checkpoint和预览媒体留在 `results/`，需要共享时另行传输；记录对应Git提交、输入指纹和运行配置。
- 第三方仓库和权重按固定版本重新获取，不复制进Git。
- 如果Git钩子拦截大文件，先取消暂存并移回数据/结果区域。只删除工作目录文件不能清除已经提交的历史大对象；不要为解决上传限制直接force push。
