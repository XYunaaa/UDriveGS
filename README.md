# UDriveGS

小云 / B 路线：视频抽帧 → COLMAP 位姿与稀疏点云 → Splatfacto → 渲染评测。当前交接重点是 **scene2 的三组独立 30k 对照**。

**当前状态：训练已按要求停止。本次只整理代码和迁移材料，没有启动新训练。** scene2 为室内门厅；已有原始30k模型，低不透明度清理未改善整体验证指标。深度目标已在原机器生成，460/474张合格；本轮三组新训练尚未完成。

## 在另一台电脑继续

详细步骤、数据传输和已知问题见 **[scene2 交接文档](docs/HANDOFF_SCENE2.md)**。

```bash
git clone https://github.com/XYunaaa/UDriveGS.git
cd UDriveGS
git config core.hooksPath .githooks

# Linux + NVIDIA；先安装 Conda、系统 gcc-12 / g++-12，进入不带 ROS 的终端。
export CONDA_EXE="$(conda info --base)/bin/conda"
bash code/setup_splatfacto.sh

# 把另行传输的两个分块及 manifest.json 放在 /path/to/scene2-inputs/。
python3 code/scene2_transfer.py unpack --bundle /path/to/scene2-inputs

# 只校验，不启动训练或渲染。
SCENE2_RESULT_DIR="$PWD/results/splatfacto/scene2_rgb_v1" \
  bash code/run_scene2_quality.sh rgb --check-only

# 准备好后，在新机器明确执行此命令才会开始三组训练。
SCENE2_RESULT_DIR="$PWD/results/splatfacto/scene2_rgb_v1" \
  bash code/run_scene2_quality.sh rgb
```

| 实验 | 相对原始 splatfacto-big 只改这一项 |
|---|---|
| `pose` | 相机优化 `off → SO3xR3` |
| `scale` | `use_scale_regularization=False → True` |
| `prune` | `cull_alpha_thresh=0.005 → 0.02` |

统一545张固定输入：474训练 / 41验证 / 30已定位测试，seed42、30k、原始分辨率、同一稀疏点初始化。RGB队列只执行三组训练和验证评测，**不会自动开始深度训练，也不需要旧模型、旧checkpoint或原始视频**。

## 代码位置

| 文件 | 用途 |
|---|---|
| `code/setup_splatfacto.sh` / `code/splatfacto-constraints.txt` | 环境安装、版本约束和 CUDA 前后向检查 |
| `configs/scene2/splatfacto_big.yml` | 从真实基线导出的完整配置，项目路径可迁移 |
| `code/scene2_quality.py` / `code/run_scene2_quality.sh` | 清理、三组控制、原生评测；另保留完整深度队列模式 |
| `code/scene2_transfer.py` | 生成、校验、恢复最小输入包；只用 Python 标准库 |
| `code/scene2_depth.py` | 可选的训练图深度对齐与正则；三组RGB实验不依赖它 |
| `code/prepare_multivideo.py` / `localize_heldout.py` / `finish_scene2.py` | 从视频和COLMAP重新准备输入的前序工具 |
| `code/export_checkpoint_cpu.py` / `export_camera_paths.py` | 模型与相机路径导出 |
| `code/prepare_sugar_scene1.py` / `setup_sugar.sh` | 先前SuGaR探索，不是三组实验前置条件 |
| `metadata/scene2/` | 输入SHA256清单、清理指标和配置来源 |
| `metadata/environment/` | 已测版本和环境审计快照 |

`docs/superpowers/plans/` 保留历史实施记录；当前状态以交接文档为准。`code/vendor/splat/` 是既有预览器，附其许可证；大型第三方仓库不纳入本仓库，版本见 `metadata/third-party.json`。

## 文件管理规则

- **Git：**第一方代码、配置、交接文档、小型JSON/TXT记录。
- **独立数据传输：**`transfer/` 内的输入分块；原始视频继续放 `data/scene2/`，本轮无需迁移。
- **留在实验机器：**`results/`、`.envs/`、`.cache/`、`code/third_party/` 和含视频/PDF/压缩包的 `doc/`。
- 根目录采用允许列表；PLY、checkpoint、模型权重、媒体和压缩包即使放进代码目录也默认忽略。
- 提交/推送钩子检查暂存文件和历史大对象：单文件最多5MiB，当前受管快照最多20MiB。克隆后需显式启用钩子；不要用 `git add -f`、`--no-verify` 绕开边界。数据分块不是Git LFS文件，不推往GitHub。

运行CPU检查：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/code" \
  .envs/splatfacto/bin/python -m unittest discover -s code -p 'test_*.py'
python3 code/check_git_size.py
```

新机器尚未完成真实GPU训练验证。本机曾在新训练启动时遇到间歇性的PyTorch导入异常，详情与采集方式见交接文档；不能将已有CPU测试通过等同于完整训练可复现。
