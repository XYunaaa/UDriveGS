# Scene1 街区视频 Splatfacto 重建实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 状态：已开始执行；Task 1 抽帧与划分已完成，安装依赖及 GPU 执行等待本机终端。进展记录见 `results/splatfacto/scene1/progress.md`。

**Goal:** 小云使用已有街区视频，在 Linux 上得到可复现的静态 3DGS 基线、留出视角评测和渲染演示。

**Architecture:** FFmpeg 固定候选图像和时间清单；COLMAP 从训练图像恢复相机与稀疏几何；Nerfstudio 官方转换工具准备数据，Splatfacto 从训练稀疏点初始化。验证和测试图像仅定位到冻结的训练地图，分别用于开发检查与最终评测。

**Tech Stack:** FFmpeg、COLMAP、Nerfstudio Splatfacto、PyTorch、gsplat；动态掩码工具在环境核验后选择。

**Spec:** 用户已选定“视频抽帧 → COLMAP 位姿与稀疏点云 → Splatfacto 训练 → 渲染评测”。项目依据见 `doc/远程驾驶_静态场景重建_四周双路线探索计划.pdf` 第 2 页；以下参数为本次建议。

## 全局约束

- 原视频 `data/scene1` 保持不变；它是 HEVC 编码的 MP4，120.9 秒、960×544、3626 帧、约 30 FPS，已完整解码通过。
- 首轮重建范围是拍摄路径附近可见的道路、墙体、建筑和固定设施；不承诺完整街区、未观测背面或任意视角。
- 首版不需要预测深度、稠密 MVS、网格或算法改造；保留原始 COLMAP 资产以供未来 Postshot 对照。
- 使用既有工具，不搭建自定义训练框架。新增辅助脚本只承担工具未覆盖的帧清单、数据审计或静态 ROI 评测，每个非平凡脚本带一个小型自检。
- COLMAP 与 Nerfstudio 均以黑色 0 表示忽略，白色 255 表示保留。掩码与对应图像严格对齐；不通过把 RGB 涂黑实现剔除。
- 图像划分在训练前冻结。验证和测试图像不得参与训练地图建图、初始化点云的几何或颜色估计、训练损失或外观参数拟合。
- 单目 SfM 具有尺度不确定性。保留坐标转换、重定向和归一化记录；没有独立尺度约束时不报告米制准确率。
- 测试图像只用于最终报告。两秒分块来自同一次拍摄，只能评价同次采集中的留出视角，不能据此声称跨日期泛化。
- 重跑写入新运行目录，不覆盖旧模型、冻结划分或原视频。正式结果绑定数据校验值、软件版本、配置、随机种子和 checkpoint。

## 方案选择

直接使用 `ns-process-data video` 最省操作，但不便逐帧核对筛选与严格控制初始化数据。完全手写坐标转换会增加错误来源。

本计划建议显式抽帧、保留 COLMAP 分步产物，再复用官方数据转换和训练工具。先用训练子集完成未加掩码的流程检查，再做带掩码的正式静态基线；流程检查结果不用于路线质量结论。

## 文件与存储

执行阶段按需生成下列产物，本次只创建本计划：

```text
data/scene1                                      原始视频
data/scene1_splatfacto/
  manifest.json                                 原视频哈希、PTS、抽帧与筛选配置
  frames.csv                                    原视频帧号、PTS、图像文件名、筛选原因
  splits.json                                   候选训练、验证、测试名单
  images/                                       固定 RGB 图像
  masks/                                        逐帧有效区掩码
  colmap_masks/                                 COLMAP 所需命名形式
  colmap/train/                                 训练数据库与冻结稀疏模型
  colmap/localized/                             追加留出相机后的独立模型副本
  processed/                                    transforms.json 与训练稀疏点云
results/splatfacto/scene1/
  environment.txt                               版本、GPU、配置与环境检查
  runs/smoke/                                   短流程训练
  runs/static_baseline/                         正式训练配置、日志、checkpoint
  evaluation/                                   逐图指标、配对图、覆盖与失败记录
  renders/                                      固定相机路径和演示视频
  report.md                                     最终结果、成本与限制
code/                                           仅在现有工具不足时放必要辅助脚本
```

项目分区当前仅剩约 7.8 GB，`/tmp` 所在分区约剩 160 GB。建议持久位置为环境与产物合计预留至少 30 GB，实际需求在安装和短训练后修正；这是容量预算，不是实测占用。`/tmp` 可承载编译缓存和临时文件，不能作为唯一交付存储。若暂在 `/tmp/udrive-gs-scene1/` 运行，必须有可用持久位置备份原始清单、地图、最终 checkpoint、配置和报告后才能认定交付完成。不清理其他项目文件。

## Review Focus

1. 视频可能存在非恒定时间间隔；抽帧映射使用真实 PTS，不能将输出序号除以 30 当作源时间。
2. 留出图像加入联合 SfM 或点云颜色平均会泄漏评测信息；训练模型先冻结，后续只定位新相机，初始化继续引用训练地图。
3. 掩码极性、文件名或去畸变错位会留下动态残影或删除静态结构；通过可视叠加及尺寸、值域断言检查。
4. COLMAP 可能生成多个不相连子模型；报告全体候选帧的注册结果，不能默默只保留最好的一块。
5. 默认图像指标可能仍包括动态区域；保留完整图指标，并用单独定义的静态 ROI 指标评价背景质量。

## Task 0 核验环境与资源

**Consumes:** 当前 Linux 主机与已有 Conda 环境。
**Produces:** `environment.txt`、确定的持久数据位置、可执行的工具与冻结版本。

- [ ] 排查当前会话 GPU 可见性。已观察到 NVIDIA PCI 设备和已加载的 550.120 内核模块，但 `nvidia-smi` 失败；这尚不能区分沙箱可见性、设备权限与驱动问题。先只读检查，不直接重装驱动。
- [ ] 在将实际执行训练的环境运行 `nvidia-smi` 和 PyTorch CUDA 张量运算，再验证一次 gsplat 前向与反向。通过条件是能识别 GPU、完成运算、无 CUDA 错误。
- [ ] 记录显存与空闲资源；默认 Splatfacto 文档估算约 6 GB 显存，实际按场景和安装版本测量，不据该数值保证能运行。[S1]
- [ ] 核对已有环境。已发现 `drivestudio` 含 torch 2.4.0+cu124、gsplat 1.3.0 和 nvcc，但没有发现可执行的 `ns-train` 或 `colmap`。这些信息只用于评估可复用性，不在原环境中直接升级依赖。
- [ ] 如需安装，使用独立项目环境；依据 GPU、驱动和 Nerfstudio 指定版本选择兼容的 Python、PyTorch、CUDA 与 gsplat，保存实际版本或提交。执行前读取该版本 CLI 帮助，不照搬不同 COLMAP 版本的选项。
- [ ] 检查持久磁盘余量后再安装；确认 `ffmpeg`、`ffprobe`、`colmap`、`ns-process-data`、`ns-train`、`ns-eval`、`ns-render`、`ns-export` 可调用。

**完成标准：** CUDA 与 gsplat 运算通过，工具版本固定，存储位置落实。GPU 未恢复时可以推进抽帧及 CPU SfM，但不能开始 Splatfacto 训练。

## Task 1 抽帧与冻结数据划分

**Consumes:** `data/scene1`。
**Produces:** `images/`、`frames.csv`、`manifest.json`、`splits.json`。

- [x] 记录视频 SHA256、分辨率、流信息与逐帧 PTS。
- [x] 以 0.5 秒为目标采样间隔，在每个采样时刻选最近的实际源帧并去重；保存源帧号与真实 PTS。约得到 242 张候选图像，确数以实际选择为准。保存原尺寸 PNG，不放大图像。
- [x] 按真实时间组成连续 2 秒块，令 `block_id=floor(PTS/2)`；`block_id % 10 == 8` 为验证，`==9` 为测试，其余为训练。预计接近 80/10/10，同一块不得拆到不同集合。
- [x] 先冻结划分，再在各集合内按清晰度、严重遮挡或解码异常做筛选；保留每张拒绝原因，不将困难验证或测试帧改到训练集。筛选时不查看模型结果。
- [x] 人工检查训练帧联系图、转弯覆盖与模糊情况。若 2 FPS 不能支持连续匹配，发布新的抽帧版本，在同一时间块划分规则内增加训练采样，不随意修改测试区域。
- [x] 为辅助脚本设置小型自检：非均匀 PTS 输入得到唯一源帧；三集合交集为空；所有候选帧恰有一个去向；图像数、尺寸和清单相符。

**完成标准：** 每张图可追溯原视频，划分与拒绝原因可复核。短流程检查只从训练帧取前 30 秒内的子集，不提前用最终测试帧训练。

## Task 2 COLMAP 位姿与训练稀疏点云

**Consumes:** 训练图像名单；正式静态版本另含 Task 3 掩码。
**Produces:** `colmap/train/` 中的数据库、`cameras.bin`、`images.bin`、`points3D.bin`，以及注册与误差记录。

- [ ] 未加掩码的短流程先取训练子集：SIFT 特征 → 时序邻近匹配 → incremental mapper → 训练图像 bundle adjustment。首轮建议时序邻域覆盖前后约 5 秒的训练观测；回访区域按需要补跨时间匹配，不预先下载词袋库。
- [ ] 从简单相机模型开始；无鱼眼标定依据时不将该视频视为鱼眼输入。同一镜头与固定焦距成立时共享内参；若视频存在变焦、强电子防抖或内参变化，先分段检查。
- [ ] 正式静态版本使用 Task 3 掩码重新提取训练特征并重建，不能沿用已含动态特征的旧数据库。
- [ ] 统计候选、拒绝、已注册、未注册图像数，所有子模型及其覆盖时段，稀疏点数、重投影误差分布和轨迹。建议首轮检查线为主要连通模型注册不少于 90% 的合格训练帧、平均重投影误差不高于 2 像素；低误差不代替轨迹与结构检查。
- [ ] 检查道路、墙体和建筑是否连续，轨迹有无跳变；转弯或末段未注册时不能只报整体平均值。模型不连通则先检查模糊、遮挡、重叠与匹配，再增加候选帧或缩小试点。
- [ ] 冻结训练模型并保存校验值。在副本数据库中匹配验证与测试图像到训练地图，以 `image_registrator` 定位留出相机；此后不运行联合 bundle adjustment、mapper 或新点三角化。[S2]
- [ ] 检查训练相机内外参与训练点 XYZ、RGB 在留出定位前后一致。定位失败的留出帧列入失败清单，不删除原评测分母。

**完成标准：** 得到结构合理的训练地图、留出相机定位与全部失败记录。缺少可靠留出位姿时停止定量新视角评分，不能用插值轨迹代替真实评测位姿。

## Task 3 动态掩码与 Nerfstudio 数据转换

**Consumes:** 固定图像、训练地图、留出相机、车辆与行人掩码。
**Produces:** `processed/transforms.json`、只来自训练地图的初始化点云及对齐掩码。

- [ ] 在未加掩码短流程通过后，选择环境中可用的分割工具生成车辆、行人和不希望固化的临时目标掩码；优先复用工具，记录权重、版本和人工修订。不要求先训练新的分割模型。
- [ ] 正式静态数据使用白色保留、黑色忽略的单通道二值图；无排除目标的图像也提供全白掩码。COLMAP 的 `frame_000001.png` 对应掩码名为 `frame_000001.png.png`；Nerfstudio 的每帧 `mask_path` 显式指向对应文件。[S2][S3]
- [ ] 抽查车辆边界、行人、漏检与误删除；RGB 与掩码若需去畸变，使用相同相机映射，掩码采用最近邻采样。保存叠加图。
- [ ] 复用 `ns-process-data images` 的已有 COLMAP 模型转换能力，关闭再次运行 COLMAP 和多余降采样；具体 `--skip-colmap`、模型路径等参数以冻结版本帮助为准。审计转换前后的文件名映射，避免官方复制重命名破坏位姿对应。[S4]
- [ ] `transforms.json` 写入明确的 `train_filenames`、`val_filenames`、`test_filenames`，指向实际存在且注册成功的图像。未定位图像继续保留在原始 `splits.json` 与失败统计中。
- [ ] 初始化 PLY 必须取冻结训练模型的 XYZ 与 RGB；审计 `ply_file_path`，不能错误导入联合建图点云。保留 Nerfstudio 对 COLMAP 坐标所作的转换，避免手动重复翻转轴。[S3][S5]
- [ ] 执行数据自检：缺图、缺掩码或尺寸不符即失败；全白掩码保留全部像素，已标记目标为黑色；训练、验证、测试路径互斥且名单有效；点云非空、相机参数有限、变换合法。

**完成标准：** 数据加载器实际读到指定划分、掩码和训练点初始化。掩码降低动态监督影响，但未拍到的背景仍可缺失。

## Task 4 Splatfacto 短流程与正式训练

**Consumes:** 已验证的 Nerfstudio 数据、CUDA 环境。
**Produces:** 两次分开保存的运行与正式 checkpoint。

- [ ] 先用 Task 1 的训练子集进行 2,000 步未加掩码短训练；确认 loss 有限、能保存并恢复 checkpoint、viewer 或离线渲染能显示场景，采集显存与磁盘占用。此轮不用于完整度或质量结论。
- [ ] 完成 Task 3 后，从其训练稀疏点重新初始化正式 `splatfacto`。首轮建议 30,000 步、种子 42、输入保持 960×544，不启用 `splatfacto-big` 或新增算法；Splatfacto 的内置渐进分辨率按冻结版本默认并写入配置。[S1]
- [ ] 核查有效配置确实加载 3D 点，未意外退回随机初始化；相机优化首轮关闭，验证和测试位姿不得被外观训练更新。
- [ ] 只用验证集检查收敛和配置，测试集最后一次性评测。首轮正式交付取固定最终 30,000 步 checkpoint，不从测试分数选择版本。
- [ ] 记录启动、预处理、CUDA 首次编译与实际训练耗时，最终高斯数、峰值显存、checkpoint 大小。首编译成本单列。
- [ ] 正式训练中出现 NaN、OOM 或明显结构错误就停止定位原因，保留日志。降分辨率等变更另建运行并记录质量代价，不覆盖基线配置。

命令模板如下，需先在实际版本核对 `--help`。`DATASET_DIR` 和 `RUN_DIR` 是确定存储位置后的路径：

```bash
ns-train splatfacto --data "$DATASET_DIR" --output-dir "$RUN_DIR" --max-num-iterations 30000
```

**完成标准：** 固定最终模型可加载、图像有限且能复现渲染，日志绑定配置与版本。

## Task 5 留出视角评测与渲染交付

**Consumes:** 最终 checkpoint、固定测试名单和位姿、测试静态 ROI、固定演示路径。
**Produces:** `evaluation/`、`renders/`、`report.md` 与可加载模型。

- [ ] 使用 `ns-eval` 保存完整图 PSNR、SSIM、LPIPS 及配对渲染。这些指标包含真实测试图像中的动态目标影响，单独标为完整图结果。[S6]
- [ ] 另算静态有效 ROI 指标：先统一图像去畸变坐标与有效边界，再排除车辆、行人等预先定义的动态区域。报告 ROI 像素分母和保留比例，不能由模型输出决定哪些像素计分。
- [ ] 最小实现中，ROI PSNR 按有效像素 MSE 计算，ROI SSIM 对预先确定的有效窗口汇总；LPIPS 先报告完整图结果。没有可靠的感受野屏蔽实现前不将 RGB 外部区域涂黑后所得 LPIPS 称为静态 ROI LPIPS。
- [ ] 静态指标脚本含小型自检：只修改忽略区像素不改变 ROI PSNR，修改有效区使指标下降，无有效像素时明确报告不可评测而不是输出 0；逐图记录有效像素数。
- [ ] 报告冻结候选测试帧数、筛选排除数、成功定位数和实际评分数。评分均值只代表成功定位部分，未定位比例必须一并出现。
- [ ] 保存训练轨迹附近的固定新视角演示及失败截图，使用 `ns-render camera-path`。偏离采集路径的视角可用于展示局限，没有参考真值时不计算 PSNR。[S7]
- [ ] 渲染性能固定在 960×544，模型加载和预热后对相同路径计时，报告每帧 GPU 耗时与 FPS，文件编码耗时另计；训练用时、离线渲染吞吐、viewer 帧率分别报告。
- [ ] 导出高斯 PLY（按安装版本 `ns-export gaussian-splat --help` 核对）并验证能重新加载；保存 checkpoint、配置、环境、训练地图、数据清单、相机路径和渲染结果。
- [ ] `report.md` 包含覆盖与孔洞、动态残影、典型新视角、SfM 注册率、图像指标、成本与失败案例。无独立深度真值或已知尺寸时只讨论外观及相对结构，不宣称绝对几何精度。

评测命令模板：[S6]

```bash
ns-eval --load-config "$CONFIG_PATH" --output-path "$METRICS_PATH" --render-output-path "$EVAL_IMAGES_DIR"
```

**完成标准：** 固定版本可复跑，评测分母可追踪，限制与失败有证据，最终文件在持久位置保存。

## 推进顺序与时间预算

先完成 Task 0；CPU 抽帧与帧清单可在 GPU 排查期间推进。Task 1 → Task 2 的未加掩码短地图 → 官方转换与 Task 4 的短训练，形成首个可查看结果。之后完成 Task 3 掩码，重跑 Task 2 的正式地图与留出定位，再转换、正式训练和评测。

建议划分为五个工作窗口：环境与存储、抽帧与筛选、SfM 与短训练、掩码与正式训练、评测与交付。每窗先预留半天到一天人工时间，机器计算单独实测；驱动、安装和位姿恢复失败会改变总日历时间，不承诺固定训练分钟数。

## 参考依据

- [S1] [Nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)：稀疏几何初始化、运行入口及显存估算。
- [S2] [COLMAP FAQ](https://colmap.github.io/faq.html)：特征掩码、已有地图上的新图定位、可选 BA 与 CPU 路径。
- [S3] [Nerfstudio 数据约定](https://docs.nerf.studio/quickstart/data_conventions.html)：相机轴向、掩码格式与极性。
- [S4] [ImagesToNerfstudioDataset 官方实现](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/process_data/images_to_nerfstudio_dataset.py)：复用已有模型和文件复制行为。
- [S5] [NerfstudioDataParser 官方实现](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/nerfstudio_dataparser.py)：显式划分、PLY 和坐标变换。实现时固定具体版本，不直接追随 main。
- [S6] [ns-eval](https://docs.nerf.studio/reference/cli/ns_eval.html) 与 [Splatfacto 官方实现](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py)：默认评测入口以及训练掩码、图像指标的行为差异。
- [S7] [ns-render](https://docs.nerf.studio/reference/cli/ns_render.html)：相机路径与数据集渲染入口。

访问日期：2026-11-03。采样频率、时间块划分、90% 注册率检查线、2 像素误差检查线、2,000/30,000 步和 30 GB 空间预算均为项目建议，尚非实测结果。
