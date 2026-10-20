# VEX YOLO — Override

Override 2.1 v3：本地赛事视频索引、可恢复的分场抽帧、机器人与队号牌标注、跨场阵营关联、视频跟踪 ID 和人工复核计分。Windows 双击 `启动软件.cmd`，完整说明见 [桌面软件说明](桌面软件说明.md)。v2 红蓝车标签可通过界面明确转换并保留原件。

**当前状态：本机已有少量同视频样本训练的机器人 / Toggle 试播，以及 Goal / Pin 半段 / Cup 的局部检测试训；尚未验证跨比赛泛化，自动有效计分仍未实现。** 试播数据、权重和视频不提交到 GitHub。通用 YOLO 预训练权重只用作训练起点。程序会拒绝把 COCO 模型当作 Override 检测模型。

## 安装

建议 Python 3.11 或 3.12，Windows / Linux 均可。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

下列命令在激活环境后运行。CPU 可运行，训练建议使用支持 CUDA 的 GPU，并按 PyTorch 官方指引安装对应版本。

## 视频 → 标注

素材线索来自 [VEX 官方赛事直播列表](https://events.vex.com/lv/webcasts?program=vex-robotics-competition)。以下链接已被列表收录，下载可用性和实际视频内容仍需检查：

- [Beach Brawl，2026-10-17](https://www.youtube.com/watch?v=gIXlJm1aV_o)
- [CyberHawks，2026-09-26](https://www.youtube.com/watch?v=_ocdVAisggw)

```powershell
python vex_yolo.py download "https://www.youtube.com/watch?v=gIXlJm1aV_o"
python vex_yolo.py extract data/videos/gIXlJm1aV_o-3600-3720.mp4 --session beach-brawl --every 3 --limit 300
```

下载器保存源信息，不提交原视频。需要安装 FFmpeg 并加入 PATH。默认只取 3600–3720 秒，可用 `--start` / `--end` 指定片段。首段两分钟已于本地成功下载并检查画面，但尚未完成标注和训练。

将抽帧导入 CVAT 等框选标注工具，导出 YOLO Detection 格式：

| ID | 类别 | 标注范围 |
|---|---|---|
| 0 | robot | 每台整车单独框；队伍与本场阵营作为独立属性 |
| 1 | team_plate | 机器人上的队号牌，另存文字标注 |
| 2–5 | toggle_red / toggle_blue / toggle_yellow / toggle_unknown | Toggle 活动杆按场内显示面颜色标注 |
| 6 | goal | 九个目标共用此类别 |
| 7–9 | pin_half_red / pin_half_blue / pin_half_yellow | 按颜色标单个半段，不标整根 Pin |
| 10 | cup | 单个 Cup |

只标可辨认的实例，严重遮挡无法区分时跳过；堆叠时分别框选可辨认实例。确认无目标的负样本必须有空 txt，缺失标签会被拒绝。截图里的旧赛季圆环、移动目标不属于这份数据集。

```text
data/override-v3/
  classes.json
  images/train/eventA__00001.jpg
  images/val/eventB__00001.jpg
  labels/train/eventA__00001.txt
  labels/val/eventB__00001.txt
```

标签每行 `class_id center_x center_y width height`，坐标归一化至 0–1。保留抽帧时的 `session__frame` 文件名，按不同赛事/拍摄场次分割训练与验证，避免相邻帧泄漏。不要把模型自动生成的框直接视作真值。

将新版标签文件夹中的 `classes.json` 复制到训练数据根目录，清单必须与 `schema.py` 匹配。机器人队号、队号牌文字及本场阵营保存到 `.identity.json`；YOLO 检测训练只使用 txt，身份/OCR 模型训练尚未实现。四台车是四个实例，用跟踪 ID 区分；四个 Toggle 和九个 Goal 是固定场地实例，不是各自独立的训练类别。

## 训练与检测

```powershell
python vex_yolo.py check
python vex_yolo.py train --device cpu --epochs 100
python vex_yolo.py predict --source match.mp4 --weights runs/train/override/weights/best.pt --track
python vex_yolo.py predict --source 0 --show --output runs/camera
```

支持图片路径、视频路径、摄像头编号。每次选择新输出目录，标注媒体位于 `annotated/`，逐帧坐标保存在 `detections.jsonl`。frame 是输出顺序编号；坐标为原图像素 xyxy，不代表场地坐标。默认阈值 0.35，可用 `--conf` 调整。

训练结果包含验证指标；上线前还应在独立赛事视频上检查每类 precision、recall、mAP50–95、误检和漏检，以及目标设备 FPS。当前少量同视频划分试验不能替代独立比赛验证。新版把队伍身份与阵营分开，未实现自动队号 OCR；跟踪 ID 可能在遮挡或切镜后变化，不是永久身份。JSONL 输出含 track_id。

## 大视频赛事工作流

赛事页仅索引原路径，不复制或上传视频。可递归扫描目录，并导入 UTF-8 CSV（列为 `match_id,video,start,end,red1,red2,blue1,blue2`）。每行对应原视频的一场，start/end 是视频内秒数或 HH:MM:SS。界面可导出模板；官方对阵表没有视频时间时需手动对齐。原视频不要改名、搬移或替换。

批量抽帧在独立进程中逐张解码并写 SQLite 进度；同参数重启可继续，不重复完成项。默认每场最多 100 张、宽度不超过 1280、项目抽帧预算 10 GiB。更改抽帧参数会产生新任务；提高预算不会产生新任务。预算仅约束已记录 JPEG，其他标注、数据库和检测记录另占空间。已完成图片需要保留，不支持自动清理后重建。

赛事页可只检测选中场次的时间段：输出 JSONL，不导出整段视频。检测进程每次只读一帧，开始新场次重置跟踪；中断检测需重跑本场，只有抽帧支持断点续跑。完成的单场目录有 `complete.json`。身份绑定按 match_id、run_id、track_id 和帧区间隔离，避免串场/串运行。人工预览点车获取 ID 后绑定该场参赛队号，导出时根据本场名单计算红蓝；未绑定保持待确认。同队跨场换边已覆盖测试。

验证采用小型合成视频，覆盖中断续跑、预算停止/恢复、原视频不复制与不修改、源变更阻止续跑、对阵表事务回滚及跨场换边。未对实际 300GB 素材做端到端吞吐、磁盘或时长测试；不承诺固定处理速度。VFR/损坏文件和很长 GOP 的定位性能需在实际素材上检查。没有 trained v3 模型时，可先完成索引、抽帧和身份标注。

原“区域与黄色半段计分”页保留黄色小计。新增“跟踪 / 九 Goal 计分”窗口支持本地视频同步回放、四队身份状态、Goal 标定和标准 V5RC 计分复核。自动从视频判断落位、接触、套叠和半段可见性尚未实现；当前 Goal 计分输入来自人工复核。

## 跟踪与九 Goal 面板

启动桌面软件，进入“跟踪 / 九 Goal 计分”，点击“打开跟踪与计分窗口”。本机已有试播可点击“打开本场试播”；原视频按路径读取，不复制或上传。其他视频需要对应的归一化身份结果 JSONL，每帧包含连续 `frame`、`seconds`、`robots`（归一化 `xyxy`、`team`、`alliance`、`track_id`）和可选 `toggles`。普通 CLI 的像素坐标 `detections` 文件不能直接使用。四队名单从身份结果读取，不把本场队号套到其他场次；跨场结果必须先分场。

四个队伍卡片不会因漏检消失。实线为本帧已关联；虚线为最多 1.5 秒的位置预测；超时隐藏预测框并显示失联和最后关联时间。这不是全场零丢失或通用 ReID 的保证。暂停后选队号并点击对应机器人框，从此帧起纠正该 track_id；ID 改变后需要重新确认。跳转会重建状态，避免把未来身份或未来分数带回较早画面。水印为 `miko yolo`。

选择“标定 Goal”，选 G0–G8 后在画面拖框。G0 为中央，G1–2 / G3–4 / G5–6 / G7–8 分别属于 red1 / red2 / blue1 / blue2；这些是项目内部编号，不是官方标签。按实际场地核对区域，不能根据画面左右随意命名。标定只记录目标位置，不自动识别 Pin 状态。

计分表内选一个 Goal，填写红、蓝、黄**有效可见 Pin 半段数**，核查 SC2/SC3 后勾选确认，再点“应用半段数”。空白是未知，0 是确认没有。设置四区的有效占领、中央机器人数量、阶段和已核查的自动奖励，点击“保存本帧计分复核快照”。未完成复核显示已知小计及 `+ ?`；随后播放显示该快照的时间并明确标为历史值，不冒充当前自动识别分数。未保存的编辑不会跨帧保留。

规则依据用户的 Override 2.1（标准 V5RC 对抗赛）：

- SC1：终场后五秒或全部静止，以先到者为准评定最终状态；回放中的中途分数仅为估算。
- SC2/SC3：确认 Placed 和可见条件后，红/蓝半段各给自身阵营 5 分，黄色给拥有阵营 10 分。嵌在 Goal 或 Cup 不透明半部内的半段不计可见；Cup 本身不直接加分。机器人接触 Pin 并不自动取消其 Placed 状态。
- SC4/SC5：Toggle 完全就位、静止、无机器人接触才构成有效占领；无效时中立。视觉颜色未经有效性确认时不能直接当成占领。每一区两 Goal 的黄色归属一致。
- SC5/SC6：中央黄色归属于终场在中央区域机器人更多的一方，平局不计；中央区域是立体空间，不能只看检测框中心。每台终场中央机器人 8 分。
- SC7：`autonomous` 预览排除中央机器人分及中央黄色分，也不把自动奖励加进自动阶段小计。`end` 加入已核查自动奖励：胜方 12、平局各 6；违规等情况由人工确认，不凭缺失观察自动判胜。

复核与标定自动保存到 `runs/dashboard/`，绑定原视频和检测结果的路径、大小、修改时间。点击“取本帧标注 Goal / Pin”可保存当前原始帧到 `data/goal-review/`，并在主工作台载入标注；重复点击不覆盖旧帧。当前未训练 Placed / 遮挡 / 套叠等计分状态，检测框数量不能直接等同于得分数量。

## 固定机位 Goal / Pin / Cup 试训

`prepare_goal_pilot.py` 读取本地 curation JSON 中的图片路径、选定时间点、物体坐标和场地标定，制作局部标注数据；视频坐标与标注保留在忽略的 `data/`、`runs/` 中，不嵌入或发布到源代码。用 `train_goal_pilot.py --curation 本地路径` 指定输入；本机也可从 `runs/goal_curation_source.json` 读取路径。`partial-review-*.json` 明确标记 `complete_scene: false`，不是原始整帧的完整标注，不能作为完整场景真值。当前试验裁剪数据的前四个时间点用于训练、后两个用于验证；裁剪来自同一视频，模型与标注均未经独立人工认证。验证指标只衡量这些局部初步标签，不能代表整场计分准确率。

`train_goal_pilot.py` 训练补充模型（goal、pin_half_red/blue/yellow、cup），默认 60 轮、320 像素，支持 `--base` 和 `--epochs`。模型保存在独立运行目录，不替换机器人模型。`field_specialist.py` 仅在九个已标定 Goal 向上的堆叠搜索区域生成局部观察窗口，排除无关地面区域；部分遮挡和灰色 Cup 可能漏检。机器人接触不直接取消 Pin 的 Placed 状态，需另行核查。该模型应通过此局部推理流程使用，直接全画面检测不是当前验证过的用法。

`render_goal_pilot.py` 复用此前确认的逐帧机器人 / Toggle 结果，生成原视频带框版并复制原音轨。九个绿色虚线框是固定 Goal **参考区域**；Pin / Cup 模型约每两秒更新，期间虚线 / `~` 表示最长 2.1 秒的光流位置估计，可能漂移或消失。全部物体状态都保持 `scoring_validity: unconfirmed`，不据此自动计分。输出需经过完整帧数检查才会接入工作台“打开本场试播”。

新版“查看本轮训练标注”会打开本次裁剪图片及对应标签，供查看和修正。旧机器人身份的长遮挡不确定性仍然存在；这轮 Goal 模型没有训练队号识别或保证机器人身份永不丢失。

## 比分看板与 Goal 堆叠筛选

双击 `打开比分看板.cmd`，或从工作台“打开跟踪与计分窗口”进入新版看板。布局参考视频旁的仪表盘：左侧视频，右侧红蓝大分数、Toggle、九个 Goal 和四队状态，下方分数走势。播放、暂停和跳转同步更新；这是本地检测结果回放，未接入新的摄像头实时推理。

`goal_observer.py` 只保留从 Goal 开口向上建立连接的 Pin / Cup 候选；仅在目标附近、落在地上、或悬空而未连到堆叠的物体不显示，也不参与候选估分。Cup 自身不加分。Goal 遮挡、不可见、连接归属不清时保留未知状态；仍可见且已连接的半段保留候选小计，卡片以 `+` 标记其余部分未知，不把遮挡视为零分。二维相接不能证明 SC2 套叠、SC3 可见性或 SC4 有效占领，因此自动结果始终是 **AI 估分**，不是已确认比赛总分；Toggle 可见颜色仅作为归属假设，中央机器人和自动奖励未经确认时保留问号。未读取直播记分牌来充当模型算分。

点击 Goal 卡片可打开复核窗口，也可把本帧候选数量送入复核。候选预填后必须核查并勾选 SC2/SC3，才可保存人工复核快照。快照只用于其对应帧，不把历史复核值当作当前自动得分。曲线只显示播放时间之前的候选估分；遮挡造成的已知小计减少不代表实际比赛扣分。

`render_score_studio.py` 导出同布局视频及只含 Goal 堆叠候选的 JSONL，原音轨复制保留，输出经过总帧数检查。所有自动候选仍标记 `scoring_validity: unconfirmed`。

补充堆叠训练时，`train_stack_refinement.py --review 本地标注.json` 将局部复核样本加入上次裁剪数据，独立保存候选模型，不立即替换已接入看板的版本。标注 JSON 的 `coordinate_size` 对应图片尺寸；每项 `samples` 包含 `id`、`image`、`split`、`review_region` 和 `objects`（`class_id`、`xyxy`）。仅复核区域用于新增裁剪，区域外像素遮去，避免把未标注邻近物体作为负样本。验证裁剪与训练分开，但同一视频的相似状态仍有相关性，不能据此宣称整场或跨比赛准确率。

`evaluate_stack_refinement.py` 在未用于训练的完整帧上比较指定 Goal 的半段数量；复核输入还需每项的 `frame`、`goal_id`。`infer_goal_records.py --weights 本地模型路径` 重新处理原视频，保留已接受的机器人身份与 Toggle 记录，默认每秒重新检测一次，间隔使用有时限的光流估计。高堆叠的搜索区域允许越过地面投影边界，避免机位透视截掉顶部；只有与 Goal 堆叠连接的候选参与显示和估分。

## 检查

```powershell
python -m unittest discover -s tests -v
```

实现依据：[Ultralytics Python API](https://docs.ultralytics.com/usage/python/)、[Override 官方比赛介绍](https://www.vexrobotics.com/v5/competition)。Ultralytics 的使用遵循其 AGPL-3.0 / Enterprise 许可要求。
