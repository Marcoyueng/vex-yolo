# VEX YOLO — Override

Override 2.1 v3：本地赛事视频索引、可恢复的分场抽帧、机器人与队号牌标注、跨场阵营关联、视频跟踪 ID 和人工复核计分。Windows 双击 `启动软件.cmd`，完整说明见 [桌面软件说明](桌面软件说明.md)。v2 红蓝车标签可通过界面明确转换并保留原件。

**当前状态：数据准备、训练、检测程序已实现；尚无经验证的 Override 专用权重，不能宣称已经能可靠识别。** 通用 YOLO 预训练权重只用作训练起点。程序会拒绝把 COCO 模型当作 Override 检测模型。

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

训练结果包含验证指标；上线前还应在独立赛事视频上检查每类 precision、recall、mAP50–95、误检和漏检，以及目标设备 FPS。当前未测量这些指标。新版把队伍身份与阵营分开，未实现自动队号 OCR；跟踪 ID 可能在遮挡或切镜后变化，不是永久身份。JSONL 输出含 track_id。

## 大视频赛事工作流

赛事页仅索引原路径，不复制或上传视频。可递归扫描目录，并导入 UTF-8 CSV（列为 `match_id,video,start,end,red1,red2,blue1,blue2`）。每行对应原视频的一场，start/end 是视频内秒数或 HH:MM:SS。界面可导出模板；官方对阵表没有视频时间时需手动对齐。原视频不要改名、搬移或替换。

批量抽帧在独立进程中逐张解码并写 SQLite 进度；同参数重启可继续，不重复完成项。默认每场最多 100 张、宽度不超过 1280、项目抽帧预算 10 GiB。更改抽帧参数会产生新任务；提高预算不会产生新任务。预算仅约束已记录 JPEG，其他标注、数据库和检测记录另占空间。已完成图片需要保留，不支持自动清理后重建。

赛事页可只检测选中场次的时间段：输出 JSONL，不导出整段视频。检测进程每次只读一帧，开始新场次重置跟踪；中断检测需重跑本场，只有抽帧支持断点续跑。完成的单场目录有 `complete.json`。身份绑定按 match_id、run_id、track_id 和帧区间隔离，避免串场/串运行。人工预览点车获取 ID 后绑定该场参赛队号，导出时根据本场名单计算红蓝；未绑定保持待确认。同队跨场换边已覆盖测试。

验证采用小型合成视频，覆盖中断续跑、预算停止/恢复、原视频不复制与不修改、源变更阻止续跑、对阵表事务回滚及跨场换边。未对实际 300GB 素材做端到端吞吐、磁盘或时长测试；不承诺固定处理速度。VFR/损坏文件和很长 GOP 的定位性能需在实际素材上检查。没有 trained v3 模型时，可先完成索引、抽帧和身份标注。

黄色 Pin Half 计分页依据用户提供的 Override 2.1 SC2–SC7，只处理标准 V5RC 对抗赛，输入为人工复核；只计算黄色半段小计，不代表完整比赛分数。自动从视频识别落位、接触、套叠、半段可见性和 Goal/Toggle 区域位置映射尚未实现。

## 检查

```powershell
python -m unittest discover -s tests -v
```

实现依据：[Ultralytics Python API](https://docs.ultralytics.com/usage/python/)、[Override 官方比赛介绍](https://www.vexrobotics.com/v5/competition)。Ultralytics 的使用遵循其 AGPL-3.0 / Enterprise 许可要求。
