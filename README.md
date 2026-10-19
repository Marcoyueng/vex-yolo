# VEX YOLO — Override

Override 2.1：红蓝机器人、Toggle 颜色状态、Goal、彩色 Pin Half 和 Cup 标注与训练；视频跟踪 ID 与人工复核计分。Windows 双击 `启动软件.cmd`，完整说明见 [桌面软件说明](桌面软件说明.md)。新版标签与旧版不兼容。

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
| 0–1 | robot_red / robot_blue | 整车，按阵营标注 |
| 2–5 | toggle_red / toggle_blue / toggle_yellow / toggle_unknown | Toggle 活动杆按场内显示面颜色标注 |
| 6 | goal | 九个目标共用此类别 |
| 7–9 | pin_half_red / pin_half_blue / pin_half_yellow | 按颜色标单个半段，不标整根 Pin |
| 10 | cup | 单个 Cup |

只标可辨认的实例，严重遮挡无法区分时跳过；堆叠时分别框选可辨认实例。确认无目标的负样本必须有空 txt，缺失标签会被拒绝。截图里的旧赛季圆环、移动目标不属于这份数据集。

```text
data/override-v2/
  classes.json
  images/train/eventA__00001.jpg
  images/val/eventB__00001.jpg
  labels/train/eventA__00001.txt
  labels/val/eventB__00001.txt
```

标签每行 `class_id center_x center_y width height`，坐标归一化至 0–1。保留抽帧时的 `session__frame` 文件名，按不同赛事/拍摄场次分割训练与验证，避免相邻帧泄漏。不要把模型自动生成的框直接视作真值。

将新版标签文件夹中的 `classes.json` 复制到训练数据根目录，清单必须与 `schema.py` 匹配。四台车是四个实例，用跟踪 ID 区分；四个 Toggle 和九个 Goal 是固定场地实例，不是各自独立的训练类别。

## 训练与检测

```powershell
python vex_yolo.py check
python vex_yolo.py train --device cpu --epochs 100
python vex_yolo.py predict --source match.mp4 --weights runs/train/override/weights/best.pt --track
python vex_yolo.py predict --source 0 --show --output runs/camera
```

支持图片路径、视频路径、摄像头编号。每次选择新输出目录，标注媒体位于 `annotated/`，逐帧坐标保存在 `detections.jsonl`。frame 是输出顺序编号；坐标为原图像素 xyxy，不代表场地坐标。默认阈值 0.35，可用 `--conf` 调整。

训练结果包含验证指标；上线前还应在独立赛事视频上检查每类 precision、recall、mAP50–95、误检和漏检，以及目标设备 FPS。当前未测量这些指标。新版支持阵营类别，未实现队号识别；跟踪 ID 可能在遮挡或切镜后变化，不是永久身份。JSONL 输出含 track_id。

黄色 Pin Half 计分页依据用户提供的 Override 2.1 SC2–SC7，只处理标准 V5RC 对抗赛，输入为人工复核；只计算黄色半段小计，不代表完整比赛分数。自动从视频识别落位、接触、套叠、半段可见性和 Goal/Toggle 区域位置映射尚未实现。

## 检查

```powershell
python -m unittest discover -s tests -v
```

实现依据：[Ultralytics Python API](https://docs.ultralytics.com/usage/python/)、[Override 官方比赛介绍](https://www.vexrobotics.com/v5/competition)。Ultralytics 的使用遵循其 AGPL-3.0 / Enterprise 许可要求。
