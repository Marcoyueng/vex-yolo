# VEX YOLO — Override

第一阶段：检测机器人、Pin 和 Cup，输出边框与置信度。

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
| 0 | robot | 机器人可见整体，含伸出机构；不含驾驶员 |
| 1 | pin | 单个 Pin，包含可见两端；不按颜色拆类 |
| 2 | cup | 单个 Cup，翻面仍标同类 |

只标可辨认的实例，严重遮挡无法区分时跳过；堆叠时分别框选可辨认实例。确认无目标的负样本必须有空 txt，缺失标签会被拒绝。截图里的旧赛季圆环、移动目标不属于这份数据集。

```text
data/override/
  images/train/eventA__00001.jpg
  images/val/eventB__00001.jpg
  labels/train/eventA__00001.txt
  labels/val/eventB__00001.txt
```

标签每行 `class_id center_x center_y width height`，坐标归一化至 0–1。保留抽帧时的 `session__frame` 文件名，按不同赛事/拍摄场次分割训练与验证，避免相邻帧泄漏。不要把模型自动生成的框直接视作真值。

## 训练与检测

```powershell
python vex_yolo.py check
python vex_yolo.py train --device cpu --epochs 100
python vex_yolo.py predict --source match.mp4 --weights runs/train/override/weights/best.pt
python vex_yolo.py predict --source 0 --show --output runs/camera
```

支持图片路径、视频路径、摄像头编号。每次选择新输出目录，标注媒体位于 `annotated/`，逐帧坐标保存在 `detections.jsonl`。frame 是输出顺序编号；坐标为原图像素 xyxy，不代表场地坐标。默认阈值 0.35，可用 `--conf` 调整。

训练结果包含验证指标；上线前还应在独立赛事视频上检查每类 precision、recall、mAP50–95、误检和漏检，以及目标设备 FPS。当前未测量这些指标。检测框不等于已经得分，也不判断阵营或队号。

## 检查

```powershell
python -m unittest discover -s tests -v
```

实现依据：[Ultralytics Python API](https://docs.ultralytics.com/usage/python/)、[Override 官方比赛介绍](https://www.vexrobotics.com/v5/competition)。Ultralytics 的使用遵循其 AGPL-3.0 / Enterprise 许可要求。
