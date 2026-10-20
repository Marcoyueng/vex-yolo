"""Local VEX Override annotation and detection workbench (Tkinter)."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import csv
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ImageOps
from schema import CLASSES, COLORS, REGIONS, ensure_schema, migrate_v2
from scoring import yellow_score
from event_project import connect, add_media, import_matches, CSV_FIELDS, VIDEO_EXTS, bind_team, remove_binding, export_identities, team_id

ROOT = Path(__file__).resolve().parent


def encode_boxes(boxes, width, height):
    return ''.join(f'{c} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} '
                   f'{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n'
                   for c, x1, y1, x2, y2 in boxes)


def decode_boxes(text, width, height):
    boxes = []
    for line in text.splitlines():
        c, x, y, w, h = map(float, line.split())
        if int(c) != c or not 0 <= c < len(CLASSES) or not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError('标注格式无效')
        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1.000001 or y+h/2 > 1.000001:
            raise ValueError('标注框超出图片边界')
        boxes.append((int(c), (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height))
    return boxes


class Workbench(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('VEX YOLO · Override 本地工作台')
        self.geometry('1180x820')
        self.minsize(960, 680)
        self.configure(bg='#101827')
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background='#101827')
        style.configure('TLabel', background='#101827', foreground='#e2e8f0', font=('Microsoft YaHei UI', 10))
        style.configure('TButton', padding=7, font=('Microsoft YaHei UI', 10))
        style.configure('TNotebook', background='#101827')
        style.configure('TNotebook.Tab', padding=(18, 10))
        self.files, self.boxes, self.identities = [], [], []
        self.index, self.image, self.photo = 0, None, None
        self.dirty = False
        self.process = None
        self.messages = queue.Queue()
        self.closing = False
        self.scale, self.offset = 1, (0, 0)
        self.start = None
        self.current_class = tk.StringVar(value='robot')
        self.label_dir = tk.StringVar(value=str(ROOT / 'data' / 'annotations-v3'))
        self.project_path = tk.StringVar(value=str(ROOT/'data/events/my-event'))
        self.match_id = tk.StringVar()
        self.annotation_team = tk.StringVar()
        self.plate_text = tk.StringVar()
        self.status = tk.StringVar(value='v3：机器人类别统一为 robot，队号和本场阵营单独保存')
        header = ttk.Frame(self, padding=16)
        header.pack(fill='x')
        ttk.Label(header, text='VEX YOLO  /  OVERRIDE', font=('Microsoft YaHei UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(header, text='赛事视频本地索引 · 单场队伍身份 · 断点抽帧 · 模型与规则复核').pack(anchor='w', pady=(6, 0))
        tabs = ttk.Notebook(self)
        tabs.pack(fill='both', expand=True, padx=16)
        annotation = ttk.Frame(tabs, padding=12)
        self.tabs, self.annotation_tab = tabs, annotation
        jobs = ttk.Frame(tabs, padding=16)
        scoring = ttk.Frame(tabs, padding=16)
        event = ttk.Frame(tabs, padding=12)
        tabs.add(event, text='赛事项目 / 大视频')
        tabs.add(annotation, text='图片标注')
        tabs.add(jobs, text='抽帧 / 训练 / 检测')
        tabs.add(scoring, text='区域与黄色半段计分')
        self.build_annotation(annotation)
        self.build_jobs(jobs)
        self.build_scoring(scoring)
        self.build_event(event)
        tracking = ttk.Frame(tabs, padding=16)
        tabs.add(tracking, text='跟踪 / 九 Goal 计分')
        ttk.Label(tracking, text='miko yolo · 全场身份与九 Goal 计分', font=('Microsoft YaHei UI', 17, 'bold')).pack(anchor='w', pady=12)
        ttk.Label(tracking, text='四支队伍持续显示；短时遮挡用虚线预测；暂停后点击机器人框可纠正队号。\n九个 Goal 支持位置标定和 SC2–SC7 计分复核，未知项保留问号。\n当前试播已接入机器人 / Toggle 结果；Goal、Pin 半段自动计分仍需要对应标注训练。', wraplength=950).pack(anchor='w', pady=12)
        ttk.Button(tracking, text='打开跟踪与计分窗口', command=self.open_dashboard).pack(anchor='w', pady=12)
        ttk.Button(tracking, text='查看本轮训练标注', command=self.open_goal_annotations).pack(anchor='w', pady=6)
        ttk.Label(self, textvariable=self.status, padding=12).pack(fill='x')
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.after(100, self.poll)

    def open_dashboard(self):
        from score_studio import ScoreStudio
        ScoreStudio(self)

    def open_goal_annotations(self):
        if not self.confirm_dirty():return
        try:
            state=json.loads((ROOT/'runs/active_goal_pilot.json').read_text(encoding='utf-8'))
            dataset=Path(state['directory'])/'dataset'
            files=sorted((dataset/'images/train').glob('*.jpg'))
            if not files:raise ValueError('尚无本轮训练标注')
            labels=dataset/'labels/train'
            # Scope includes assistant-reviewed crops, not complete original scenes.
            ensure_schema(labels)
            self.files,self.index=files,0;self.label_dir.set(str(labels))
            self.current_class.set('goal');self.load_image();self.tabs.select(self.annotation_tab)
            self.status.set('本轮为助手初步标注的局部裁剪；可查看和修正。尚未标注落位、套叠和有效得分状态。')
        except Exception as error:messagebox.showerror('无法打开训练标注',str(error))

    def button(self, parent, text, command):
        b = ttk.Button(parent, text=text, command=command)
        b.pack(side='left', padx=(0, 8))
        return b

    def build_annotation(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill='x')
        self.button(bar, '打开图片文件夹', self.open_folder)
        self.button(bar, '标注保存位置', self.choose_labels)
        ttk.Label(bar, text='类别').pack(side='left')
        ttk.Combobox(bar, values=CLASSES, textvariable=self.current_class, state='readonly', width=17).pack(side='left', padx=8)
        self.button(bar, '撤销最后一个框', self.undo)
        self.button(bar, '保存标注', self.save)
        identity = ttk.Frame(parent); identity.pack(fill='x', pady=5)
        ttk.Label(identity, text='场次').pack(side='left')
        self.annotation_match_combo = ttk.Combobox(identity, textvariable=self.match_id, state='readonly', width=14)
        self.annotation_match_combo.pack(side='left', padx=5)
        self.annotation_match_combo.bind('<<ComboboxSelected>>', lambda _: self.refresh_teams())
        ttk.Label(identity, text='此框队号').pack(side='left')
        self.team_combo = ttk.Combobox(identity, textvariable=self.annotation_team, width=14, state='readonly')
        self.team_combo.pack(side='left', padx=5)
        ttk.Label(identity, text='队号牌文字（逐框填写）').pack(side='left')
        ttk.Entry(identity,textvariable=self.plate_text,width=14).pack(side='left',padx=5)
        self.button(identity,'转换旧v2标签',self.upgrade_labels)
        ttk.Label(parent, textvariable=self.label_dir).pack(anchor='w', pady=8)
        self.canvas = tk.Canvas(parent, bg='#070e1c', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _: self.redraw())
        self.canvas.bind('<ButtonPress-1>', self.begin_box)
        self.canvas.bind('<B1-Motion>', self.move_box)
        self.canvas.bind('<ButtonRelease-1>', self.end_box)
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(10, 0))
        self.button(bar, '上一张', lambda: self.navigate(-1))
        self.button(bar, '下一张', lambda: self.navigate(1))
        ttk.Label(bar, text='每台车单独框 robot；队号牌另框 team_plate。先选场次和队号，再画对应框；看不清可留空。').pack(side='left')

    def choose_labels(self):
        if not self.confirm_dirty():
            return
        folder = filedialog.askdirectory(title='选择 YOLO 标签保存文件夹')
        if folder:
            self.label_dir.set(folder)
            if self.files:
                self.load_image()

    def upgrade_labels(self):
        if not self.confirm_dirty():return
        source=filedialog.askdirectory(title='选择旧 annotations-v2 标签文件夹')
        if not source:return
        target=filedialog.askdirectory(title='选择新的空标签文件夹（保留旧标签原件）')
        if not target:return
        try:
            count=migrate_v2(source,target)
            self.label_dir.set(target)
            if self.files:self.load_image()
            self.status.set(f'已转换 {count} 个标签；原件保留。旧红蓝颜色不等于队伍身份，需要补充队号。')
        except Exception as error:messagebox.showerror('转换失败',str(error))

    def open_folder(self):
        if not self.confirm_dirty():
            return
        folder = filedialog.askdirectory(title='选择图片文件夹')
        if not folder:
            return
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
        if len({p.stem for p in files}) != len(files):
            messagebox.showerror('文件名冲突', '同名不同扩展名的图片会覆盖标签，请先修改图片名称。')
            return
        if not files:
            messagebox.showinfo('没有图片', '文件夹内没有支持的图片。')
            return
        self.files, self.index = files, 0
        self.load_image()

    def label_path(self):
        return Path(self.label_dir.get()) / (self.files[self.index].stem + '.txt')

    def load_image(self):
        try:
            self.plate_text.set('')
            self.frame_match = None
            with connect(self.project_path.get()) as db:
                row=db.execute('SELECT j.match_id FROM frames f JOIN jobs j ON j.id=f.job_id WHERE f.path=?',
                               (str(self.files[self.index].resolve()),)).fetchone()
                if row:
                    self.frame_match=row['match_id'];self.match_id.set(row['match_id']);self.refresh_teams()
            ensure_schema(self.label_dir.get(), create=True)
            with Image.open(self.files[self.index]) as image:
                self.image = ImageOps.exif_transpose(image).convert('RGB')
            label = self.label_path()
            self.boxes = decode_boxes(label.read_text(encoding='utf-8'), *self.image.size) if label.exists() else []
            self.identities = [{} for _ in self.boxes]
            sidecar = label.with_suffix('.identity.json')
            if sidecar.exists():
                data = json.loads(sidecar.read_text(encoding='utf-8'))
                digest = hashlib.sha256(label.read_bytes()).hexdigest()
                if data.get('labels_sha256') != digest or len(data.get('objects',[])) != len(self.boxes):
                    raise ValueError('标签与队号记录不一致，请先备份并检查该图片的 identity.json。')
                self.identities = data['objects']
            self.dirty = False
            self.redraw()
            self.update_status()
        except Exception as error:
            self.image, self.boxes = None, []
            self.redraw()
            messagebox.showerror('打开失败', str(error))

    def update_status(self):
        if self.files:
            self.status.set(f'{self.index+1} / {len(self.files)}  ·  {self.files[self.index].name}  ·  {len(self.boxes)} 个框' + ('  ·  未保存' if self.dirty else ''))

    def redraw(self):
        self.canvas.delete('all')
        if self.image is None:
            self.canvas.create_text(360, 180, text='打开图片文件夹，开始标注', fill='#94a3b8', font=('Microsoft YaHei UI', 20))
            return
        w, h = self.image.size
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(cw/w, ch/h)
        size = (max(1, int(w*self.scale)), max(1, int(h*self.scale)))
        self.offset = ((cw-size[0])/2, (ch-size[1])/2)
        self.photo = ImageTk.PhotoImage(self.image.resize(size))
        self.canvas.create_image(*self.offset, image=self.photo, anchor='nw')
        for index, (c, x1, y1, x2, y2) in enumerate(self.boxes):
            ox, oy = self.offset
            coords = (ox+x1*self.scale, oy+y1*self.scale, ox+x2*self.scale, oy+y2*self.scale)
            self.canvas.create_rectangle(*coords, outline=COLORS[c], width=2)
            team = self.identities[index].get('team') if index < len(self.identities) else None
            self.canvas.create_text(coords[0]+4, coords[1]+4, text=CLASSES[c]+(' · '+team if team else ''), anchor='nw', fill=COLORS[c], font=('Arial', 12, 'bold'))

    def point(self, event):
        ox, oy = self.offset
        w, h = self.image.size
        return (min(w, max(0, (event.x-ox)/self.scale)), min(h, max(0, (event.y-oy)/self.scale)))

    def begin_box(self, event):
        if self.image is not None:
            self.start = self.point(event)

    def move_box(self, event):
        if self.start is None or self.image is None:
            return
        self.canvas.delete('draft')
        ox, oy = self.offset
        x, y = self.point(event)
        self.canvas.create_rectangle(ox+self.start[0]*self.scale, oy+self.start[1]*self.scale,
                                     ox+x*self.scale, oy+y*self.scale, outline='white', dash=(4, 3), tags='draft')

    def end_box(self, event):
        if self.start is None or self.image is None:
            return
        x, y = self.point(event)
        x1, x2 = sorted((self.start[0], x))
        y1, y2 = sorted((self.start[1], y))
        self.start = None
        if x2-x1 >= 2 and y2-y1 >= 2:
            try:
                metadata = {}
                if self.current_class.get() in ('robot','team_plate'):
                    if self.frame_match and self.match_id.get()!=self.frame_match:
                        raise ValueError('此图片来自另一场比赛，请切回 '+self.frame_match)
                    team = self.annotation_team.get().strip()
                    if team:
                        with connect(self.project_path.get()) as db:
                            match = db.execute('SELECT * FROM matches WHERE match_id=?',(self.match_id.get(),)).fetchone()
                            if not match or team not in [match[k] for k in CSV_FIELDS[4:]]:
                                raise ValueError('请先选择正确场次，再从该场四个队伍中选择队号。')
                            metadata = {'match_id':self.match_id.get(),'team':team,
                                        'alliance':'red' if team in (match['red1'],match['red2']) else 'blue'}
                    if self.current_class.get() == 'team_plate' and self.plate_text.get().strip():
                        metadata['plate_text'] = team_id(self.plate_text.get())
                        if team and metadata['plate_text'] != team:
                            raise ValueError('队号牌文字与所选队号不一致，请先修正。')
            except ValueError as error:
                self.redraw(); messagebox.showerror('队号输入错误',str(error)); return
            self.boxes.append((CLASSES.index(self.current_class.get()), x1, y1, x2, y2))
            self.identities.append(metadata)
            if self.current_class.get()=='team_plate':
                self.plate_text.set('')
            self.dirty = True
        self.redraw()
        self.update_status()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.identities.pop()
            self.dirty = True
            self.redraw()
            self.update_status()

    def save(self):
        if self.image is None:
            return False
        try:
            ensure_schema(self.label_dir.get(), create=True)
            target = self.label_path()
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix('.txt.tmp')
            temp.write_text(encode_boxes(self.boxes, *self.image.size), encoding='utf-8')
            sidecar = target.with_suffix('.identity.json')
            meta_temp = sidecar.with_suffix('.tmp')
            meta_temp.write_text(json.dumps({'labels_sha256':hashlib.sha256(temp.read_bytes()).hexdigest(),
                                             'objects':self.identities},ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(target)
            meta_temp.replace(sidecar)
            self.dirty = False
            self.update_status()
            return True
        except Exception as error:
            messagebox.showerror('保存失败', str(error))
            return False

    def confirm_dirty(self):
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel('未保存的标注', '保存当前图片的标注吗？')
        return self.save() if answer else answer is not None

    def navigate(self, step):
        if self.files and self.confirm_dirty():
            self.index = max(0, min(len(self.files)-1, self.index+step))
            self.load_image()

    def field(self, parent, label, default='', picker=None):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=4)
        ttk.Label(row, text=label, width=16).pack(side='left')
        value = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=value).pack(side='left', fill='x', expand=True, padx=8)
        if picker:
            def choose():
                result = picker()
                if result:
                    value.set(result)
            self.button(row, '选择', choose)
        return value

    def build_jobs(self, parent):
        self.weights = self.field(parent, '模型权重 .pt', str(ROOT/'runs/train/override/weights/best.pt'), lambda: filedialog.askopenfilename(filetypes=[('模型', '*.pt')]))
        self.source = self.field(parent, '图片 / 视频 / 摄像头', '', lambda: filedialog.askopenfilename(filetypes=[('媒体', '*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv'), ('所有文件', '*.*')]))
        self.output = self.field(parent, '结果保存文件夹', str(ROOT/'runs/desktop'), lambda: filedialog.askdirectory())
        self.confidence = self.field(parent, '置信度阈值', '0.35')
        self.device = self.field(parent, '运行设备', 'cpu')
        self.tracking = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text='启用视频目标跟踪 ID（需要训练好的新版模型；遮挡后 ID 可能变化）', variable=self.tracking).pack(anchor='w')
        ttk.Label(parent, text='摄像头：在输入栏填写 0；检测会打开实时预览窗口。每次检测自动新建结果目录。').pack(anchor='w', pady=4)
        row = ttk.Frame(parent); row.pack(fill='x', pady=8)
        self.button(row, '开始检测', self.detect)
        self.button(row, '打开结果文件夹', self.open_output)
        self.button(row, '停止当前任务', self.stop)
        ttk.Separator(parent).pack(fill='x', pady=8)
        self.session = self.field(parent, '抽帧场次名称', 'event-a')
        row = ttk.Frame(parent); row.pack(fill='x', pady=6)
        self.button(row, '从所选视频抽帧', self.extract)
        ttk.Label(row, text='每 5 秒一张，最多 300 张，保存到 data/unlabelled').pack(side='left')
        self.dataset = self.field(parent, '训练数据文件夹', str(ROOT/'data/override-v3'), lambda: filedialog.askdirectory())
        self.epochs = self.field(parent, '训练轮数', '100')
        row = ttk.Frame(parent); row.pack(fill='x', pady=6)
        self.button(row, '检查数据', lambda: self.run_job(['check', '--data', self.dataset.get()]))
        self.button(row, '开始训练', self.train)
        ttk.Label(parent, text='训练数据需按 images/train、images/val、labels/train、labels/val 整理；详见桌面软件说明。').pack(anchor='w', pady=4)
        self.log = tk.Text(parent, height=10, bg='#070e1c', fg='#cbd5e1', insertbackground='white', wrap='word', font=('Consolas', 10))
        self.log.pack(fill='both', expand=True, pady=8)
        self.log.configure(state='disabled')

    def detect(self):
        try:
            conf = float(self.confidence.get())
            if not 0 <= conf <= 1:
                raise ValueError()
        except ValueError:
            messagebox.showerror('输入错误', '置信度必须在 0–1 之间'); return
        if not Path(self.weights.get()).is_file() or not self.source.get().strip():
            messagebox.showerror('缺少输入', '请选择 Override 专用 .pt 权重，并指定图片、视频或摄像头编号。'); return
        from datetime import datetime
        self.last_output = Path(self.output.get()) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        arguments = ['predict', '--weights', self.weights.get(), '--source', self.source.get(),
                     '--conf', str(conf), '--device', self.device.get(), '--output', str(self.last_output), '--show']
        if self.tracking.get():
            arguments.append('--track')
        self.run_job(arguments)

    def build_scoring(self, parent):
        ttk.Label(parent, text='标准 V5RC 对抗赛 · SC2–SC7 · 人工复核输入', font=('Microsoft YaHei UI', 15, 'bold')).pack(anchor='w')
        ttk.Label(parent, text='这里只计算黄色 Pin Half 分数，当前不从视频自动填数。未知留空，不会按 0 分认定。').pack(anchor='w', pady=6)
        ttk.Label(parent, text='区域名是固定位置，不代表当前归属。Goal 编号是本项目约定，不是官方 AprilTag 编号。').pack(anchor='w')
        self.toggle_inputs = {}
        for region in REGIONS:
            row = ttk.Frame(parent); row.pack(fill='x', pady=7)
            ttk.Label(row, text=region, width=10).pack(side='left')
            values = []
            for label, options in [('颜色', ('unknown', 'red', 'blue', 'yellow')),
                                   ('完全落位静止', ('未知', '是', '否')), ('接触机器人', ('未知', '是', '否'))]:
                ttk.Label(row, text=label).pack(side='left', padx=8)
                var = tk.StringVar(value=options[0])
                ttk.Combobox(row, values=options, textvariable=var, state='readonly', width=10).pack(side='left')
                values.append(var)
            self.toggle_inputs[region] = values
        ttk.Separator(parent).pack(fill='x', pady=6)
        ttk.Label(parent, text='填写各 Goal 已确认 Placed 且可见的黄色半段数量；0 表示确认没有，空白表示未知。').pack(anchor='w')
        grid = ttk.Frame(parent); grid.pack(fill='x', pady=8)
        self.goal_counts = {}
        for i in range(9):
            region = 'midfield' if i == 0 else REGIONS[(i-1)//2]
            cell = ttk.Frame(grid); cell.grid(row=i//3, column=i%3, padx=12, pady=5, sticky='w')
            ttk.Label(cell, text=f'G{i} · {region}', width=17).pack(side='left')
            var = tk.StringVar()
            ttk.Entry(cell, textvariable=var, width=7).pack(side='left')
            self.goal_counts[f'G{i}'] = var
        self.mid_red = self.field(parent, '中央红方车数量', '')
        self.mid_blue = self.field(parent, '中央蓝方车数量', '')
        row = ttk.Frame(parent); row.pack(fill='x', pady=8)
        self.phase = tk.StringVar(value='end')
        ttk.Label(row, text='结算阶段').pack(side='left', padx=8)
        ttk.Combobox(row, values=('end', 'autonomous'), textvariable=self.phase, state='readonly', width=14).pack(side='left', padx=8)
        self.button(row, '计算黄色半段分数', self.calculate_score)
        self.button(row, '导出复核记录', self.export_score)
        self.score_text = tk.StringVar(value='尚未计算。end=比赛结束；autonomous=自动阶段（中央黄色半段不计入）。')
        ttk.Label(parent, textvariable=self.score_text, wraplength=1050).pack(anchor='w', pady=8)
        ttk.Label(parent, text='未落位或仍接触机器人 → 中立；无法确认 → 待复核。普通二维框不能单独证明套叠、遮挡或中央区域相交。', wraplength=1050).pack(anchor='w')

    def calculate_score(self):
        def number(value):
            return int(value) if value.strip() else None
        try:
            flags = {'未知': None, '是': True, '否': False}
            toggles = {region: {'color': v[0].get(), 'settled': flags[v[1].get()], 'robot_contact': flags[v[2].get()]}
                       for region, v in self.toggle_inputs.items()}
            counts = {goal: number(var.get()) for goal, var in self.goal_counts.items()}
            result = yellow_score(toggles, counts, number(self.mid_red.get()), number(self.mid_blue.get()), self.phase.get())
            state = '已填状态下的小计' if result['complete'] else '已确认部分小计；仍有待复核项'
            owners = ' / '.join(f'{k}: {v}' for k, v in result['owners'].items())
            self.score_text.set(f'{state}：红方 {result["red"]} 分，蓝方 {result["blue"]} 分（仅黄色半段）。\n{owners}')
            self.score_record = {'rule_version': 'Override 2.1 V5RC', 'source': 'manual_review', 'phase': self.phase.get(),
                                 'toggles': toggles, 'midfield_red': number(self.mid_red.get()),
                                 'midfield_blue': number(self.mid_blue.get()), 'result': result}
            return True
        except (ValueError, TypeError) as error:
            messagebox.showerror('输入错误', str(error)); return False

    def export_score(self):
        if self.calculate_score():
            target = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('复核记录', '*.json')])
            if target:
                try:
                    Path(target).write_text(json.dumps(self.score_record, ensure_ascii=False, indent=2), encoding='utf-8')
                except OSError as error:
                    messagebox.showerror('导出失败', str(error))

    def extract(self):
        if not Path(self.source.get()).is_file():
            messagebox.showerror('缺少视频', '先选择本地视频文件。'); return
        self.run_job(['extract', self.source.get(), '--session', self.session.get(), '--every', '5', '--limit', '300'])

    def train(self):
        try:
            epochs = int(self.epochs.get())
            if epochs < 1:
                raise ValueError()
        except ValueError:
            messagebox.showerror('输入错误', '训练轮数必须是正整数。'); return
        self.run_job(['train', '--data', self.dataset.get(), '--epochs', str(epochs), '--device', self.device.get()])

    def build_event(self, parent):
        top = ttk.Frame(parent); top.pack(fill='x')
        ttk.Label(top,text='赛事项目目录').pack(side='left')
        ttk.Entry(top,textvariable=self.project_path).pack(side='left',fill='x',expand=True,padx=8)
        self.button(top,'新建 / 打开',self.choose_project)
        self.event_summary = tk.StringVar(value='视频保留在原路径，仅保存索引和小型结果；无云端上传。')
        ttk.Label(parent,textvariable=self.event_summary).pack(anchor='w',pady=5)
        pages = ttk.Notebook(parent); pages.pack(fill='both',expand=True)
        media_page=ttk.Frame(pages,padding=10); identity_page=ttk.Frame(pages,padding=10)
        pages.add(media_page,text='视频目录 / 对阵表 / 抽帧'); pages.add(identity_page,text='队伍身份绑定')
        row=ttk.Frame(media_page); row.pack(fill='x')
        self.button(row,'添加多个视频',self.add_videos)
        self.button(row,'扫描视频目录',self.scan_videos)
        self.button(row,'下载对阵表模板',self.csv_template)
        self.button(row,'导入对阵表 CSV',self.load_matches)
        self.button(row,'刷新',self.refresh_event)
        ttk.Label(media_page,text='CSV 每行一场：视频、开始/结束时间及四个队号。时间是视频内时间，不是比赛倒计时。').pack(anchor='w',pady=6)
        self.media_tree=ttk.Treeview(media_page,columns=('name','size'),show='headings',height=3)
        self.media_tree.heading('name',text='原视频路径（不复制）'); self.media_tree.heading('size',text='GB')
        self.media_tree.column('name',width=850); self.media_tree.column('size',width=90)
        self.media_tree.pack(fill='x')
        self.match_tree=ttk.Treeview(media_page,columns=('match','time','red','blue','status'),show='headings',height=7,selectmode='browse')
        for key,title,width in [('match','场次',100),('time','视频时间段 / 秒',180),('red','红方队伍',180),('blue','蓝方队伍',180),('status','抽帧进度',180)]:
            self.match_tree.heading(key,text=title); self.match_tree.column(key,width=width)
        self.match_tree.pack(fill='both',expand=True,pady=7)
        self.match_tree.bind('<<TreeviewSelect>>',self.select_match)
        options=ttk.Frame(media_page); options.pack(fill='x',pady=5)
        self.sample_every=tk.StringVar(value='5'); self.sample_limit=tk.StringVar(value='100')
        self.sample_width=tk.StringVar(value='1280'); self.sample_budget=tk.StringVar(value='10')
        for title,var in [('间隔秒',self.sample_every),('每场最多张数',self.sample_limit),('最大宽度',self.sample_width),('抽帧预算 GB',self.sample_budget)]:
            ttk.Label(options,text=title).pack(side='left',padx=5)
            ttk.Entry(options,textvariable=var,width=7).pack(side='left')
        row=ttk.Frame(media_page); row.pack(fill='x',pady=6)
        self.button(row,'开始 / 继续全部抽帧',self.event_extract)
        self.button(row,'停止',self.stop)
        self.button(row,'标注已抽取图片',self.annotate_event)
        self.button(row,'仅检测选中场次',self.event_detect)
        ttk.Label(media_page,text='抽帧按张记录进度；再次开始可续跑。每次只解码一帧，默认不输出整段画框视频。日志在“抽帧 / 训练 / 检测”页。',wraplength=1050).pack(anchor='w')
        ttk.Label(identity_page,text='按单场对阵表绑定队号：同一个队伍在另一场可以换成不同阵营。',font=('Microsoft YaHei UI',13,'bold')).pack(anchor='w',pady=8)
        ttk.Label(identity_page,text='先在赛事列表选场次，再运行该场检测。自动队号 OCR 尚未接入；此处是人工确认，不猜测身份。').pack(anchor='w')
        ttk.Label(identity_page,textvariable=self.match_id).pack(anchor='w',pady=5)
        self.identity_report=self.field(identity_page,'单场检测 JSONL','',lambda:filedialog.askopenfilename(filetypes=[('检测记录','*.jsonl')]))
        self.binding_track=self.field(identity_page,'机器人 track_id','')
        self.binding_first=self.field(identity_page,'起始帧（从0计）','0')
        self.binding_last=self.field(identity_page,'结束帧（含）','')
        self.binding_team=self.field(identity_page,'确认队号','')
        row=ttk.Frame(identity_page);row.pack(fill='x',pady=8)
        self.button(row,'保存区间绑定',self.save_binding)
        self.button(row,'撤销此起始帧绑定',self.undo_binding)
        self.button(row,'预览起始帧 / 点车取ID',self.preview_identity)
        self.button(row,'导出带队号的记录',self.export_binding)
        self.binding_view=tk.Text(identity_page,height=9,bg='#070e1c',fg='#cbd5e1',wrap='word')
        self.binding_view.pack(fill='both',expand=True)
        self.binding_view.configure(state='disabled')
        ttk.Label(identity_page,text='绑定仅适用于同一场、同一次运行、指定帧区间。遮挡后换 ID 要重新绑定；只绑定 robot 的 ID。',wraplength=1050).pack(anchor='w',pady=5)
        self.refresh_event()

    def choose_project(self):
        folder=filedialog.askdirectory(title='选择赛事项目目录（原视频不会复制到这里）')
        if folder:
            self.project_path.set(folder);self.match_id.set('');self.refresh_event()

    def catalog_media(self, paths=None, folder=None):
        import threading
        project=self.project_path.get()
        def work():
            try:
                count=0
                videos=paths if paths is not None else (str(p) for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
                for path in videos:
                    add_media(project,[path]);count+=1
                self.messages.put(('event',f'已索引 {count} 个视频；未复制原文件。'))
            except Exception as error:
                self.messages.put(('event',f'索引失败：{error}'))
        threading.Thread(target=work,daemon=True).start()

    def add_videos(self):
        paths=filedialog.askopenfilenames(title='选择本地视频',filetypes=[('视频','*.mp4 *.mkv *.mov *.avi *.m4v *.mts')])
        if paths:self.catalog_media(paths=paths)

    def scan_videos(self):
        folder=filedialog.askdirectory(title='扫描本地视频目录（包含子目录）')
        if folder:self.catalog_media(folder=folder)

    def refresh_event(self):
        try:
            with connect(self.project_path.get()) as db:
                videos=db.execute('SELECT * FROM media ORDER BY path LIMIT 5000').fetchall()
                size=db.execute('SELECT COUNT(*),COALESCE(SUM(size),0) FROM media').fetchone()
                matches=db.execute('SELECT * FROM matches ORDER BY match_id').fetchall()
                jobs={}
                for job in db.execute('SELECT * FROM jobs ORDER BY rowid'):
                    jobs[job['match_id']]=f'{job["status"]} · {job["next_index"]} 张'+(' · '+job['error'] if job['error'] else '')
                for tree in (self.media_tree,self.match_tree):
                    tree.delete(*tree.get_children())
                for video in videos:self.media_tree.insert('', 'end',values=(video['path'],f'{video["size"]/1024**3:.2f}'))
                for match in matches:
                    self.match_tree.insert('', 'end',iid=match['match_id'],values=(match['match_id'],f'{match["start"]:g}–{match["end"]:g}',
                        match['red1']+' / '+match['red2'],match['blue1']+' / '+match['blue2'],jobs.get(match['match_id'],'未开始')))
                self.annotation_match_combo.configure(values=[m['match_id'] for m in matches])
                self.event_summary.set(f'{size[0]} 个原视频 · {size[1]/1024**3:.2f} GB · {len(matches)} 场 · 本地索引，不上传、不复制')
            self.refresh_teams()
        except Exception as error:
            self.status.set(f'赛事目录打开失败：{error}')

    def select_match(self, _=None):
        selected=self.match_tree.selection()
        if selected:self.match_id.set(selected[0]);self.refresh_teams()

    def refresh_teams(self):
        with connect(self.project_path.get()) as db:
            match=db.execute('SELECT * FROM matches WHERE match_id=?',(self.match_id.get(),)).fetchone()
            teams=['']+([match[k] for k in CSV_FIELDS[4:]] if match else [])
            self.team_combo.configure(values=teams)
            if self.annotation_team.get() not in teams:self.annotation_team.set('')
            if hasattr(self,'binding_view'):
                rows=db.execute('SELECT * FROM bindings WHERE match_id=? ORDER BY run_id,track_id,start_frame',(self.match_id.get(),)).fetchall()
                self.binding_view.configure(state='normal');self.binding_view.delete('1.0','end')
                for row in rows:self.binding_view.insert('end',f'{row["run_id"]} · ID {row["track_id"]} · 帧 {row["start_frame"]}–{row["end_frame"]} → {row["team"]}\n')
                self.binding_view.configure(state='disabled')

    def csv_template(self):
        target=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='matches.csv',filetypes=[('对阵表','*.csv')])
        if not target:return
        try:
            with connect(self.project_path.get()) as db:video=db.execute('SELECT path FROM media LIMIT 1').fetchone()
            with Path(target).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f);writer.writerow(CSV_FIELDS)
                writer.writerow(['Q1',video['path'] if video else 'video.mp4','00:00:00','00:02:00','100A','200B','300C','400D'])
            messagebox.showinfo('模板已保存','请用真实场次、视频内时间和四个队号替换示例，再导入。')
        except Exception as error:messagebox.showerror('保存失败',str(error))

    def load_matches(self):
        source=filedialog.askopenfilename(filetypes=[('对阵表','*.csv')])
        if source:
            try:
                total=import_matches(self.project_path.get(),source);self.refresh_event()
                self.status.set(f'已导入 {total} 场；同一队伍可以跨场换边。')
            except Exception as error:messagebox.showerror('导入失败',str(error))

    def event_extract(self):
        try:
            every=float(self.sample_every.get());limit=int(self.sample_limit.get());width=int(self.sample_width.get());budget=float(self.sample_budget.get())
            if not (every>0 and limit>0 and 320<=width<=3840 and budget>0):raise ValueError()
        except ValueError:
            messagebox.showerror('参数错误','填写正数间隔、张数和预算；宽度 320–3840。');return
        self.run_job([self.project_path.get(),'--every',str(every),'--limit',str(limit),'--width',str(width),'--budget-gb',str(budget)],script='event_worker.py')

    def annotate_event(self):
        if not self.confirm_dirty():return
        folder=Path(self.project_path.get())/'frames'
        self.files=sorted(folder.glob('*.jpg'))
        if not self.files:messagebox.showinfo('暂无图片','先按对阵表执行抽帧。');return
        self.index=0;self.label_dir.set(str(Path(self.project_path.get())/'annotations-v3'));self.load_image()
        self.status.set('图片已打开，请切换到“图片标注”。选择对应场次后再绑定队号。')

    def event_detect(self):
        if not self.match_id.get():messagebox.showinfo('先选择场次','先在赛事列表选择一场。');return
        self.run_job([self.project_path.get(),'--mode','detect','--match',self.match_id.get(),'--weights',self.weights.get(),
                      '--device',self.device.get(),'--conf',self.confidence.get()],script='event_worker.py')

    def binding_record(self):
        report=Path(self.identity_report.get())
        with report.open(encoding='utf-8') as f:record=json.loads(f.readline())
        if record.get('match_id')!=self.match_id.get():raise ValueError('记录不属于所选场次。')
        if not (report.parent/'complete.json').is_file():raise ValueError('单场检测尚未完整结束，请重新运行该场。')
        marker=json.loads((report.parent/'complete.json').read_text(encoding='utf-8'))
        if marker.get('match_id')!=record.get('match_id') or marker.get('run_id')!=record.get('run_id'):
            raise ValueError('单场完成标记与检测记录不一致。')
        return report,record['run_id']

    def save_binding(self):
        try:
            report,run_id=self.binding_record()
            first,last=int(self.binding_first.get()),int(self.binding_last.get())
            track=int(self.binding_track.get())
            complete=json.loads((report.parent/'complete.json').read_text(encoding='utf-8'))
            if not 0<=first<=last<complete['frames']:raise ValueError('帧区间超出本场检测范围。')
            found=False
            with report.open(encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line)
                    if first<=row['frame']<=last and any(obj['class']=='robot' and obj.get('track_id')==track for obj in row['detections']):
                        found=True;break
            if not found:raise ValueError('该区间没有这个 robot 跟踪 ID，请先预览确认。')
            bind_team(self.project_path.get(),self.match_id.get(),run_id,int(self.binding_track.get()),
                      int(self.binding_first.get()),int(self.binding_last.get()),self.binding_team.get())
            self.refresh_teams();self.status.set('队号绑定已保存。')
        except Exception as error:messagebox.showerror('绑定失败',str(error))

    def undo_binding(self):
        try:
            _,run_id=self.binding_record()
            remove_binding(self.project_path.get(),self.match_id.get(),run_id,int(self.binding_track.get()),int(self.binding_first.get()))
            self.refresh_teams();self.status.set('绑定已撤销；原记录保存在赛事项目的修改历史中。')
        except Exception as error:messagebox.showerror('撤销失败',str(error))

    def preview_identity(self):
        if not self.confirm_dirty():return
        try:
            import cv2
            report,_=self.binding_record()
            requested=int(self.binding_first.get())
            record=None
            with report.open(encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line)
                    if row['frame']==requested:record=row;break
            if record is None:raise ValueError('找不到这帧，请输入检测记录内的帧编号。')
            cap=cv2.VideoCapture(record['source'])
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES,record['source_frame']);ok,frame=cap.read()
                if not ok:raise ValueError('无法读取原视频这一帧')
                image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            finally:cap.release()
            scale=min(1000/image.width,620/image.height,1)
            popup=tk.Toplevel(self);popup.title(f'{record["match_id"]} · 帧 {requested} · 点击机器人框选择 ID')
            canvas=tk.Canvas(popup,width=int(image.width*scale),height=int(image.height*scale),bg='black');canvas.pack()
            canvas.photo=ImageTk.PhotoImage(image.resize((int(image.width*scale),int(image.height*scale))))
            canvas.create_image(0,0,image=canvas.photo,anchor='nw')
            targets=[]
            for obj in record['detections']:
                box=[n*scale for n in obj['xyxy']]
                color=COLORS[obj['class_id']]
                canvas.create_rectangle(*box,outline=color,width=2)
                canvas.create_text(box[0],box[1],text=f'{obj["class"]} ID {obj["track_id"]}',anchor='nw',fill=color)
                if obj['class']=='robot' and obj['track_id'] is not None:targets.append((box,obj['track_id']))
            def pick(event):
                for box,track in sorted(targets,key=lambda item:(item[0][2]-item[0][0])*(item[0][3]-item[0][1])):
                    if box[0]<=event.x<=box[2] and box[1]<=event.y<=box[3]:
                        self.binding_track.set(str(track));popup.destroy();return
            canvas.bind('<Button-1>',pick)
        except Exception as error:messagebox.showerror('预览失败',str(error))

    def export_binding(self):
        try:
            report,run_id=self.binding_record()
            target=filedialog.asksaveasfilename(defaultextension='.jsonl',filetypes=[('身份记录','*.jsonl')])
            if target:
                export_identities(self.project_path.get(),report,target,self.match_id.get(),run_id)
                self.status.set('已导出队号与本场阵营；未绑定的机器人保留待确认。')
        except Exception as error:messagebox.showerror('导出失败',str(error))

    def run_job(self, arguments, script='vex_yolo.py'):
        if self.process and self.process.poll() is None:
            messagebox.showinfo('任务运行中', '请先停止或等待当前任务完成。'); return
        import threading
        env = os.environ.copy()
        env['YOLO_CONFIG_DIR'] = str(ROOT/'runs/config')
        env['PYTHONIOENCODING'] = 'utf-8'
        try:
            self.process = subprocess.Popen([sys.executable, '-u', str(ROOT/script), *arguments],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as error:
            messagebox.showerror('启动失败', str(error)); return
        self.status.set('任务运行中：' + script)
        process = self.process
        def read():
            for line in process.stdout:
                self.messages.put(('log', line))
            self.messages.put(('done', process.wait()))
        threading.Thread(target=read, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == 'done':
                    self.status.set('任务完成' if value == 0 else f'任务已停止或失败（{value}），请查看日志')
                    self.refresh_event()
                elif kind == 'event':
                    self.status.set(value)
                    self.refresh_event()
                else:
                    self.log.configure(state='normal')
                    self.log.insert('end', value)
                    if int(self.log.index('end-1c').split('.')[0]) > 1500:
                        self.log.delete('1.0', '500.0')
                    self.log.see('end'); self.log.configure(state='disabled')
        except queue.Empty:
            pass
        if self.closing and (not self.process or self.process.poll() is not None):
            self.destroy(); return
        self.after(100, self.poll)

    def open_output(self):
        folder = getattr(self, 'last_output', Path(self.output.get()))
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            process = self.process
            self.after(2000, lambda: process.kill() if process.poll() is None else None)

    def close(self):
        if not self.confirm_dirty():
            return
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno('结束任务', '关闭软件会停止当前任务，继续关闭吗？'):
                return
            self.closing = True
            self.stop()
        else:
            self.destroy()


if __name__ == '__main__':
    Workbench().mainloop()
