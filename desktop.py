"""Local VEX Override annotation and detection workbench (Tkinter)."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ImageOps
from schema import CLASSES, COLORS, REGIONS, ensure_schema
from scoring import yellow_score

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
        self.files, self.boxes = [], []
        self.index, self.image, self.photo = 0, None, None
        self.dirty = False
        self.process = None
        self.messages = queue.Queue()
        self.closing = False
        self.scale, self.offset = 1, (0, 0)
        self.start = None
        self.current_class = tk.StringVar(value='robot_red')
        self.label_dir = tk.StringVar(value=str(ROOT / 'data' / 'annotations-v2'))
        self.status = tk.StringVar(value='新版类别：红蓝机器人、Toggle 状态、Goal、彩色 Pin Half、Cup')
        header = ttk.Frame(self, padding=16)
        header.pack(fill='x')
        ttk.Label(header, text='VEX YOLO  /  OVERRIDE', font=('Microsoft YaHei UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(header, text='Override 2.1 · 红蓝机器人跟踪 · Toggle 状态标注 · 黄色 Pin Half 归属估算').pack(anchor='w', pady=(6, 0))
        tabs = ttk.Notebook(self)
        tabs.pack(fill='both', expand=True, padx=16)
        annotation = ttk.Frame(tabs, padding=12)
        jobs = ttk.Frame(tabs, padding=16)
        scoring = ttk.Frame(tabs, padding=16)
        tabs.add(annotation, text='图片标注')
        tabs.add(jobs, text='抽帧 / 训练 / 检测')
        tabs.add(scoring, text='区域与黄色半段计分')
        self.build_annotation(annotation)
        self.build_jobs(jobs)
        self.build_scoring(scoring)
        ttk.Label(self, textvariable=self.status, padding=12).pack(fill='x')
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.after(100, self.poll)

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
        ttk.Label(bar, text='四台车按阵营分类；四个 Toggle 按颜色分类；九个 Goal 都选 goal。编号由跟踪/场地映射负责。').pack(side='left')

    def choose_labels(self):
        if not self.confirm_dirty():
            return
        folder = filedialog.askdirectory(title='选择 YOLO 标签保存文件夹')
        if folder:
            self.label_dir.set(folder)
            if self.files:
                self.load_image()

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
            ensure_schema(self.label_dir.get(), create=True)
            with Image.open(self.files[self.index]) as image:
                self.image = ImageOps.exif_transpose(image).convert('RGB')
            label = self.label_path()
            self.boxes = decode_boxes(label.read_text(encoding='utf-8'), *self.image.size) if label.exists() else []
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
        for c, x1, y1, x2, y2 in self.boxes:
            ox, oy = self.offset
            coords = (ox+x1*self.scale, oy+y1*self.scale, ox+x2*self.scale, oy+y2*self.scale)
            self.canvas.create_rectangle(*coords, outline=COLORS[c], width=2)
            self.canvas.create_text(coords[0]+4, coords[1]+4, text=CLASSES[c], anchor='nw', fill=COLORS[c], font=('Arial', 12, 'bold'))

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
            self.boxes.append((CLASSES.index(self.current_class.get()), x1, y1, x2, y2))
            self.dirty = True
        self.redraw()
        self.update_status()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
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
            temp.replace(target)
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
        self.dataset = self.field(parent, '训练数据文件夹', str(ROOT/'data/override-v2'), lambda: filedialog.askdirectory())
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

    def run_job(self, arguments):
        if self.process and self.process.poll() is None:
            messagebox.showinfo('任务运行中', '请先停止或等待当前任务完成。'); return
        import threading
        env = os.environ.copy()
        env['YOLO_CONFIG_DIR'] = str(ROOT/'runs/config')
        env['PYTHONIOENCODING'] = 'utf-8'
        try:
            self.process = subprocess.Popen([sys.executable, '-u', str(ROOT/'vex_yolo.py'), *arguments],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as error:
            messagebox.showerror('启动失败', str(error)); return
        self.status.set('任务运行中：' + arguments[0])
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
