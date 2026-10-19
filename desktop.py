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

ROOT = Path(__file__).resolve().parent
CLASSES = ('robot', 'pin', 'cup')
COLORS = ('#38bdf8', '#fbbf24', '#fb7185')


def encode_boxes(boxes, width, height):
    return ''.join(f'{c} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} '
                   f'{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n'
                   for c, x1, y1, x2, y2 in boxes)


def decode_boxes(text, width, height):
    boxes = []
    for line in text.splitlines():
        c, x, y, w, h = map(float, line.split())
        if int(c) != c or not 0 <= c < 3 or not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
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
        self.current_class = tk.StringVar(value='robot')
        self.label_dir = tk.StringVar(value=str(ROOT / 'data' / 'annotations'))
        self.status = tk.StringVar(value='打开图片文件夹，开始标注；robot / pin / cup')
        header = ttk.Frame(self, padding=16)
        header.pack(fill='x')
        ttk.Label(header, text='VEX YOLO  /  OVERRIDE', font=('Microsoft YaHei UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(header, text='本地标注 · 素材准备 · 模型检测     |     专用模型尚需训练，当前不提供自动预标注').pack(anchor='w', pady=(6, 0))
        tabs = ttk.Notebook(self)
        tabs.pack(fill='both', expand=True, padx=16)
        annotation = ttk.Frame(tabs, padding=12)
        jobs = ttk.Frame(tabs, padding=16)
        tabs.add(annotation, text='图片标注')
        tabs.add(jobs, text='抽帧 / 训练 / 检测')
        self.build_annotation(annotation)
        self.build_jobs(jobs)
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
        ttk.Combobox(bar, values=CLASSES, textvariable=self.current_class, state='readonly', width=9).pack(side='left', padx=8)
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
        ttk.Label(bar, text='拖动鼠标画框；选错类别或框选范围可撤销后重画。无目标图片也需保存空标注。').pack(side='left')

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
        self.dataset = self.field(parent, '训练数据文件夹', str(ROOT/'data/override'), lambda: filedialog.askdirectory())
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
        self.run_job(['predict', '--weights', self.weights.get(), '--source', self.source.get(),
                      '--conf', str(conf), '--device', self.device.get(), '--output', str(self.last_output), '--show'])

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
