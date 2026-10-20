"""Video-first scoreboard UI. Existing review controls remain available separately."""
import json
import time
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image,ImageTk
from live_dashboard import Dashboard,ROOT
from goal_overlay import goal_view
from goal_observer import observe_goal_stacks
from scoreboard_view import compose


class ScoreStudio(tk.Toplevel):
    def __init__(self,master=None):
        super().__init__(master)
        self.title('miko yolo · 实时比分看板')
        self.geometry('1600x1010');self.minsize(1100,760);self.configure(bg='#080e19')
        style=ttk.Style(self);style.theme_use('clam')
        style.configure('Studio.TFrame',background='#080e19')
        style.configure('Studio.TLabel',background='#080e19',foreground='#a8b9d0',font=('Microsoft YaHei UI',9))
        style.configure('Studio.TButton',background='#17243a',foreground='#edf3ff',padding=7,font=('Microsoft YaHei UI',9))
        style.map('Studio.TButton',background=[('active','#263b5b')])
        self.history=[];self.history_key=None;self.photo=None;self.toggle_rois={}
        self.backend=Dashboard(self.master);self.backend.withdraw()
        self.original_draw=self.backend.draw;self.backend.draw=self.update_view
        bar=ttk.Frame(self,padding=7,style='Studio.TFrame');bar.pack(fill='x')
        for title,action in (('打开本场比赛',self.backend.open_pilot),('打开其他结果',self.backend.open_files),
                             ('播放 / 暂停',self.backend.toggle_play),('上一帧',lambda:self.backend.seek(self.backend.frame-1)),
                             ('下一帧',lambda:self.backend.seek(self.backend.frame+1)),('复核 / 队号 / 标定',self.show_review),
                             ('将候选半段送入复核',self.review_candidates)):
            ttk.Button(bar,text=title,command=action,style='Studio.TButton').pack(side='left',padx=3)
        ttk.Label(bar,text='≈ 为估分，? 为待确认',style='Studio.TLabel').pack(side='right',padx=8)
        self.screen=tk.Canvas(self,bg='#080e19',highlightthickness=0);self.screen.pack(fill='both',expand=True)
        self.screen.bind('<Configure>',lambda _:self.update_view())
        self.screen.bind('<Button-1>',self.select_card)
        self.slider=ttk.Scale(self,from_=0,to=1);self.slider.pack(fill='x',padx=16,pady=4)
        self.slider.bind('<ButtonPress-1>',lambda _:self.backend.pause())
        self.slider.bind('<ButtonRelease-1>',lambda _:self.backend.seek(round(self.slider.get())))
        self.note=tk.StringVar(value='打开本场比赛；仅显示 Goal 堆叠候选，地面物体不显示、不参与估分。')
        ttk.Label(self,textvariable=self.note,padding=6,style='Studio.TLabel').pack(fill='x')
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.backend.protocol('WM_DELETE_WINDOW',self.backend.withdraw)
        self.review_timer=self.after(150,self.poll_review)

    def load_history(self):
        key=(str(self.backend.records.path),json.dumps(self.backend.session['goal_rois'],sort_keys=True))
        if key==self.history_key:return
        self.history_key=key;self.history=[]
        for i in range(0,len(self.backend.records.offsets),max(1,round(self.backend.fps))):
            row=self.backend.records.get(i)
            observed=observe_goal_stacks(row,self.backend.session['goal_rois'])
            self.history.append(dict(seconds=row['seconds'],red=observed['score']['red'],blue=observed['score']['blue']))
        # Toggle calibration is local data, never embedded in the UI source.
        pointer=ROOT/'runs/active_toggle_pilot.json'
        if pointer.exists():
            target=json.loads(pointer.read_text(encoding='utf-8'))['directory']
            from pathlib import Path
            path=Path(target)/'toggle-rois.json'
            self.toggle_rois=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def update_view(self):
        b=self.backend
        if b.image is None:return
        now=time.monotonic()
        if b.playing and now-getattr(self,'last_paint',0)<1/15:return
        self.last_paint=now
        if b.state()!='withdrawn':self.original_draw()
        self.load_history()
        overlay,observed=goal_view(b.image,b.current,self.toggle_rois,b.session['goal_rois'])
        review=next((r for r in b.session['reviews'] if r['frame']==b.frame),None)
        picture=compose(Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)),observed,b.roster,b.states,b.current.get('toggles',{}),self.history,review)
        width,height=max(1,self.screen.winfo_width()),max(1,self.screen.winfo_height())
        scale=min(width/1600,height/960);size=(max(1,round(1600*scale)),max(1,round(960*scale)))
        self.scale=scale;self.offset=((width-size[0])//2,(height-size[1])//2)
        self.photo=ImageTk.PhotoImage(picture.resize(size,Image.Resampling.LANCZOS))
        self.screen.delete('all');self.screen.create_image(*self.offset,image=self.photo,anchor='nw')
        self.slider.configure(to=len(b.records.offsets)-1);self.slider.set(b.frame)
        self.note.set(f'视频 {b.current["seconds"]:.2f}s · 已隐藏 {observed["excluded_ground_or_unconnected"]} 个地面 / 未连接候选 · AI 估分尚未确认落位、可见性和有效占领')
        self.observed=observed

    def show_review(self):
        self.backend.pause();self.backend.deiconify();self.original_draw();self.backend.lift()

    def review_candidates(self):
        b=self.backend
        if b.current is None:return
        b.pause();observed=observe_goal_stacks(b.current,b.session['goal_rois'])
        for goal,row in observed['goals'].items():
            b.draft[goal]=dict(row['counts']) if row['status']=='estimate' else dict.fromkeys(('red','blue','yellow'),None)
        b.refresh_scores();b.load_goal();self.show_review()
        b.pending_candidate_review=True
        b.review_time.set('候选数量已填入；请逐个核查 SC2/SC3 和占领状态，再保存。尚未确认为有效得分。')

    def select_card(self,event):
        if not getattr(self,'scale',0):return
        x=(event.x-self.offset[0])/self.scale;y=(event.y-self.offset[1])/self.scale
        if 1070<=x<1574 and 373<=y<752:
            col=int((x-1070)//172);row=int((y-373)//130)
            if col<3 and row<3:
                self.backend.goal.set(f'G{row*3+col}');self.backend.load_goal();self.show_review()

    def poll_review(self):
        if not self.winfo_exists():return
        signature=json.dumps(self.backend.session['reviews'],sort_keys=True)
        if signature!=getattr(self,'review_signature',None):
            self.review_signature=signature;self.update_view()
        self.review_timer=self.after(200,self.poll_review)

    def close(self):
        if getattr(self,'review_timer',None) is not None:self.after_cancel(self.review_timer)
        self.backend.close();self.destroy()


if __name__=='__main__':
    root=tk.Tk();root.withdraw();studio=ScoreStudio(root)
    studio.protocol('WM_DELETE_WINDOW',lambda:(studio.close(),root.destroy()))
    studio.after(100,studio.backend.open_pilot)
    root.mainloop()
