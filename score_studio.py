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
from scoreboard_view import compose,player_picture,paint_time
from goal_memory import GoalMemoryTimeline,gate_scoring
from replay_timeline import ReplayTimeline


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
        style.configure('Transport.TButton',padding=(6,2),font=('Microsoft YaHei UI',9))
        self.history=[];self.history_key=None;self.photo=None;self.toggle_rois={}
        self.backend=Dashboard(self.master);self.backend.withdraw()
        self.backend.display_fps=15
        self.original_draw=self.backend.draw;self.backend.draw=self.update_view
        bar=ttk.Frame(self,padding=7,style='Studio.TFrame');bar.pack(fill='x')
        for title,action in (('打开本场比赛',self.backend.open_pilot),('打开其他结果',self.backend.open_files),
                             ('播放 / 暂停',self.backend.toggle_play),('上一帧',lambda:self.backend.seek(self.backend.frame-1)),
                             ('下一帧',lambda:self.backend.seek(self.backend.frame+1)),('复核 / 队号 / 标定',self.show_review),
                             ('将候选半段送入复核',self.review_candidates)):
            ttk.Button(bar,text=title,command=action,style='Studio.TButton').pack(side='left',padx=3)
        ttk.Button(bar,text='计分设置',command=self.toggle_settings,style='Studio.TButton').pack(side='right',padx=5)
        self.settings=ttk.Frame(self,style='Studio.TFrame')
        options=ttk.Frame(self.settings,padding=(10,0,10,5),style='Studio.TFrame');options.pack(fill='x')
        self.keep_placed=tk.BooleanVar(value=True)
        ttk.Checkbutton(options,text='已放置半段保持',variable=self.keep_placed,command=self.change_persistence).pack(side='left')
        ttk.Label(options,text='连续两次独立检测后保留数量；黄色按当前占领重新计分',style='Studio.TLabel').pack(side='left',padx=12)
        ttk.Button(options,text='从此帧清除半段记录',command=self.reset_persistence,style='Studio.TButton').pack(side='left')
        ttk.Button(options,text='本轮训练视频',command=self.open_batch,style='Studio.TButton').pack(side='left',padx=8)
        timing=ttk.Frame(self.settings,padding=(10,0,10,5),style='Studio.TFrame');timing.pack(fill='x')
        ttk.Button(timing,text='设当前帧为比赛开始',command=self.set_match_start,style='Studio.TButton').pack(side='left')
        ttk.Button(timing,text='取消开赛点',command=self.clear_match_start,style='Studio.TButton').pack(side='left',padx=6)
        self.timing_note=tk.StringVar(value='开赛点未设置：暂不计分')
        ttk.Label(timing,textvariable=self.timing_note,style='Studio.TLabel').pack(side='left',padx=10)
        self.screen=tk.Canvas(self,bg='#080e19',highlightthickness=0);self.screen.pack(fill='both',expand=True)
        self.screen.bind('<Configure>',lambda _:self.update_view())
        self.screen.bind('<Button-1>',self.select_card)
        transport=ttk.Frame(self,padding=(10,2),style='Studio.TFrame');transport.pack(fill='x')
        for label,delta in (('−5 秒',-5),('+5 秒',5)):
            ttk.Button(transport,text=label,command=lambda d=delta:self.jump(d),style='Transport.TButton').pack(side='left',padx=3)
        ttk.Label(transport,text='播放速度',style='Studio.TLabel').pack(side='left',padx=(14,4))
        self.speed=tk.StringVar(value='1×')
        speed=ttk.Combobox(transport,textvariable=self.speed,values=['0.25×','0.5×','0.75×','1×','1.5×','2×','4×'],width=7,state='readonly')
        speed.pack(side='left');speed.bind('<<ComboboxSelected>>',lambda _:self.backend.set_playback_rate(float(self.speed.get().rstrip('×'))))
        ttk.Button(transport,text='− 缩放',command=lambda:self.timeline.set_zoom(1/1.5),style='Transport.TButton').pack(side='right',padx=3)
        ttk.Button(transport,text='+ 缩放',command=lambda:self.timeline.set_zoom(1.5),style='Transport.TButton').pack(side='right',padx=3)
        ttk.Button(transport,text='展开 / 收起时间轴',command=lambda:self.timeline.toggle_size(),style='Transport.TButton').pack(side='right',padx=3)
        self.timeline=ReplayTimeline(self,on_seek=self.backend.seek,on_pause=self.backend.pause);self.timeline.pack(fill='x',padx=10)
        self.bind('<space>',self.key_play)
        self.bind('<Left>',lambda e:self.key_step(e,-1));self.bind('<Right>',lambda e:self.key_step(e,1))
        self.bind('<Shift-Left>',lambda e:self.key_jump(e,-5));self.bind('<Shift-Right>',lambda e:self.key_jump(e,5))
        self.note=tk.StringVar(value='打开本场比赛；仅显示 Goal 堆叠候选，地面物体不显示、不参与估分。')
        ttk.Label(self,textvariable=self.note,padding=6,style='Studio.TLabel').pack(fill='x')
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.backend.protocol('WM_DELETE_WINDOW',self.backend.withdraw)
        self.review_timer=self.after(150,self.poll_review)

    def load_history(self):
        b=self.backend
        enabled=b.session.get('keep_placed_halves',True)
        start=b.session.get('match_start_frame')
        self.keep_placed.set(enabled)
        key=(str(b.records.path),json.dumps(b.session['goal_rois'],sort_keys=True),enabled,
             json.dumps(b.session['reviews'],sort_keys=True),tuple(b.session.get('stack_memory_resets',[])),start)
        if key==self.history_key:return
        self.history_key=key;self.history=[]
        self.stack_timeline=None
        if enabled:
            self.stack_timeline=GoalMemoryTimeline(b.records,b.fps,b.session['goal_rois'],b.session['reviews'],b.session.get('stack_memory_resets',[]),start_frame=start)
            self.history=self.stack_timeline.history
        else:
            for i in range(0,len(b.records.offsets),max(1,round(b.fps))):
                row=b.records.get(i);observed=gate_scoring(observe_goal_stacks(row,b.session['goal_rois']),i,start)
                self.history.append(dict(seconds=row['seconds'],red=observed['score']['red'],blue=observed['score']['blue']))
        # Toggle calibration is local data, never embedded in the UI source.
        pointer=ROOT/'runs/active_toggle_pilot.json'
        first=b.records.get(0)
        if first.get('toggle_rois'):
            self.toggle_rois=first['toggle_rois']
        elif pointer.exists():
            target=json.loads(pointer.read_text(encoding='utf-8'))['directory']
            from pathlib import Path
            path=Path(target)/'toggle-rois.json'
            self.toggle_rois=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def update_view(self):
        b=self.backend
        if b.image is None:return
        if b.state()!='withdrawn':self.original_draw()
        self.load_history()
        observed=observe_goal_stacks(b.current,b.session['goal_rois'])
        if self.stack_timeline is not None:observed=self.stack_timeline.view(b.frame,observed)
        else:observed=gate_scoring(observed,b.frame,b.session.get('match_start_frame'))
        state=observed['match_timing']['state'];start=b.session.get('match_start_frame')
        self.timing_note.set('开赛点未设置：暂不计分' if state=='unset' else f'开赛点 {start/b.fps:.2f}s · '+('开赛前不计分' if state=='prestart' else '比赛中 · AI 估分待复核'))
        frame=cv2.resize(b.image,(1280,720)) if b.playing else b.image
        overlay,observed=goal_view(frame,b.current,self.toggle_rois,b.session['goal_rois'],observed)
        review=next((r for r in b.session['reviews'] if r['frame']==b.frame),None)
        if state!='active':review=None
        now=time.monotonic()
        # Render the video on each presentation tick, but reuse the expensive
        # text/chart panel briefly during playback. Pause and seek are exact.
        if not b.playing or now-getattr(self,'panel_time',-1e9)>=.2 or getattr(self,'panel_history_key',None)!=self.history_key:
            self.panel=compose(None,observed,b.roster,b.states,b.current.get('toggles',{}),self.history,review)
            self.panel_time=now;self.panel_history_key=self.history_key
        width,height=max(1,self.screen.winfo_width()),max(1,self.screen.winfo_height())
        panel=self.panel.copy();paint_time(panel,b.current['seconds'])
        picture,self.player_layout=player_picture(panel,Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)),width,height,b.playing)
        self.scale=1;self.offset=(0,0)
        self.photo=ImageTk.PhotoImage(picture)
        self.screen.delete('all');self.screen.create_image(*self.offset,image=self.photo,anchor='nw')
        self.timeline.configure_clip(b.session['source']['video']['path'],len(b.records.offsets),b.fps,self.history,start)
        self.timeline.position(b.frame)
        self.note.set(f'视频 {b.current["seconds"]:.2f}s · 已隐藏 {observed["excluded_ground_or_unconnected"]} 个地面 / 未连接候选 · AI 估分尚未确认落位、可见性和有效占领')
        self.observed=observed

    def toggle_settings(self):
        if self.settings.winfo_manager():self.settings.pack_forget()
        else:self.settings.pack(fill='x',before=self.screen)

    def jump(self,seconds):
        b=self.backend
        if b.records:b.seek(b.frame+round(seconds*b.fps))

    def key_play(self,event):
        if isinstance(event.widget,(tk.Entry,ttk.Entry,ttk.Combobox)):return
        self.backend.toggle_play();return 'break'

    def key_step(self,event,delta):
        if isinstance(event.widget,(tk.Entry,ttk.Entry,ttk.Combobox)):return
        if self.backend.records:self.backend.seek(self.backend.frame+delta)
        return 'break'

    def key_jump(self,event,delta):
        if isinstance(event.widget,(tk.Entry,ttk.Entry,ttk.Combobox)):return
        self.jump(delta);return 'break'

    def set_match_start(self):
        b=self.backend
        if b.current is None:return
        b.pause();b.session['match_start_frame']=b.frame;b.save_session()
        self.history_key=None;self.update_view()

    def clear_match_start(self):
        b=self.backend
        if b.current is None:return
        b.pause();b.session.pop('match_start_frame',None);b.save_session()
        self.history_key=None;self.update_view()

    def change_persistence(self):
        self.backend.pause()
        self.backend.session['keep_placed_halves']=self.keep_placed.get()
        if self.backend.path:self.backend.save_session()
        self.history_key=None;self.update_view()

    def reset_persistence(self):
        b=self.backend
        if b.current is None:return
        b.pause();b.session.setdefault('stack_memory_resets',[]).append(b.frame)
        b.save_session();self.history_key=None;self.update_view()

    def open_batch(self):
        from tkinter import messagebox
        pointer=ROOT/'runs/latest_elims_results.json'
        if not pointer.exists():
            messagebox.showinfo('本轮训练视频','本轮结果尚未生成。',parent=self);return
        state=json.loads(pointer.read_text(encoding='utf-8'))
        dialog=tk.Toplevel(self);dialog.title('本轮训练视频');dialog.geometry('900x320')
        tree=ttk.Treeview(dialog,columns=('match','red','blue','status'),show='headings')
        for key,title in [('match','场次'),('red','红方'),('blue','蓝方'),('status','结果状态')]:
            tree.heading(key,text=title);tree.column(key,width=210)
        tree.pack(fill='both',expand=True,padx=10,pady=10)
        videos=state.get('videos',[])
        for i,video in enumerate(videos):tree.insert('', 'end',iid=str(i),values=(video['match'],' / '.join(video['red']),' / '.join(video['blue']),'可打开' if video['status']=='complete' else '处理中'))
        def load():
            selection=tree.selection()
            if not selection:return
            video=videos[int(selection[0])]
            if video['status']!='complete':return
            try:self.backend.load(video['source'],video['records']);dialog.destroy()
            except Exception as error:messagebox.showerror('打开失败',str(error),parent=dialog)
        ttk.Button(dialog,text='打开选中场次',command=load).pack(pady=5)
        tree.bind('<Double-1>',lambda _:load())

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
        x=event.x;y=event.y
        b=self.backend
        if not hasattr(self,'player_layout'):return
        vx,vy,vw,vh=self.player_layout['video']
        if b.current is not None and vx<=x<=vx+vw and vy<=y<=vy+vh:
            nx=(x-vx)/vw;ny=(y-vy)/vh
            hits=[o for o in b.current['robots'] if o['xyxy'][0]<=nx<=o['xyxy'][2] and o['xyxy'][1]<=ny<=o['xyxy'][3]]
            if hits:
                b.pause();obj=min(hits,key=lambda o:(o['xyxy'][2]-o['xyxy'][0])*(o['xyxy'][3]-o['xyxy'][1]))
                menu=tk.Menu(self,tearoff=False)
                for team,color in b.roster.items():
                    menu.add_command(label=f'设为 {team} · '+('红方' if color=='red' else '蓝方'),command=lambda t=team,tid=obj['track_id']:self.bind_robot(tid,t))
                menu.tk_popup(event.x_root,event.y_root);return
        sx,sy,sw,sh=self.player_layout['sidebar']
        if not(sx<=x<=sx+sw and sy<=y<=sy+sh):return
        x=1066+(x-sx)/sw*518;y=103+(y-sy)/sh*833
        if 1070<=x<1574 and 373<=y<752:
            col=int((x-1070)//172);row=int((y-373)//130)
            if col<3 and row<3:
                self.backend.goal.set(f'G{row*3+col}');self.backend.load_goal();self.show_review()

    def bind_robot(self,track_id,team):
        b=self.backend;b.pause()
        b.session['bindings'].append(dict(frame=b.frame,track_id=track_id,team=team,source='user_video_click'))
        b.save_session();b.seek(b.frame)

    def poll_review(self):
        if not self.winfo_exists():return
        signature=json.dumps([self.backend.session['reviews'],self.backend.session['bindings']],sort_keys=True)
        if signature!=getattr(self,'review_signature',None):
            self.review_signature=signature;self.update_view()
        self.review_timer=self.after(200,self.poll_review)

    def close(self):
        if getattr(self,'review_timer',None) is not None:self.after_cancel(self.review_timer)
        self.timeline.close()
        self.backend.close();self.destroy()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',action='store_true',help='Open the local batch result selector')
    args=parser.parse_args()
    root=tk.Tk();root.withdraw();studio=ScoreStudio(root)
    studio.protocol('WM_DELETE_WINDOW',lambda:(studio.close(),root.destroy()))
    studio.after(100,studio.open_batch if args.batch else studio.backend.open_pilot)
    root.mainloop()
