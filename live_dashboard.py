"""Synchronized local replay and scoring review panel.

Video is referenced in place; JSONL is indexed by byte offsets. No media copies.
This panel consumes existing identity/Toggle results. It does not pretend that
untrained Goal/Pin classes can already supply automatic scoring observations.
"""
import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from match_state import TeamMemory, match_score
from schema import REGIONS

ROOT = Path(__file__).resolve().parent
ROSTER = {'R1':'red', 'R2':'red', 'B1':'blue', 'B2':'blue'}
COLORS = {'red':'#fa6060','blue':'#50b8ff','yellow':'#f9da52','unknown':'#a1a1aa'}
STATUS = {'observed':'本帧已关联','predicted':'短时预测','lost':'遮挡 / 待重新确认','waiting':'等待首次确认'}
OWNER_LABELS = {'待确认':'unknown','红方有效占领':'red','蓝方有效占领':'blue','中立':'neutral'}


class RecordIndex:
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = []
        self.times = []
        self.roster = {}
        self.handle = self.path.open('rb')
        try:
            while True:
                offset = self.handle.tell()
                line = self.handle.readline()
                if not line:
                    break
                row = json.loads(line)
                if not isinstance(row.get('robots'),list):
                    raise ValueError('需要含 robots 的归一化身份结果；原始检测 JSONL 请先完成队伍绑定与转换')
                listed=row.get('roster',{})
                if not isinstance(listed,dict) or any(c not in ('red','blue') for c in listed.values()):
                    raise ValueError('无效的本场队伍名单')
                for team,color in listed.items():
                    if team in self.roster and self.roster[team]!=color:raise ValueError('同一视频内名单阵营不一致')
                    self.roster[team]=color
                for obj in row['robots']:
                    team,color=obj.get('team'),obj.get('alliance')
                    if team and color in ('red','blue'):
                        if listed and team not in listed:raise ValueError('机器人队号不属于本场名单')
                        if team in self.roster and self.roster[team] != color:
                            raise ValueError('同一视频内队伍阵营发生变化；请先按单场比赛切分结果')
                        self.roster[team]=color
                if row['frame'] != len(self.offsets) or (self.times and row['seconds'] < self.times[-1]):
                    raise ValueError('检测结果必须从第 0 帧开始连续排列，且时间递增')
                self.offsets.append(offset)
                self.times.append(float(row['seconds']))
            if not self.offsets:
                raise ValueError('检测结果为空')
        except BaseException:
            self.close()
            raise

    def get(self, index):
        self.handle.seek(self.offsets[index])
        return json.loads(self.handle.readline())

    def close(self):
        self.handle.close()


class Dashboard(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('miko yolo · 跟踪与计分复核')
        self.geometry('1480x930')
        self.minsize(1100,760)
        self.configure(bg='#101827')
        self.capture = self.records = None
        self.frame = 0
        self.playing = False
        self.timer = None
        self.memory = TeamMemory(ROSTER)
        self.roster = dict(ROSTER)
        self.session = {'version':1,'roster':self.roster,'bindings':[],'goal_rois':{},'reviews':[]}
        self.image = self.photo = self.current = None
        self.states = []
        self.start = None
        self.seek_active = False
        self.mode = tk.StringVar(value='选择机器人')
        self.team = tk.StringVar(value=next(iter(ROSTER)))
        self.goal = tk.StringVar(value='G0')
        self.status = tk.StringVar(value='请打开本场试播，或选择视频及对应检测 JSONL。原视频不复制、不改写。')
        self.score_title = tk.StringVar(value='红 —    蓝 —')
        self.review_time = tk.StringVar(value='尚无得分复核；Goal / Pin 自动计分模型尚未训练')
        self.path = None
        self.build()
        self.protocol('WM_DELETE_WINDOW', self.close)

    def build(self):
        bar = ttk.Frame(self,padding=8);bar.pack(fill='x')
        for title, action in (('打开本场试播',self.open_pilot),('打开视频与结果',self.open_files),('播放 / 暂停',self.toggle_play),('上一帧',lambda:self.seek(self.frame-1)),('下一帧',lambda:self.seek(self.frame+1))):
            ttk.Button(bar,text=title,command=action).pack(side='left',padx=3)
        ttk.Label(bar,text='纠正队号').pack(side='left',padx=(12,3))
        self.team_combo = ttk.Combobox(bar,textvariable=self.team,values=list(self.roster),state='readonly',width=9)
        self.team_combo.pack(side='left')
        ttk.Combobox(bar,textvariable=self.mode,values=['选择机器人','标定 Goal'],state='readonly',width=12).pack(side='left',padx=5)
        goal_combo=ttk.Combobox(bar,textvariable=self.goal,values=[f'G{i}' for i in range(9)],state='readonly',width=5)
        goal_combo.pack(side='left');goal_combo.bind('<<ComboboxSelected>>',lambda _:self.load_goal())
        ttk.Button(bar,text='取本帧标注 Goal / Pin',command=self.export_sample).pack(side='left',padx=5)
        pane=ttk.Panedwindow(self,orient='horizontal');pane.pack(fill='both',expand=True,padx=8)
        left=ttk.Frame(pane); right=ttk.Frame(pane,padding=8);pane.add(left,weight=3);pane.add(right,weight=2)
        self.canvas=tk.Canvas(left,bg='#070e1c',highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda _:self.draw())
        self.canvas.bind('<ButtonPress-1>',self.press)
        self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<ButtonRelease-1>',self.release)
        self.slider=ttk.Scale(left,from_=0,to=1,command=self.slide);self.slider.pack(fill='x',pady=5)
        self.slider.bind('<ButtonPress-1>',lambda _:self.pause())
        self.slider.bind('<ButtonRelease-1>',lambda _:self.seek(round(self.slider.get())))
        ttk.Label(left,text='实线：本帧关联；虚线：最多 1.5 秒预测；失联后保留队伍卡片。\n暂停后点击机器人框纠正队号；标定 Goal 模式中拖框设置 G0–G8。').pack(anchor='w')
        self.team_labels=[]
        for team,color in self.roster.items():
            label=ttk.Label(right,text=f'{team} | {color.upper()} | 等待',foreground=COLORS[color])
            label.pack(anchor='w',pady=3);self.team_labels.append(label)
        ttk.Separator(right).pack(fill='x',pady=7)
        ttk.Label(right,textvariable=self.score_title,font=('Microsoft YaHei UI',23,'bold')).pack(anchor='w')
        ttk.Label(right,textvariable=self.review_time,wraplength=440).pack(anchor='w',pady=4)
        self.table=ttk.Treeview(right,columns=('region','r','b','y','owner','score'),show='headings',height=9,selectmode='browse')
        for col,title,width in zip(('region','r','b','y','owner','score'),('Goal / 区域','红半','蓝半','黄半','黄归属','红 / 蓝分'),(125,42,42,42,65,85)):
            self.table.heading(col,text=title);self.table.column(col,width=width,stretch=col=='region')
        self.table.pack(fill='x',pady=5)
        self.table.bind('<<TreeviewSelect>>',self.select_goal)
        entry=ttk.Frame(right);entry.pack(fill='x',pady=4)
        ttk.Label(entry,text='当前 Goal').pack(side='left');ttk.Label(entry,textvariable=self.goal).pack(side='left',padx=4)
        self.counts={c:tk.StringVar() for c in ('red','blue','yellow')}
        for color,title in (('red','红'),('blue','蓝'),('yellow','黄')):
            ttk.Label(entry,text=title).pack(side='left');ttk.Entry(entry,textvariable=self.counts[color],width=4).pack(side='left',padx=3)
        ttk.Button(entry,text='应用半段数',command=self.apply_goal).pack(side='left',padx=4)
        self.valid=tk.BooleanVar(value=False)
        ttk.Checkbutton(right,text='已按 SC2/SC3 核查：只计 Placed Pin 的可见半段',variable=self.valid).pack(anchor='w')
        ttk.Label(right,text='空白 = 未知；0 = 已确认没有。Cup 本身不直接加分。',wraplength=440).pack(anchor='w')
        self.owners={}
        for region in REGIONS:
            row=ttk.Frame(right);row.pack(fill='x',pady=2)
            ttk.Label(row,text=f'{region} 有效占领',width=17).pack(side='left')
            var=tk.StringVar(value='待确认');self.owners[region]=var
            ttk.Combobox(row,textvariable=var,values=list(OWNER_LABELS),state='readonly',width=15).pack(side='left')
        ttk.Label(right,text='SC4：Toggle 完全就位、静止且无机器人接触，才有效。',wraplength=440).pack(anchor='w',pady=3)
        mid=ttk.Frame(right);mid.pack(fill='x')
        self.midred=tk.StringVar();self.midblue=tk.StringVar();self.phase=tk.StringVar(value='end')
        for title,var in (('中央红车',self.midred),('蓝车',self.midblue)):
            ttk.Label(mid,text=title).pack(side='left');ttk.Entry(mid,textvariable=var,width=3).pack(side='left',padx=3)
        ttk.Combobox(mid,textvariable=self.phase,values=['end','autonomous'],state='readonly',width=12).pack(side='left',padx=4)
        bonus=ttk.Frame(right);bonus.pack(fill='x',pady=4)
        self.bonusred=tk.StringVar();self.bonusblue=tk.StringVar()
        for title,var in (('自动奖励 红',self.bonusred),('蓝',self.bonusblue)):
            ttk.Label(bonus,text=title).pack(side='left');ttk.Combobox(bonus,textvariable=var,values=['','0','6','12'],state='readonly',width=4).pack(side='left',padx=3)
        ttk.Button(right,text='保存本帧计分复核快照',command=self.save_review).pack(fill='x',pady=4)
        ttk.Label(right,text='G0 中央；G1–2 red1；G3–4 red2；G5–6 blue1；G7–8 blue2。\n先按场地区域标定，区域名不代表当前占领阵营。中央按 SC6 的立体边界人工复核。\nend：终场估分含中央机器人各 8 分；autonomous：排除中央机器人和中央黄色分，不加自动奖励。\n实时画面仅供估算；SC1 最终状态按终场后规则判定，裁判结果为准。',wraplength=440).pack(anchor='w',pady=4)
        ttk.Label(self,textvariable=self.status,padding=8,wraplength=1400).pack(fill='x')
        self.draft={f'G{i}':dict(red=None,blue=None,yellow=None) for i in range(9)}
        self.refresh_scores()

    def pause(self):
        self.playing=False
        if self.timer is not None:
            self.after_cancel(self.timer);self.timer=None

    def open_pilot(self):
        try:
            directory=Path(json.loads((ROOT/'runs/active_toggle_pilot.json').read_text(encoding='utf-8'))['directory'])
            source=json.loads((directory/'render-summary.json').read_text(encoding='utf-8'))['source']
            records=directory/'corrected-identities.jsonl'
            goal_pointer=ROOT/'runs/active_goal_pilot.json'
            if goal_pointer.exists():
                goal_state=json.loads(goal_pointer.read_text(encoding='utf-8'))
                candidate=Path(goal_state.get('preview_jsonl',''))
                if goal_state.get('source')==source and candidate.is_file():records=candidate
            self.load(source,records)
        except Exception as error:
            messagebox.showerror('无法打开本场试播',str(error),parent=self)

    def open_files(self):
        video=filedialog.askopenfilename(parent=self,title='选择原视频',filetypes=[('视频','*.mp4 *.mkv *.avi *.mov'),('全部','*')])
        if not video:return
        records=filedialog.askopenfilename(parent=self,title='选择与该原视频逐帧对应的身份 / Toggle JSONL',filetypes=[('检测结果','*.jsonl')])
        if records:
            try:self.load(video,records)
            except Exception as error:messagebox.showerror('打开失败',str(error),parent=self)

    def load(self, video, records):
        self.pause()
        capture=cv2.VideoCapture(str(video)); indexed=None
        try:
            if not capture.isOpened():raise ValueError('视频无法读取')
            indexed=RecordIndex(records)
            fps=capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(indexed.offsets):
                raise ValueError('视频与结果帧数不一致，不能安全对齐')
            if any(abs(t-i/fps) > max(.1,2/fps) for i,t in enumerate(indexed.times)):
                raise ValueError('结果时间与视频帧率不匹配')
            video_path=Path(video).resolve();record_path=Path(records).resolve()
            signature={name:dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for name,p in (('video',video_path),('records',record_path))}
            # Keep all generated sessions inside the workspace, even for Z: media.
            import hashlib
            key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]
            path=ROOT/'runs/dashboard'/f'{key}.json'
            TeamMemory(indexed.roster)
            session=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(version=1,roster=indexed.roster,bindings=[],goal_rois=indexed.get(0).get('goal_rois',{}),reviews=[],source=signature)
            if session.get('source') != signature:raise ValueError('复核文件属于不同视频 / 检测结果')
            memory=TeamMemory(session['roster'])
        except BaseException:
            capture.release()
            if indexed:indexed.close()
            raise
        if self.capture:self.capture.release()
        if self.records:self.records.close()
        self.capture,self.records,self.fps,self.path,self.session,self.memory=capture,indexed,fps,path,session,memory
        self.roster=dict(session['roster']);self.team_combo.configure(values=list(self.roster));self.team.set(next(iter(self.roster)))
        self.slider.configure(to=len(indexed.offsets)-1)
        self.seek(0)

    def save_session(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.session,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(self.path)

    def export_sample(self):
        if self.image is None:return
        self.pause()
        parent=self.master
        if hasattr(parent,'confirm_dirty') and not parent.confirm_dirty():return
        try:
            folder=ROOT/'data/goal-review'/self.path.stem
            images=folder/'images';images.mkdir(parents=True,exist_ok=True)
            target=images/f'goal-{self.path.stem}__{self.frame:08d}.jpg'
            if not target.exists():
                ok,encoded=cv2.imencode('.jpg',self.image,[cv2.IMWRITE_JPEG_QUALITY,95])
                if not ok:raise ValueError('保存图片失败')
                with target.open('xb') as handle:handle.write(encoded.tobytes())
            from schema import ensure_schema
            labels=folder/'annotations-v3';ensure_schema(labels,create=True)
            if hasattr(parent,'load_image'):
                parent.files=sorted(images.glob('*.jpg'));parent.index=parent.files.index(target)
                parent.label_dir.set(str(labels));parent.current_class.set('goal');parent.load_image()
                self.status.set('当前原始帧已打开到主工作台“图片标注”。请标 goal、三个颜色 pin_half 和 cup；缺少可见性 / Placed 状态时仍须复核。')
            else:
                self.status.set(f'标注图片：{target}；标签目录：{labels}')
        except Exception as error:messagebox.showerror('取帧失败',str(error),parent=self)

    def corrected(self, row):
        row=dict(row,robots=[dict(o) for o in row.get('robots',[])])
        overrides={}
        for binding in sorted(self.session['bindings'],key=lambda b:b['frame']):
            if binding['frame'] <= row['frame'] <= binding.get('end_frame',float('inf')):
                overrides[binding['track_id']]=binding
        # Only the newest reviewed track can own each team at a given time.
        newest={}
        for binding in overrides.values():
            if binding['team'] not in newest or binding['frame'] >= newest[binding['team']]['frame']:
                newest[binding['team']]=binding
        applied=set()
        for obj in row['robots']:
            binding=overrides.get(obj.get('track_id'))
            if binding:
                if newest[binding['team']] is binding:
                    obj.update(team=binding['team'],alliance=self.roster[binding['team']],identity_source='manual_review')
                    applied.add(binding['team'])
                else:
                    obj.update(team=None,alliance=None,identity_source='superseded_review')
        # An explicit current-box correction takes priority over a stale automatic assignment.
        for obj in row['robots']:
            if obj.get('team') in applied and obj.get('identity_source') != 'manual_review':
                obj.update(team=None,alliance=None)
        # A roster is fixed for this match. Never turn a blue team red because
        # of an appearance guess, or display the same team on two robots.
        owners={}
        for obj in sorted(row['robots'],key=lambda o:(o.get('identity_source')=='manual_review',o.get('confidence',0)),reverse=True):
            team=obj.get('team')
            if team in self.roster and team not in owners:
                obj['alliance']=self.roster[team];owners[team]=obj
            elif team is not None:obj.update(team=None,alliance=None,identity_source='identity_conflict')
        slots={c:sum(self.roster[t]==c for t in owners) for c in ('red','blue')}
        for obj in row['robots']:
            if obj.get('team') is None and obj.get('alliance') in slots:
                color=obj['alliance']
                if slots[color]>=2:obj['alliance']=None
                else:slots[color]+=1
        return row

    def show_frame(self, index, read_sequential=False):
        if not read_sequential:self.capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,image=self.capture.read()
        if not ok:
            self.pause();raise ValueError(f'无法读取第 {index} 帧')
        self.frame=index;self.image=image
        self.current=self.corrected(self.records.get(index))
        self.states=self.memory.update(self.current['seconds'],self.current.get('robots',[]))
        self.seek_active=True;self.slider.set(index);self.seek_active=False
        self.restore_review(force=not read_sequential)
        self.draw()

    def seek(self, index):
        if not self.records:return
        self.pause()
        index=max(0,min(index,len(self.records.offsets)-1))
        # Rebuild only the recent history needed for bounded prediction. No giant video decode.
        self.memory=TeamMemory(self.roster)
        first=max(0,index-round(self.fps*2))
        # Preserve a last-seen card even if the team disappeared earlier than the window.
        seen=set()
        seeds=[]
        for i in range(first-1,-1,-1):
            row=self.corrected(self.records.get(i))
            objects=[o for o in row.get('robots',[]) if o.get('team') in self.roster and o['team'] not in seen]
            if objects:seeds.append((row['seconds'],objects));seen.update(o['team'] for o in objects)
            if len(seen)==4:break
        for seconds,objects in reversed(seeds):self.memory.update(seconds,objects)
        for i in range(first,index):
            row=self.corrected(self.records.get(i));self.memory.update(row['seconds'],row.get('robots',[]))
        self.show_frame(index)

    def slide(self, value):
        # Seek on release, not on every pixel while dragging a long recording.
        pass

    def toggle_play(self):
        if not self.capture:return
        if self.playing:self.pause();self.draw();return
        if self.frame >= len(self.records.offsets)-1:self.seek(0)
        self.playing=True;self.play_started=time.monotonic();self.play_start_frame=self.frame;self.tick()

    def set_playback_rate(self,rate):
        rate=float(rate)
        if rate not in (.25,.5,.75,1.,1.5,2.,4.):raise ValueError('不支持的播放速度')
        self.playback_rate=rate
        if self.playing:
            self.play_start_frame=self.frame;self.play_started=time.monotonic()

    def advance_to(self,index):
        # Decode past late display frames, while consuming every identity record.
        # Dropping a display frame must not drop a binding or last-seen update.
        for skipped in range(self.frame+1,index):
            if not self.capture.grab():raise ValueError(f'无法读取第 {skipped} 帧')
            row=self.corrected(self.records.get(skipped))
            self.memory.update(row['seconds'],row.get('robots',[]))
        self.show_frame(index,read_sequential=True)

    def tick(self):
        self.timer=None
        if not self.playing:return
        last=len(self.records.offsets)-1
        target=min(last,self.play_start_frame+int((time.monotonic()-self.play_started)*self.fps*getattr(self,'playback_rate',1.)))
        try:
            if target>self.frame:self.advance_to(target)
        except Exception as error:
            self.pause();messagebox.showerror('播放停止',str(error),parent=self);return
        if self.frame>=last:self.pause();self.draw();return
        # Presentation follows elapsed time, rather than slowing the match to
        # however many expensive scoreboard paints the computer can finish.
        rate=min(self.fps,getattr(self,'display_fps',30))
        elapsed=time.monotonic()-self.play_started
        deadline=self.play_started+(int(elapsed*rate)+1)/rate
        self.timer=self.after(max(1,round((deadline-time.monotonic())*1000)),self.tick)

    def draw(self):
        if self.image is None:return
        w,h=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        ih,iw=self.image.shape[:2];self.scale=min(w/iw,h/ih)
        dw,dh=max(1,round(iw*self.scale)),max(1,round(ih*self.scale));self.offset=((w-dw)//2,(h-dh)//2)
        image=Image.fromarray(cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)).resize((dw,dh))
        self.photo=ImageTk.PhotoImage(image);self.canvas.delete('all')
        self.canvas.create_image(*self.offset,image=self.photo,anchor='nw')
        for state,label in zip(self.states,self.team_labels):
            color=COLORS[state['alliance']];age='' if state['age'] is None else f' | 距上次关联 {state["age"]:.1f}s'
            label.configure(text=f'{state["team"]} | {state["alliance"].upper()} | {STATUS[state["status"]]}{age}',foreground=color)
            if state['xyxy']:
                coords=self.screen_box(state['xyxy'])
                self.canvas.create_rectangle(*coords,outline=color,width=2,dash=(6,4) if state['status']=='predicted' else ())
                self.canvas.create_text(coords[0]+3,coords[1]+3,text=f'{state["team"]} | {STATUS[state["status"]]}',fill=color,anchor='nw',font=('Microsoft YaHei UI',10,'bold'))
        for obj in self.current.get('robots',[]):
            if obj.get('team') not in self.roster:
                coords=self.screen_box(obj['xyxy']);self.canvas.create_rectangle(*coords,outline='#999999',dash=(2,3))
                self.canvas.create_text(coords[0],coords[1],anchor='sw',text=f'待关联 #{obj.get("track_id")}',fill='white')
        for goal,box in self.session['goal_rois'].items():
            coords=self.screen_box(box);self.canvas.create_rectangle(*coords,outline='#4ade80',width=2)
            self.canvas.create_text(coords[0],coords[1],text=goal,anchor='sw',fill='#4ade80')
        from goal_observer import observe_goal_stacks
        goal_observation=observe_goal_stacks(self.current,self.session['goal_rois'])
        for obj in goal_observation['objects']:
            if obj.get('class_id')==6:continue
            coords=self.screen_box(obj['xyxy']);cls=obj['class_id']
            color={7:'#fa6060',8:'#50b8ff',9:'#f9da52',10:'#e892ef'}.get(cls,'#aaaaaa')
            estimated=obj.get('observation')=='optical_flow_estimate'
            self.canvas.create_rectangle(*coords,outline=color,width=1,dash=(4,3) if estimated else ())
            label={7:'红半',8:'蓝半',9:'黄半',10:'Cup'}.get(cls,'?')+(' ~' if estimated else '')
            self.canvas.create_text(coords[0],coords[1],text=label,fill=color,anchor='sw',font=('Microsoft YaHei UI',8))
        self.canvas.create_text(self.offset[0]+10,self.offset[1]+dh-15,text='miko yolo',anchor='sw',fill='white',font=('Arial',12,'bold'))
        toggles='  '.join(f'{key}:{value.get("color","unknown")}' for key,value in self.current.get('toggles',{}).items())
        drafts=len(goal_observation['objects'])
        self.status.set(f'{self.current["seconds"]:.2f}s | 帧 {self.frame} | Toggle 视觉颜色 {toggles} | 场地候选框 {drafts}（得分未确认） | 点击机器人框绑定至 {self.team.get()}')

    def screen_box(self, box):
        h,w=self.image.shape[:2];ox,oy=self.offset
        return [ox+box[0]*w*self.scale,oy+box[1]*h*self.scale,ox+box[2]*w*self.scale,oy+box[3]*h*self.scale]

    def normalized(self,x,y):
        h,w=self.image.shape[:2];ox,oy=self.offset
        return max(0,min(1,(x-ox)/w/self.scale)),max(0,min(1,(y-oy)/h/self.scale))

    def press(self,event):
        if self.image is None:return
        self.pause();self.start=(event.x,event.y)

    def drag(self,event):
        if self.start and self.mode.get()=='标定 Goal':
            self.canvas.delete('drag');self.canvas.create_rectangle(*self.start,event.x,event.y,outline='#4ade80',tags='drag')

    def release(self,event):
        if not self.start or not self.current:return
        sx,sy=self.start;self.start=None
        if self.mode.get()=='标定 Goal':
            a=self.normalized(sx,sy);b=self.normalized(event.x,event.y)
            if abs(sx-event.x)<8 or abs(sy-event.y)<8:return
            self.session['goal_rois'][self.goal.get()]=[min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])]
            self.save_session();self.draw();return
        x,y=self.normalized(event.x,event.y)
        candidates=[o for o in self.current.get('robots',[]) if o['xyxy'][0]<=x<=o['xyxy'][2] and o['xyxy'][1]<=y<=o['xyxy'][3] and o.get('track_id') is not None]
        if not candidates:return
        obj=min(candidates,key=lambda o:(o['xyxy'][2]-o['xyxy'][0])*(o['xyxy'][3]-o['xyxy'][1]))
        # Replace any same-team binding while two track IDs coexist.
        for binding in list(self.session['bindings']):
            if binding['frame']==self.frame and (binding['track_id']==obj['track_id'] or binding['team']==self.team.get()):
                self.session['bindings'].remove(binding)
        self.session['bindings'].append(dict(frame=self.frame,track_id=obj['track_id'],team=self.team.get()))
        self.save_session();self.seek(self.frame)

    def select_goal(self,event=None):
        selected=self.table.selection()
        if selected:self.goal.set(selected[0]);self.load_goal()

    def load_goal(self):
        for color,var in self.counts.items():
            value=self.draft[self.goal.get()][color];var.set('' if value is None else str(value))
        self.valid.set(False)

    @staticmethod
    def number(var):
        return None if not var.get().strip() else int(var.get())

    def apply_goal(self):
        if not self.records:return
        try:
            if not self.valid.get():raise ValueError('请先核查 Placed / 可见条件，再勾选确认')
            values={color:self.number(var) for color,var in self.counts.items()}
            from scoring import count
            for value in values.values():
                if value is not None:count(value)
            self.draft[self.goal.get()]=values;self.refresh_scores()
        except Exception as error:messagebox.showerror('半段数量无效',str(error),parent=self)

    def payload(self):
        return dict(goals=self.draft,regions={r:OWNER_LABELS[v.get()] for r,v in self.owners.items()},
                    midfield_red=self.number(self.midred),midfield_blue=self.number(self.midblue),phase=self.phase.get(),
                    bonus_red=self.number(self.bonusred),bonus_blue=self.number(self.bonusblue))

    def refresh_scores(self, result=None):
        if result is None:result=match_score(**self.payload())
        for item in self.table.get_children():self.table.delete(item)
        for i in range(9):
            key=f'G{i}';row=result['goals'][key];region='中央' if i==0 else REGIONS[(i-1)//2]
            show=lambda value:'?' if value is None else str(value)
            owner={'unknown':'?','neutral':'中立','red':'红','blue':'蓝'}[row['owner']]
            self.table.insert('', 'end',iid=key,values=(f'{key} / {region}',*(show(row['counts'][c]) for c in ('red','blue','yellow')),owner,f'{show(row["red"])} / {show(row["blue"])}'))
        complete=result['complete'];prefix='' if complete else '已知小计 '
        self.score_title.set(f'{prefix}红 {result["red"]}    蓝 {result["blue"]}'+('' if complete else ' + ?'))
        return result

    def save_review(self):
        if not self.current:return
        try:
            if getattr(self,'pending_candidate_review',False) and not self.valid.get():
                raise ValueError('候选数量尚未复核；请检查各 Goal 的 Placed 和可见条件，勾选 SC2/SC3 确认后再保存。')
            payload=self.payload();result=match_score(**payload)
            snapshot=dict(frame=self.frame,seconds=self.current['seconds'],source='manual_sc2_sc6_review',input=payload,result=result)
            self.session['reviews']=[r for r in self.session['reviews'] if r['frame']!=self.frame]+[snapshot]
            self.session['reviews'].sort(key=lambda r:r['frame']);self.save_session();self.restore_review(force=True)
            self.pending_candidate_review=False
        except Exception as error:messagebox.showerror('无法保存计分',str(error),parent=self)

    def restore_review(self, force=False):
        reviews=[r for r in self.session['reviews'] if r['frame']<=self.frame]
        signature=(reviews[-1]['frame'],reviews[-1]['frame']==self.frame) if reviews else (None,)
        if not force and getattr(self,'review_signature',None)==signature:return
        self.pending_candidate_review=False
        self.review_signature=signature
        if reviews:
            review=reviews[-1];data=review['input']
            self.draft={key:dict(value) for key,value in data['goals'].items()}
            reverse={v:k for k,v in OWNER_LABELS.items()}
            for key,var in self.owners.items():var.set(reverse[data['regions'][key]])
            for var,key in ((self.midred,'midfield_red'),(self.midblue,'midfield_blue'),(self.bonusred,'bonus_red'),(self.bonusblue,'bonus_blue')):
                var.set('' if data[key] is None else str(data[key]))
            self.phase.set(data['phase'])
            suffix='本帧复核' if review['frame']==self.frame else '历史快照，非当前自动分数'
            self.review_time.set(f'{review["seconds"]:.2f}s {suffix}；红/蓝含已填中央分与奖励。')
        else:
            self.draft={f'G{i}':dict(red=None,blue=None,yellow=None) for i in range(9)}
            for var in self.owners.values():var.set('待确认')
            for var in (self.midred,self.midblue,self.bonusred,self.bonusblue):var.set('')
            self.phase.set('end');self.review_time.set('尚无得分复核；Goal / Pin 候选检测不等于有效得分' if 'field_objects' in self.current else '尚无得分复核；当前结果没有 Goal / Pin 检测')
        self.refresh_scores();self.load_goal()

    def close(self):
        self.pause()
        if self.capture:self.capture.release()
        if self.records:self.records.close()
        self.destroy()


if __name__=='__main__':
    root=tk.Tk();root.withdraw();window=Dashboard(root)
    window.protocol('WM_DELETE_WINDOW',lambda:(window.close(),root.destroy()))
    root.mainloop()
