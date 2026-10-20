"""Local video review timeline: filmstrip, score track, scrubbing and zoom."""
import math
import queue
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
import cv2
from PIL import Image,ImageTk


def timecode(frame,fps):
    seconds=int(frame/fps);sub=min(round(fps)-1,int(frame-seconds*fps))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}:{sub:02d}'


class ReplayTimeline(tk.Canvas):
    LEFT=78
    def __init__(self,master,on_seek,on_pause,**kwargs):
        super().__init__(master,height=155,bg='#101722',highlightthickness=0,**kwargs)
        self.on_seek=on_seek;self.on_pause=on_pause;self.total=1;self.fps=30;self.frame=0
        self.zoom=1.;self.first=0.;self.history=[];self.start=None;self.source=None
        self.thumbnails={};self.images=[];self.dragging=False;self.pending=None;self.scrub_timer=None
        self.results=queue.Queue();self.pool=ThreadPoolExecutor(max_workers=1);self.closed=False
        self.bind('<Configure>',lambda _:self.paint())
        self.bind('<ButtonPress-1>',self.press);self.bind('<B1-Motion>',self.motion);self.bind('<ButtonRelease-1>',self.release)
        self.bind('<Control-MouseWheel>',self.wheel_zoom);self.bind('<MouseWheel>',self.wheel_pan)
        self.bind('<ButtonPress-2>',self.pan_start);self.bind('<B2-Motion>',self.pan_motion)
        self.poll_timer=self.after(80,self.poll)

    def configure_clip(self,path,total,fps,history,start):
        if path!=self.source:
            self.source=path;self.total=total;self.fps=fps;self.first=0.;self.zoom=1.;self.thumbnails={}
            self.pool.submit(self.make_thumbnails,path,total)
        self.history=history;self.start=start

    def make_thumbnails(self,path,total):
        cap=cv2.VideoCapture(path)
        try:
            for i in range(16):
                if self.closed or path!=self.source:break
                frame=round(i*(total-1)/15);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
                if ok:self.results.put((path,frame,cv2.cvtColor(cv2.resize(image,(100,56)),cv2.COLOR_BGR2RGB)))
        finally:cap.release()

    def poll(self):
        changed=False
        while True:
            try:path,frame,image=self.results.get_nowait()
            except queue.Empty:break
            if path==self.source:self.thumbnails[frame]=ImageTk.PhotoImage(Image.fromarray(image));changed=True
        if changed:self.paint()
        if not self.closed:self.poll_timer=self.after(80,self.poll)

    def span(self):return max(1,(self.total-1)/self.zoom)
    def width(self):return max(1,self.winfo_width()-self.LEFT-12)
    def x(self,frame):return self.LEFT+(frame-self.first)/self.span()*self.width()
    def at(self,x):return max(0,min(self.total-1,round(self.first+(x-self.LEFT)/self.width()*self.span())))

    def position(self,frame):
        self.frame=frame
        if not self.dragging and not self.first<=frame<=self.first+self.span():self.first=max(0,min(self.total-1-self.span(),frame-self.span()*.2))
        self.paint()

    def paint(self):
        self.delete('all');self.images=[];right=self.winfo_width()-12
        if right<=self.LEFT:return
        self.create_rectangle(self.LEFT,28,right,91,fill='#202c3d',outline='#426080')
        self.create_rectangle(self.LEFT,96,right,132,fill='#172131',outline='#29384e')
        for y,label,color in ((57,'V1 视频','#87bff4'),(112,'S1 估分','#8e9db3')):
            self.create_text(9,y,text=label,anchor='w',fill=color,font=('Microsoft YaHei UI',9))
        seconds=self.span()/self.fps;raw=seconds/max(2,self.width()/95)
        steps=(.1,.2,.5,1,2,5,10,15,30,60,120,300,600,1800)
        step=next((v for v in steps if v>=raw),3600)
        for tick in range(math.ceil(self.first/self.fps/step),math.floor((self.first+self.span())/self.fps/step)+1):
            frame=tick*step*self.fps;x=self.x(frame)
            self.create_line(x,19,x,27,fill='#6a7d96');self.create_text(x+3,10,text=timecode(round(frame),self.fps),fill='#96a7bc',anchor='w',font=('Consolas',9))
        if self.thumbnails:
            for x in range(self.LEFT,round(right),100):
                frame=self.at(x+50);nearest=min(self.thumbnails,key=lambda f:abs(f-frame));photo=self.thumbnails[nearest]
                self.create_image(x,32,image=photo,anchor='nw');self.images.append(photo)
        self.create_rectangle(0,28,self.LEFT-1,133,fill='#101722',outline='')
        self.create_text(9,57,text='V1 视频',anchor='w',fill='#87bff4');self.create_text(9,112,text='S1 估分',anchor='w',fill='#8e9db3')
        visible=[p for p in self.history if self.first/self.fps<=p['seconds']<=(self.first+self.span())/self.fps]
        maximum=max(10,max((max(p['red'],p['blue']) for p in visible),default=10))
        for key,color in (('red','#ff6578'),('blue','#48b9ff')):
            points=[(self.x(p['seconds']*self.fps),128-p[key]/maximum*27) for p in visible]
            if len(points)>1:self.create_line(*[v for point in points for v in point],fill=color,width=2)
        if self.start is not None and self.first<=self.start<=self.first+self.span():
            x=self.x(self.start);self.create_line(x,25,x,133,fill='#5ad4ad',dash=(3,3));self.create_text(x+3,80,text='开赛',anchor='w',fill='#5ad4ad')
        if self.first<=self.frame<=self.first+self.span():
            x=self.x(self.frame);self.create_line(x,20,x,135,fill='#ff5c72',width=2);self.create_polygon(x-6,22,x+6,22,x,30,fill='#ff5c72')
        self.create_text(self.LEFT,146,text='拖动回看 · Ctrl+滚轮缩放 · 滚轮平移',anchor='w',fill='#7d8ea6',font=('Microsoft YaHei UI',9))
        self.create_text(right,146,text=timecode(self.frame,self.fps),anchor='e',fill='#dbe8f8',font=('Consolas',10))

    def press(self,event):
        if event.x<self.LEFT:return
        self.on_pause();self.dragging=True;self.motion(event)

    def motion(self,event):
        if not self.dragging:return
        self.pending=self.at(event.x);self.frame=self.pending;self.paint()
        if self.scrub_timer is None:self.scrub_timer=self.after(70,self.flush)

    def flush(self):
        self.scrub_timer=None
        if self.pending is not None:
            target=self.pending;self.pending=None;self.on_seek(target)

    def release(self,event):
        if not self.dragging:return
        if self.scrub_timer is not None:self.after_cancel(self.scrub_timer);self.scrub_timer=None
        self.pending=self.at(event.x);self.flush();self.dragging=False

    def set_zoom(self,factor,anchor=None):
        anchor=self.frame if anchor is None else anchor
        fraction=(anchor-self.first)/self.span();self.zoom=max(1,min(64,self.zoom*factor))
        self.first=max(0,min(max(0,self.total-1-self.span()),anchor-fraction*self.span()));self.paint()

    def wheel_zoom(self,event):self.set_zoom(1.5 if event.delta>0 else 1/1.5,self.at(event.x));return 'break'
    def wheel_pan(self,event):
        self.first=max(0,min(max(0,self.total-1-self.span()),self.first-self.span()*.12*(1 if event.delta>0 else -1)));self.paint();return 'break'
    def pan_start(self,event):self.pan_anchor=(event.x,self.first)
    def pan_motion(self,event):
        self.first=max(0,min(max(0,self.total-1-self.span()),self.pan_anchor[1]-(event.x-self.pan_anchor[0])/self.width()*self.span()));self.paint()

    def close(self):
        self.closed=True;self.after_cancel(self.poll_timer)
        if self.scrub_timer is not None:self.after_cancel(self.scrub_timer)
        self.pool.shutdown(wait=False,cancel_futures=True)
