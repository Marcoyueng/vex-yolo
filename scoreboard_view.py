"""Shared dark local scoreboard rendering for the desktop UI and video export."""
from functools import lru_cache
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

BG='#080e19';PANEL='#111b2b';EDGE='#23314a';TEXT='#edf3ff';MUTED='#889ab5'
RED='#ff6578';BLUE='#48b9ff';YELLOW='#ffda62';GREEN='#57dbb0'
SIZE=(1600,960)


def paste_video(image,video):
    frame=video.copy();frame.thumbnail((1024,576))
    image.paste(frame,(24+(1024-frame.width)//2,107+(576-frame.height)//2))


def paint_time(image,seconds):
    d=ImageDraw.Draw(image);d.rectangle((24,69,850,97),fill=BG)
    d.text((25,70),f'比赛画面   /   视频时间 {int(seconds//60):02d}:{seconds%60:04.1f}',font=font(16),fill=MUTED)


@lru_cache(maxsize=32)
def font(size,bold=False):
    for path in (Path('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/arial.ttf')):
        if path.exists():return ImageFont.truetype(str(path),size)
    return ImageFont.load_default()


def compose(video,observation,roster,states,toggles,history=(),review=None):
    image=Image.new('RGB',SIZE,BG);d=ImageDraw.Draw(image)
    def text(x,y,value,size=16,fill=TEXT,bold=False):d.text((x,y),str(value),font=font(size,bold),fill=fill)
    def card(box,fill=PANEL):d.rounded_rectangle(box,radius=13,fill=fill,outline=EDGE,width=1)
    def dot(x,y,color):d.ellipse((x,y,x+8,y+8),fill=color)
    text(25,20,'MIKO YOLO',31,bold=True);text(260,31,'OVERRIDE  /  '+(observation.get('match') or '比赛分析'),16,MUTED)
    d.rounded_rectangle((1250,25,1574,64),radius=18,fill='#15372f')
    text(1271,33,'● 本地回放  ·  AI 估分待复核',15,GREEN)
    seconds=observation['seconds'];text(25,70,f'比赛画面   /   视频时间 {int(seconds//60):02d}:{seconds%60:04.1f}',16,MUTED)
    card((22,103,1050,690))
    if video is not None:paste_video(image,video)
    score=review['result'] if review is not None else observation['score']
    label='人工复核快照' if review else 'AI 稳定记录估分' if observation.get('persistence',{}).get('enabled') else 'AI 可见候选估分'
    unknown=not score['complete'];prefix='' if review else '≈ '
    for x,color,key,title in ((1070,RED,'red','红方 RED'),(1330,BLUE,'blue','蓝方 BLUE')):
        card((x,103,x+246,239),'#241725' if key=='red' else '#102638')
        dot(x+16,121,color);text(x+32,114,title,17,color,True)
        text(x+15,146,prefix+str(score[key])+(' + ?' if unknown else ''),37,color,True)
        text(x+16,205,label,13,MUTED)
    text(1070,253,'TOGGLE  /  可见颜色，未确认有效占领',14,MUTED)
    color_map={'red':RED,'blue':BLUE,'yellow':YELLOW,'unknown':MUTED}
    for i,(site,title) in enumerate((('left','左'),('bottom','下'),('right','右'),('top','上'))):
        x=1070+i*128;card((x,282,x+118,327))
        color=toggles.get(site,{}).get('color','unknown');dot(x+12,301,color_map.get(color,MUTED))
        text(x+29,291,title+' / '+{'red':'红','blue':'蓝','yellow':'中立','unknown':'?'}.get(color,'?'),14)
    text(1070,344,'GOAL  /  目标堆叠 · 下行为红 / 蓝估分',14,MUTED)
    for index in range(9):
        x=1070+(index%3)*172;y=373+(index//3)*130
        card((x,y,x+160,y+119))
        goal=f'G{index}';row=observation['goals'][goal];points=score['goals'][goal]
        values=points['counts'] if review else row['counts']
        if not review and row.get('partial') and not any(values.values()):
            values=dict.fromkeys(('red','blue','yellow'),None)
        text(x+12,y+9,goal+(' · 中央' if index==0 else ''),18,bold=True)
        status=row['status'];valid=status in ('estimate','stable','remembered')
        if review:
            valid=points['red'] is not None and points['blue'] is not None
            status_text='人工复核' if valid else '部分复核 · 待确认'
        else:
            status_text={'stable':'稳定记录','remembered':'遮挡保持 · 已记录','pending':'等待连续检测'}.get(status,'候选半段' if valid else {'occluded':'遮挡 · 待确认','unobserved':'未识别 · 待确认','ambiguous':'归属不明'}.get(status,'待确认'))
            if row.get('partial') and any(values.values()) and status not in ('remembered','pending'):status_text='已见半段 · 其余待确认'
        text(x+12,y+38,status_text,12,MUTED)
        for j,(key,color) in enumerate((('red',RED),('blue',BLUE),('yellow',YELLOW))):
            dot(x+12+j*48,y+63,color);text(x+25+j*48,y+54,'?' if values[key] is None else values[key],16,color,True)
        show=lambda value:'?' if value is None else str(value)+('+?' if not review and row.get('partial') else '')
        text(x+12,y+88,('≈ ' if not review else '')+show(points['red'])+' / '+show(points['blue']),17,GREEN if valid else MUTED,True)
    text(1070,779,'TEAMS  /  身份持续显示',14,MUTED)
    state_map={s['team']:s for s in states}
    for i,(team,alliance) in enumerate(roster.items()):
        x=1070+(i%2)*260;y=809+(i//2)*62
        card((x,y,x+246,y+53));color=RED if alliance=='red' else BLUE
        text(x+12,y+7,team,18,color,True)
        state=state_map.get(team,{}).get('status','waiting')
        text(x+105,y+13,{'observed':'已关联','predicted':'短时预测','lost':'失联待确认','waiting':'等待'}.get(state,'?'),12,MUTED)
    card((22,710,1050,936))
    text(40,725,'分数走势',18,bold=True);text(173,730,'红 / 蓝可见候选估分  ·  随视频时间更新',13,MUTED)
    left,top,right,bottom=72,777,1020,903
    for frac in (0,.5,1):
        y=bottom-frac*(bottom-top);d.line((left,y,right,y),fill=EDGE)
    data=[p for p in history if p['seconds']<=seconds]
    maximum=max(20,max((max(p['red'],p['blue']) for p in data),default=0)*1.15)
    end=max(10,seconds)
    for frac in (0,.5,1):text(38,bottom-frac*(bottom-top)-8,round(maximum*frac),11,MUTED)
    for key,color in (('red',RED),('blue',BLUE)):
        points=[(left+p['seconds']/end*(right-left),bottom-p[key]/maximum*(bottom-top)) for p in data]
        if len(points)>1:d.line(points,fill=color,width=3)
    text(left,909,'00:00',11,MUTED);text(right-55,909,f'{int(seconds//60):02d}:{int(seconds%60):02d}',11,MUTED)
    text(1070,940,'仅估分 · 落位/可见性/中央与奖励待确认',12,MUTED)
    return image
