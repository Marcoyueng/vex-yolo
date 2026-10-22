"""Large-video replay sidebar, goal contributions and a full-width score chart."""
from PIL import ImageDraw
from scoreboard_view import font,draw_player_chart

COLORS={'red':'#ff6e7b','blue':'#55baff','yellow':'#ffdf60','unknown':'#95a1b2'}
PHASES={'prestart':'开赛前 · 不计分','autonomous':'自动阶段','intermission':'自动结束 / 等待手动','driver':'手动阶段'}


def draw_score_panel(image,row,history):
    d=ImageDraw.Draw(image);view=row['score_estimate'];score=view['score'];x=1296
    phase=view['match_timing']['period']
    def text(a,b,s,size=16,color='#dce7f7',bold=False):
        d.text((a,b),str(s),font=font(size,bold),fill=color)
    def card(a,b,w,h,fill='#111d2e'):
        d.rounded_rectangle((a,b,a+w,b+h),radius=9,fill=fill,outline='#26384d')
    text(x,65,PHASES[phase],20,'#80dfc9',True)
    for i,(color,label) in enumerate((('red','红方 RED'),('blue','蓝方 BLUE'))):
        a=x+i*228;card(a,101,220,142,'#271c28' if color=='red' else '#112c3c')
        text(a+12,112,label,20,COLORS[color],True)
        text(a+12,145,('' if phase=='prestart' else '≈ ')+str(score[color]),38,COLORS[color],True)
        text(a+12,199,f'Goal {score["goal_"+color]}  ·  奖励 '+('?' if score[color+'_bonus'] is None else str(score[color+'_bonus'])),15)
        text(a+12,222,'未计未知项 / AI 估分' if phase!='prestart' else '等待比赛开始',12,'#a4b5cc')
    text(x,257,'TOGGLE / 场内侧颜色 → 黄色归属',17)
    for i,site in enumerate(('left','bottom','right','top')):
        a=x+i*114;c=row['toggles'][site]['color'] if view['scene_valid'] else 'unknown'
        label={'left':'左','bottom':'下','right':'右','top':'上'}[site]+' / '+{'red':'红','blue':'蓝','yellow':'中立','unknown':'?'}[c]
        card(a,287,106,35);text(a+10,294,label,16,COLORS[c])
    text(x,337,'GOAL / 有效半段 · 得分（红 / 蓝）',17)
    for i in range(9):
        goal=f'G{i}';v=view['goals'][goal];s=score['goals'][goal]
        a=x+(i%3)*152;b=368+(i//3)*113
        card(a,b,144,105)
        text(a+9,b+7,goal+(' 中央' if i==0 else ''),19,bold=True)
        for k,c in enumerate(('red','blue','yellow')):
            value=v['counts'][c]
            label='?' if value is None else ('≥' if c in v.get('minimum_counts',{}) else '')+str(value)
            text(a+9+44*k,b+34,label,19,COLORS[c],True)
        contributions=[]
        for c in ('red','blue'):
            # Show known contributions even if another color remains unknown.
            n=v['counts'][c];y=v['counts']['yellow'];owner=s['owner']
            subtotal=(n or 0)*5+((y or 0)*10 if owner==c else 0)
            if phase=='prestart':subtotal=0
            lower_bound=c in v.get('minimum_counts',{}) or (owner==c and 'yellow' in v.get('minimum_counts',{}))
            contributions.append(('≥' if lower_bound and phase!='prestart' else '')+str(subtotal)+('?' if s[c] is None and phase!='prestart' else ''))
        text(a+9,b+60,' / '.join(contributions),17,'#68ddc2',True)
        removed=any(r.get('event_kind')=='observed_removal' for r in v['reviewed_counts'].values())
        state=v.get('reviewed_scoring_state')
        status=('未开赛' if phase=='prestart' else '罩住不计分 · SC3' if state=='nested_not_scored'
            else '地面 Pin · 不计分' if state=='ground_pins_not_scored'
            else '无有效半段 · 已复核' if state=='confirmed_no_scoring_halves'
            else '可见半段 · 已复核' if state=='visible_scoring_halves'
            else '已确认下限 · 已保留' if v.get('minimum_counts') else '脱落后 · 按现存数量' if removed else '人工纠正 · 已保留' if v['reviewed_counts']
            else '已保留 / 待复核' if v.get('memory_held') else '数量待确认' if v['partial'] else '稳定记录')
        text(a+9,b+85,status,11,'#a3b3c8')
    text(x,717,'TEAMS / 两红 + 两蓝',18)
    for i,s in enumerate(row['team_states']):
        a=x+(i%2)*228;b=749+(i//2)*65;card(a,b,220,56)
        text(a+10,b+6,s['team'],20,COLORS[s['alliance']],True)
        status='已关联' if s['status']=='observed' else '待接回'
        if s.get('identity_source')=='unique_remaining_robot_reconnect' and s['status']=='observed':status='四台排除接回'
        text(a+10,b+34,status,12,'#a3b3c8')
    notes=['红/蓝半段各 5 分；黄色半段按区域归属 10 分。',
           '套入 Goal / Cup 不透明半部或未相连的 Pin 不计分。',
           '中央归属、终场停留及 Toggle 有效性待复核。',
           '人工纠正已入账；模型尚未因此完成补训。']
    if view.get('reviewed_incidents'):
        incident=view['reviewed_incidents'][-1]
        notes[-1]=f'用户标注：{incident["team"]} 拉倒 {incident["goal"]}，判罚待复核。'
    for i,line in enumerate(notes):text(x,890+i*23,line,13,'#91a5bf')
    draw_player_chart(image,(12,792,1268,965),history,row['seconds'])
    text(16,974,'正常速度 · 原声 · AI 估分含人工纠正，非官方裁判分数',14,'#9eb3cf')
