"""Generic overlays; all team identities and positions come from local records."""
import cv2

COLORS={'red':(40,40,240),'blue':(250,130,30),None:(180,180,180),'yellow':(30,230,250),'unknown':(170,170,170)}


def caption(image,text,position,color,scale=.75):
    x,y=position
    (w,h),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
    x=max(0,min(image.shape[1]-w-8,x));y=max(h+5,min(image.shape[0]-baseline-3,y))
    cv2.rectangle(image,(x-2,y-h-4),(x+w+4,y+baseline+2),(15,15,15),-1)
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)


def draw(image,robots,toggles,rois):
    image=image.copy();h,w=image.shape[:2]
    for obj in robots:
        color=COLORS.get(obj.get('alliance'),COLORS[None])
        x1,y1,x2,y2=[round(n*d) for n,d in zip(obj['xyxy'],[w,h,w,h])]
        cv2.rectangle(image,(x1,y1),(x2,y2),color,3)
        text=f'{obj["team"]} | {obj["alliance"].upper()}' if obj.get('team') and obj.get('alliance') else 'Team unconfirmed'
        caption(image,text,(x1+5,y1+29),color,.8)
    for key,state in toggles.items():
        if key not in rois:continue
        x1,y1,x2,y2=[round(n*d) for n,d in zip(rois[key],[w,h,w,h])];color=COLORS[state['color']]
        cv2.rectangle(image,(x1,y1),(x2,y2),color,3)
        caption(image,f'Toggle {key}: {state["color"].upper()}',(x1,y1-8),color,.65)
    caption(image,'miko yolo',(24,h-24),(245,245,245),.85)
    return image
