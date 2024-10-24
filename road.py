import cv2 as cv 
import numpy as np
def process_frame(frame):
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(gray,(5,5),0)
    edges=cv.Canny(blur, 50,150)
    height,width=edges.shape
    mask=np.zeros_like(edges)
    polygon=np.array([[(0,height*3/5),(width,height*3/5),(width,height),(0,height),]],np.int32)
    cv.fillPoly(mask,polygon,255)
    masked_edges=cv.bitwise_and(edges,mask)
    lines=cv.HoughLinesP(masked_edges,1,np.pi/180,50,maxLineGap=50)
    line_image=np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)
    combo_image=cv.addWeighted(frame,0.8,line_image,1,0)
    return combo_image


cap=cv.VideoCapture('C:/Users/sana/Downloads/line_rode.mp4')

while(cap.isOpened()):
    ret,frame=cap.read()
    if ret:
        processed_frame=process_frame(frame)
        cv.imshow('result',processed_frame)
        if cv.waitKey(5) & 0xFF == 27:
          break
    else:
        break
cap.release()
cv.destroyAllWindows()
    
        
