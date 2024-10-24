import cv2 as cv
import numpy as np
import math
def process_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height * 3 / 5), (width, height * 3 / 5), (width, height), (0, height)]], np.int32)
    cv.fillPoly(mask, polygon, 255)
    masked_edges = cv.bitwise_and(edges, mask)
    lines = cv.HoughLinesP(masked_edges, 1, np.pi / 180, 50, maxLineGap=50)
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    combo_image = cv.addWeighted(frame, 0.8, line_image, 1, 0)
    return combo_image
def process_frame1(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height * 3 / 5), (width, height * 3 / 5), (width, height), (0, height)]], np.int32)
    cv.fillPoly(mask, polygon, 255)
    masked_edges = cv.bitwise_and(edges, mask)
    lines = cv.HoughLinesP(masked_edges, 1, np.pi / 180, 50, maxLineGap=50)
    line_info = []  
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))  
            line_info.append((length, angle))  
    return line_info 
cap = cv.VideoCapture('C:/Users/sana/Downloads/line_rode.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        lines_info = process_frame1(frame)
        for length, angle in lines_info:
            print(f"Length: {length}, Angle: {angle}")
        processed_frame = process_frame(frame)
        cv.imshow('result', processed_frame)
        _, buffer = cv.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        if cv.waitKey(5) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
