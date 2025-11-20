import cv2
import numpy as np

video = cv2.VideoCapture(0)
while True:
    r,frame = video.read()
    if r==True:
        cv2.imshow("video",frame)
        if cv2.waitKey(1) & 0xff==ord("p"):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()