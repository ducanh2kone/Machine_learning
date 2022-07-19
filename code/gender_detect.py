from unittest import result
from cv2 import cvtColor
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier
import cv2 as cv
#load NN model
mlp= pickle.load(open('code/gender_detect.sav', "rb"))
#load image
frame = cv.imread("data/4people.jpg",cv.IMREAD_COLOR)
gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
# load cascade xml
face_cas = cv.CascadeClassifier()
face_cas.load('code/face_detect.xml')

## detect human faces
faces = face_cas.detectMultiScale(gray)
for (x,y,w,h) in faces:
    cv.rectangle(frame,(x-int(w/5),y-int(h/3)),(x+w+int(w/5),y+h+int(h/10)),(0,255,0),2)
    #crop roi
    roi=gray[y-int(h/3):y+h+int(h/5), x-int(w/5) : x+w+int(w/5)]
    #resize 45x60
    roi= cv.resize(roi,(45,60))
    # normalize
    roi = roi.astype(float)
    cv.normalize(roi,roi,0,1,cv.NORM_MINMAX)
    #reshape
    input = np.reshape(roi,(1,2700))
    #predict
    result = mlp.predict(input)
    if result == 0:
        text= 'female'
    else:
        text = "male"
    cv.putText(frame,text,(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)

cv.imshow("img",frame)
cv.waitKey(0)
cv.destroyAllWindows()

