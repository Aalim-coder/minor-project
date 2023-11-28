import cv2
from PIL import Image
import numpy as np

def CreateModel(image):
    img = image.resize((450,250))
    img_arr = np.array(img)
    grey = cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    dilated = cv2.dilate(blur,np.ones((3,3)))
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,kernel)
    return [closing,img_arr]

model = CreateModel(Image.open('cars.png'))
car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(model[0], 1.1, 1)
cnt = 0
for(x,y,w,h) in cars:
  cv2.rectangle(model[1],(x,y),(x+w,y+h),(255,0,0),2)
  cnt+=1
print(cnt, "cars found")
img = Image.fromarray(model[1])
cv2.imshow("cars",img)

