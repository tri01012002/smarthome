import time
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import imageio
import sys
import serial

#import matplotlib.pyplot as plt

# step 1 load ảnh từ kho ảnh nhận dạng
#DataSerial = serial.Serial('COM3', 115200)

path="pic"
images = []
classNames = []
myList =os.listdir(path)
print(myList)
for cl in myList:
    print(cl)
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # splitext sẽ tách path ra thành 2 phần, phần trước đuôi mở rộng và phần mở rộng
print(len(images))
print(classNames)

#step encoding
def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR được chuyển đổi sang RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = Mahoa(images)
print("ma hoa thanh cong")
#print(len(encodeListKnow))

def thamdu(name):
    with open("thamdu.csv","r+") as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtstring}")


#khởi dộng webcam
cap = cv2.VideoCapture(2)

while True:
    ret, frame= cap.read()
    cv2.imshow('tri', frame)
    framS = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    # xác định vị trí khuôn mặt trên cam và encode hình ảnh trên cam
    facecurFrame = face_recognition.face_locations(framS) # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame): # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại theo cặp
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis) #đẩy về index của faceDis nhỏ nhất


        if faceDis[matchIndex] <0.50 :
            name = classNames[matchIndex].upper()
            thamdu(name)

            DataSerial = serial.Serial('COM3', 115200)
            time.sleep(1)

            def ledON():
                DataSerial.write('ON\r'.encode())
                ledON()
                time.sleep(3)

        else:
            name = "Unknow"
            def ledOFF():
                DataSerial.write('OFF\r'.encode())
                ledOFF()



        #print tên lên frame
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
        cv2.rectangle(frame,(x1,y1), (x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('tri', frame)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        break
cap.release()  # giải phóng camera
cv2.destroyAllWindows()  # thoát tất cả các cửa sổ

