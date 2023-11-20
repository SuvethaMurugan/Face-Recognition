import cv2
import numpy as np
import face_recognition

imgbill=face_recognition.load_image_file('bill gates.png')
imgbill=cv2.cvtColor(imgbill,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('elon musk.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgbill)[0];
encodebill=face_recognition.face_encodings(imgbill)[0]
cv2.rectangle(imgbill,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest=face_recognition.face_locations(imgtest)[0];
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodebill],encodetest)
facedis=face_recognition.face_distance([encodebill],encodetest)
print(results,facedis)
cv2.putText(imgtest,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)


cv2.imshow('bill gates',imgbill)
cv2.imshow('bill test',imgtest)
cv2.waitKey(0)
