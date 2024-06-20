import cv2
import numpy as np
cam=cv2.VideoCapture(0)
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
filename=input("ENTER THE PERSON'S NAME:")
filepath=''
faceData=[]
skip=0
while True:
    success,img=cam.read()
    faces=model.detectMultiScale(img,1.2,1)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    offset=1
    if len(faces)>0:
        f=faces[-1]
        x,y,w,h=f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cropped_face=img[y-offset:y+h+offset,x-offset:x+w+offset]
        cropped_face=cv2.resize(cropped_face,(100,100))
        skip+=1
        if skip%10==0:
            faceData.append(cropped_face)
            print("Photo Saved So Far" + str(len(faceData))) 
            if len(faceData)>=20:
                break
        cv2.imshow("croppedimage",cropped_face)
    cv2.imshow("image",img)
    key=cv2.waitKey(1)
    if key== ord('q'):
        break
faceData=np.array(faceData)
m=faceData.shape[0]
faceData=faceData.reshape((m,-1))
file_path=filepath+filename+".npy"
np.save(file_path,faceData)
cam.release()
cv2.destroyAllWindows()