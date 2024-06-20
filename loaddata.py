import cv2
import numpy as np
import os
def distance(p,q):
    return np.sqrt(np.sum((p-q)**2))
def knn(x,y,xt,k=5):
    dlist=[]
    m=x.shape[0]
    for i in range(m):
        d=distance(x[i],xt)
        dlist.append((d,y[i]))
    dlist.sort()
    dlist=dlist[:k]
    dlist=np.array(dlist)
    labels=dlist[:,1]
    labels,cnts=np.unique(labels,return_counts=True)
    idx=cnts.argmax()
    return int(labels[idx])
dataset_path=""
faceData=[]
labels=[]
classId=0
Idmap={}
for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        dataItem=np.load(dataset_path+f)
        faceData.append(dataItem)
        Idmap[classId]=f[:-4]
        m=dataItem.shape[0]
        target=classId*np.ones((m,))
        classId+=1
        labels.append(target)

faceData=np.concatenate(faceData,axis=0)
labels=np.concatenate(labels,axis=0).reshape((-1,1))
cam=cv2.VideoCapture(0)
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    success,img=cam.read()
    faces=model.detectMultiScale(img,1.2,1)
    offset=1
    for f in faces:
        x,y,w,h=f
        
        cropped_face=img[y-offset:y+h+offset,x-offset:x+w+offset]
        cropped_face=cv2.resize(cropped_face,(100,100))
        cv2.imshow("croppedimage",cropped_face)
        classpredicted=knn(faceData,labels,cropped_face.flatten())
        print(classpredicted)
        namepredicted=Idmap[classpredicted]
        cv2.putText(img,namepredicted,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) 
    cv2.imshow("prediction window",img)
    key=cv2.waitKey(1)
    if key== ord('q'):
        break

cam.release()
cv2.destroyAllWindows()    