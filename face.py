#face detection
import cv2
import numpy as np
import datetime
#these two are models built with lots of images
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_classifier=cv2.CascadeClassifier("haarcascade_eye.xml")
   
#It will read the first frame/image of the video
video=cv2.VideoCapture(0)
while True:
    #capture the first frame
    check,frame=video.read()#black&white
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#making into color
    
    #detect the faces & eyes from the video using detectMultiScale function
    faces=face_classifier.detectMultiScale(gray,1.3,5)#returns 4 values
    eyes=eye_classifier.detectMultiScale(gray,1.3,5) #returns 4 values

    print(faces)#matrix form displaying
    
    #drawing rectangle boundries for the detected face
    for(x,y,w,h) in faces:
        #x and y axis of rectangle, RGB values,
        cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
        cv2.imshow('Face detection', frame)
        #cv2.imwrite('facenow.jpg',frame)
        picname = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        cv2.imwrite(picname+".jpg",frame)
    #Note: if we rerun it will update the same file
        
    #drawing rectangle boundries for the detected eyes
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (127,0,255), 2)
        cv2.imshow('Face detection', frame)

    #waitKey(1)- for every 1 millisecond new frame will be captured
    Key=cv2.waitKey(1)
    if Key==ord('q'):
        #release the camera
        video.release()
        #destroy all windows
        cv2.destroyAllWindows()
        break

