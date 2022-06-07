import cv2
import dlib
from scipy.spatial import distance
import keyboard
import selenium
from  selenium import webdriver

driver = webdriver.Firefox(executable_path = '/Users/osho/Desktop/geckodriver')
driver.get("https://dino-chrome.com/")

def eye_ratio(eye):
    ratio1 = distance.euclidean(eye[1], eye[5])
    ratio2 = distance.euclidean(eye[2], eye[4])
    ratio3 = distance.euclidean(eye[0], eye[3])
    ear_ratio = (ratio1+ratio2)/(2.0*ratio3)
    return ear_ratio

#cap = cv2.VideoCapture("test_video.mp4")
cap = cv2.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()

face_landmark = dlib.shape_predictor("trained_model/shape_predictor_68_face_landmarks.dat")

while cap.isOpened():
    _, img = cap.read()
    img=cv2.resize(img,(600,400),interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    for face in faces:

        face_landmarks = face_landmark(gray, face)
        left_eye = []
        right_eye = []

       
        for i in range(36,48):
         if(i>41):
          x = face_landmarks.part(i).x
          y = face_landmarks.part(i).y
          right_eye.append((x,y))
         if(i<43):
            x = face_landmarks.part(i).x
            y = face_landmarks.part(i).y
            left_eye.append((x,y))
            
            

        left_ratio = eye_ratio(left_eye)
        right_ratio = eye_ratio(right_eye)

        total_ratio = (left_ratio+right_ratio)/2
        total_ratio = round(total_ratio,2)
        
        
        if total_ratio<0.25:
         cv2.rectangle(img,(0,0),(600,80),(0,0,0),-1,lineType=cv2.LINE_AA)
         text_format="Blink | EAR: {}".format(total_ratio)
         cv2.putText(img,text_format,(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,lineType=cv2.LINE_AA)
         keyboard.press("space")
        
        

    cv2.imshow("Eye Blink Controlled Game", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
