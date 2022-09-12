import cv2 as cv 
import mediapipe as mp
import time
#cap = cv.VideoCapture(0)
cap = cv.VideoCapture("C0037.MP4")


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   model_complexity=1,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)



while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w ,c = img.shape
            print(id, lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv.circle(img,(cx,cy),5,(255,0,1),cv.FILLED)

    
    
    
    
    cv.imshow("IMAGE",img)
    
   
    if cv.waitKey(1)==ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
    