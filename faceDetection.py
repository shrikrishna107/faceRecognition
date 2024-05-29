import platform
import cv2
from deepface import DeepFace
import tf_keras

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

reference_img = cv2.imread("reference.jpg")
reference_img2 = cv2.imread("reference2.jpg")

while True:
    ret, frame = cap.read()
    
    if ret:
        if counter % 30 == 0:  # Check face every 30 frames
            try:
                result = DeepFace.verify(frame, reference_img.copy())
                result2 = DeepFace.verify(frame, reference_img2.copy())
                if result['verified'] and result2['verified']:
                    face_match = True
                else:
                    face_match = False
            except Exception as e:
                print(f"Error during face verification: {e}")
                face_match = False
        
        counter += 1
        
        if face_match:
            cv2.putText(frame, "MATCH!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imshow("Video", frame)
    
    key = cv2.waitKey(1)
    
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
