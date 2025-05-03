from facial_emotion_recognition import EmotionRecognition
import cv2

# Initialize EmotionRecognition
er = EmotionRecognition(device='cpu')

# Start video capture
cam = cv2.VideoCapture(0)



try:
    while True:
        ret, frame = cam.read()
      
        
        # Process frame for emotion recognition
        frame = er.recognise_emotion(frame, return_type='BGR')
        
        # Show frame
        cv2.imshow("Emotion Recognition Frame", frame)
        
        # Exit on 'Esc' key
        if cv2.waitKey(1)  == 27:
            break
finally:
    cam.release()
    cv2.destroyAllWindows()
