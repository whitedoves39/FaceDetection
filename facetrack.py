
import face_recognition 
import cv2 

video_capture = cv2.VideoCapture(0)

while True:
    
    ret, frame = video_capture.read()
    resized_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

    # Track current frame of video using the deep learning model "hog" (default one)
    face_locations = face_recognition.face_locations(resized_frame, model="hog")

    for top, right, bottom, left in face_locations:
        top *= 10
        right *= 10
        bottom *= 10
        left *= 10
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)
        print(face_locations)
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
