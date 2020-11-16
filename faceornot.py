import face_recognition #importing facial recognition library
import cv2 #importing webcam or opencv library
import numpy as np # array library is imported
#Made By Liam Chae

""" device index or the name of a video file is taken into the Videopcature object as an argument, so this code recognizes 
0, the single main webcam on the computer."""

video_capture =  cv2.VideoCapture(0)

# Create arrays of known face encodings and their names

# Initialization of 2 variables.
face_locations = [] # Used  to find the face and used for displaying the lebelled box
Compute_frame = True # So that we won't have to eqaul it to true in if statement. And the frame logically cannot be false.

#while the condition is true
while True:
    # get camera frame then obtain the return values. Then equals it to the written values.
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for quicker computiation.
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert a image from BGR color to RGB color for change of purpose, from OpenCV to face_recgontion
    rgb_small_frame = small_frame[:, :, ::-1]

    # process frames
    if Compute_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label for the face
        cv2.rectangle(frame, (left, bottom - 55), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Congrats, you have a face", (left + 5, bottom - 5), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Click q on keyboard to exit out of program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()