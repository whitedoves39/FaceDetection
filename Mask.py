import face_recognition #importing facial recognition library
import cv2 #importing webcam or opencv library
import numpy as np # array library is imported as np to shorten it and avoid name issues.
#Made By Liam Chae
""" device index or the name of a video file is taken into the Videopcature object as an argument, so this code recognizes 
0, the single main webcam on the computer."""

video_capture =  cv2.VideoCapture(0)

#Teach the program how to recognize faces by giving it the image.
# Load examples and teach it how to recognize it.
Masked1_image= face_recognition.load_image_file("./img/masked/5558UB_67.jpg")
Masked1_encoding = face_recognition.face_encodings(Masked1_image)[0]

Masked2_image= face_recognition.load_image_file("./img/masked/1227809769.jpg.0.jpg")
Masked2_encoding = face_recognition.face_encodings(Masked2_image)[0]

Masked3_image= face_recognition.load_image_file("./img/masked/Depositphotos_5398452_s-2019-e1580369327239.jpg")
Masked3_encoding = face_recognition.face_encodings(Masked3_image)[0]

Masked4_image= face_recognition.load_image_file("./img/masked/KN95-Disposable-Face-Mask-White-1.jpg")
Masked4_encoding = face_recognition.face_encodings(Masked4_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [ 
    Masked1_encoding,
    Masked2_encoding,
    Masked3_encoding,
    Masked4_encoding
    # Can add new variables here
]
known_face_names = [
    "You have a mask.",
    "You have a mask.",
    "You have a mask.",
    "You have a mask."
    
    #Can add more known face names

]
# Create arrays of known face encodings and their names

# Initialization of 2 variables.
face_coordinates = [] #Used  to find the face and used for displaying the labelled  box
face_encodings = [] # The xy index of the face 
face_names = [] # give name to the face images.
Compute_frame = True # Delcared as true for faster processing of the program

#while the condition is true
while True:
    # get camera frame then obtain the return values. Then equals it to the written values.
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for quicker computiation.
    Resize_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    #Convert an image from BGR color to RGB color for change of purpose, from OpenCV to face_recgontion
    New_frame = Resize_frame[:, :, ::-1]

    # process frames
    if Compute_frame:
        # Find all the faces and face encodings in the current frame of video
        face_coordinates = face_recognition.face_locations(New_frame)
        face_encodings = face_recognition.face_encodings(New_frame, face_coordinates)

        face_names=[]
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.6)
            name = "Tha'is nowt masked"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            print(face_distances)
    Compute_frame = not Compute_frame

    # Display the results
    for (t, r, b, l), name in zip(face_coordinates, face_names):
        # Scale back up face coordinates since the frame we detected in was scaled to 1/4 size
        t *= 4
        r *= 4
        b *= 4
        l *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (l, t), (r, b), (50, 205, 50), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (l, b - 25), (r, b), (255, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, name, (l + 5, b - 5), font, 1.0, (0, 0, 0), 1)


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()