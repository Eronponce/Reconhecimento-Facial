import face_recognition
import os
import sys
import cv2
import numpy as np
import math

# Helper function to calculate face confidence


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
        # Return confidence as a percentage
    else:
        # Calculate confidence using a non-linear function
        value = (linear_val + ((1.0 - linear_val) * math.pow(
            (linear_val - 0.5) * 2, 0.2
            ))) * 100
        return str(round(value, 2)) + '%'
        # Return confidence as a percentage

# FaceRecognition class for face recognition functionality


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        # Call method to encode known faces

    def encode_faces(self):
        # Iterate over face images in the 'faces' directory and encode them
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            # Append face encoding to known_face_encodings list
            self.known_face_names.append(image)
            # Append face name to known_face_names list
        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)  # Open video capture

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()  # Read frame from video capture

            # Only process every other frame of video to save time
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
                # Resize frame for faster processing

                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                # Convert BGR to RGB color

                # Find all faces and encodings in the current frame
                self.face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame,
                    self.face_locations,
                    model="small"
                    )

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding
                        )
                    name = "Unknown"
                    confidence = '???'

                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings,
                        face_encoding
                        )

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(
                            face_distances[best_match_index]
                            )

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(
                self.face_locations,
                self.face_names
            ):
                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (0, 0, 255),
                    2)  # Draw rectangle around face
                cv2.rectangle(
                    frame,
                    (left, bottom - 35),
                    (right, bottom),
                    (0, 0, 255),
                    cv2.FILLED)  # Draw background for name
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    1
                )  # Add name text

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()  # Release video capture
        cv2.destroyAllWindows()


fr = FaceRecognition()  # Create instance of FaceRecognition class
fr.run_recognition()  # Run the face recognition process
