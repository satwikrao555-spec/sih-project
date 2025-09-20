import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import threading
import queue


def load_face_data():
    face_data = [
        ("faces/satwik.jpg", "M. Satwik Rao"),
        ("faces/newton.jpg", "Newton Mishra"),
        ("faces/aniketh.jpg", "Aniketh Kumar Yadav"),
        ("faces/pragg.jpg", "Pragyanshu Priyadarshi Padhy"),
        ("faces/k.vashishta.jpg", "K. Vishishta")
    ]

    encodings = []
    names = []

    for file_path, name in face_data:
        try:
            image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(image)[0]
            encodings.append(encoding)
            names.append(name)
        except (IndexError, FileNotFoundError):
            print(f"Warning: Could not load {file_path}")

    return encodings, names


def setup_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def process_frame(frame, student_encodings, student_names, scale_factor=0.2):
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []
    scale_up = int(1 / scale_factor)

    for i, face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(student_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(student_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                name = student_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
            else:
                name = "Unknown"
                confidence = 0

            if i < len(face_locations):
                top, right, bottom, left = face_locations[i]
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (left * scale_up, top * scale_up, right * scale_up, bottom * scale_up)
                })

    return results


def draw_results(frame, results):
    for result in results:
        left, top, right, bottom = result['location']
        name = result['name']
        confidence = result['confidence']

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

        display_text = f"{name} ({confidence:.2f})" if name != "Unknown" else name
        cv2.putText(frame, display_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

    return frame


def write_attendance(csv_writer, f, name, recognized_today):
    if name != "Unknown" and name not in recognized_today:
        current_time = datetime.now().strftime("%H:%M:%S")
        csv_writer.writerow([name, current_time])
        f.flush()
        recognized_today.add(name)
        print(f"Attendance: {name} at {current_time}")


def main():
    student_encodings, student_names = load_face_data()

    if not student_encodings:
        print("No face data loaded. Exiting.")
        return

    cap_vid = setup_camera()
    recognized_today = set()

    current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    f = open(f"{current_date}.csv", "w+", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Name", "Time"])

    frame_count = 0
    process_interval = 2
    last_results = []

    print("Face recognition started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap_vid.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % process_interval == 0:
                last_results = process_frame(frame, student_encodings, student_names)

                for result in last_results:
                    write_attendance(csv_writer, f, result['name'], recognized_today)

            frame = draw_results(frame, last_results)

            cv2.putText(frame, f"Recognized: {len(recognized_today)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Face Recognition System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap_vid.release()
        cv2.destroyAllWindows()
        f.close()
        print(f"Session ended. {len(recognized_today)} people recognized.")


if __name__ == "__main__":
    main()