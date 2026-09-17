"""
enroll.py
Run this once per employee to register their face.

Usage:
    python enroll.py "Jane Doe"

It opens your webcam, captures several good frames, averages the face
encodings for robustness, and saves the employee to the database.
"""

import sys
import time
import numpy as np
import cv2

import database
from recognizer import compute_encoding_from_frame

SAMPLES_NEEDED = 15


def main():
    if len(sys.argv) < 2:
        print('Usage: python enroll.py "Employee Name"')
        sys.exit(1)

    name = " ".join(sys.argv[1:]).strip()
    database.init_db()

    existing = [e["name"].lower() for e in database.get_all_employees()]
    if name.lower() in existing:
        print(f"'{name}' is already enrolled. Delete them first if you want to re-enroll.")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check it's connected and not used by another app.")
        sys.exit(1)

    print(f"Enrolling '{name}'. Look at the camera. Move your head slightly between captures.")
    print("Press 'q' to cancel at any time.\n")

    encodings = []
    last_capture = 0

    while len(encodings) < SAMPLES_NEEDED:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        cv2.putText(
            display, f"Captured: {len(encodings)}/{SAMPLES_NEEDED}",
            (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.imshow("Enrollment - press q to cancel", display)

        # Capture a sample roughly every 0.4s so the user can move slightly
        if time.time() - last_capture > 0.4:
            encoding = compute_encoding_from_frame(frame)
            if encoding is not None:
                encodings.append(encoding)
                last_capture = time.time()
                print(f"  captured sample {len(encodings)}/{SAMPLES_NEEDED}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    averaged_encoding = np.mean(encodings, axis=0)
    employee_id = database.add_employee(name, averaged_encoding)
    print(f"\nDone. '{name}' enrolled with employee_id={employee_id}.")
    print("Restart or rerun the dashboard so it picks up the new face.")


if __name__ == "__main__":
    main()
