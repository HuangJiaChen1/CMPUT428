import cv2
import sys

def select_point(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.namedWindow("Frame")

points = []
cv2.setMouseCallback("Frame", select_point)

while True:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

trackers = cv2.MultiTracker_create()
for point in points:
    trackers.add(cv2.TrackerCSRT_create(), frame, (point[0] - 50, point[1] - 50, 100, 100))
saved_frames = 0
coordinates = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, boxes = trackers.update(frame)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # Save the current frame
        img_name = f"frame_{saved_frames}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")

        # Collect and print coordinates
        current_coords = [(int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)) for box in boxes]
        coordinates.append(current_coords)
        print(f"Coordinates for {img_name}: {current_coords}")

        saved_frames += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("All collected coordinates:", coordinates)
