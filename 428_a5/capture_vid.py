import cv2
import os

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Specify the folder to save the frames
save_folder = 'frames'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Capture 120 frames
for i in range(120):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Construct the filename and save the frame as an image
    frame_filename = f'{save_folder}/frame_{i:03d}.jpg'
    cv2.imwrite(frame_filename, frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()

print(f"Frames have been saved to '{save_folder}'.")
