import pathlib
import cv2

# Load the Haar Cascade for face detection
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

# Open the default camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Define the center box dimensions and position
    box_width = frame_width // 5
    box_height = frame_height // 4
    box_x = (frame_width - box_width) // 2
    box_y = (frame_height - box_height) // 2

    # Draw the center box
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 0), 2)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Calculate the center of the face
        center_x = x + width // 2
        center_y = y + height // 2

        # Draw the center point of the face
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Display the coordinates of the center
        coords_text = f"Center: ({center_x}, {center_y})"
        cv2.putText(frame, coords_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check if the face center is within the center box
        if box_x < center_x < box_x + box_width and box_y < center_y < box_y + box_height:
            cv2.putText(frame, "Face in Box", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Faces", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the camera and close windows
camera.release()
cv2.destroyAllWindows()