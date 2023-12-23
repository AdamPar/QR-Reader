import cv2
import numpy as np


def find_qr_code(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter rectangles based on area to find the largest ones
    rectangles = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has four vertices, which indicates a potential QR code
        if len(approx) == 4:
            # Check if the aspect ratio is approximately 1, which is common for QR codes
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                rectangles.append(approx)

    # Sort rectangles based on area in descending order
    rectangles = sorted(rectangles, key=cv2.contourArea, reverse=True)

    # Draw a rectangle around the whole QR code based on the three largest squares with red color
    if len(rectangles) >= 3:
        x, y, w, h = cv2.boundingRect(np.vstack((rectangles[0], rectangles[1], rectangles[2])))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw each of the three rectangles with green color
        for rect in rectangles[:3]:
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Print a message to the console
        print("QR code detected")

    return image


# Open a connection to the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to grab frame")
        break

    # Apply QR code detection
    result_frame = find_qr_code(frame)

    # Display the result
    cv2.imshow('QR Code Reader', result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
