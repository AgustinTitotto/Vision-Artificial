import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Display the webcam feed
    cv2.imshow('Image', frame)

    # Wait for a key press to capture the image
    key = cv2.waitKey(1)

    if key == ord('c'):
        # Save the captured image as input_image.jpg
        cv2.imwrite('input_image.jpg', frame)
        print("Captured image saved as input_image.jpg")
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


# Function to create a circular seed with the specified label
def create_seed(event, x, y, flags, param):
    global seeds, seed_label

    if event == cv2.EVENT_LBUTTONDOWN:
        # Create a circular seed with a diameter of 7 pixels
        cv2.circle(seed_map, (x, y), 7, (seed_label,), -1)
        # Annotate the seed label on the image
        cv2.putText(image, str(seed_label), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Load the image to be segmented
image = cv2.imread('input_image.jpg')

# Initialize the seed map with zeros
seed_map = np.zeros(image.shape[:2], dtype=np.int32)

# Create a window for the image and set the mouse callback
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', create_seed)

# Initialize seed label to 1
seed_label = 1

while True:
    # Display the image with seeds
    cv2.imshow('Image', image)

    # Create a colormap for seed visualization
    seed_colormap = cv2.applyColorMap((seed_map * 10).astype(np.uint8), cv2.COLORMAP_JET)

    # Display the seed map
    cv2.imshow('Seed Map', seed_colormap)

    if(key >= ord('1')) and (key <= ord('9')):
        seed_label = key - ord('0')

    # Wait for user input
    key = cv2.waitKey(1)

    # If spacebar is pressed, perform watershed segmentation
    if key == ord(' '):
        # Apply watershed segmentation
        markers = cv2.watershed(image, seed_map)

        # Overlay the segmentation result on the original image
        image[markers == -1] = [0, 0, 255]

        # Show the segmented image
        cv2.imshow('Segmented Image', image)

    # If 'q' is pressed, exit the loop
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
