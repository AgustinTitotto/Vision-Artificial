import cv2
import numpy as np


def grab_image():
    global key
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Display the webcam feed
        cv2.imshow('Image', frame)

        # Wait for a key press to capture the image
        key = cv2.waitKey(1)

        if key == ord('c'):
            # Save the captured image as input_image.jpg
            cv2.imwrite('input_image.jpg', frame)
            print("Captured image saved as input_image.jpg")
            break

    cap.release()


def initialize_image():
    global image, seed_map, seed_label
    # Load the image to be segmented
    image = cv2.imread('input_image.jpg')

    # Initialize the seed map with zeros
    seed_map = np.zeros(image.shape[:2], dtype=np.int32)

    # Create a window for the image and set the mouse callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', create_seed)

    # Initialize seed label to 1
    seed_label = 1


# Function to create a circular seed with the specified label
def create_seed(event, x, y, flags, param):
    global seeds, seed_label

    if event == cv2.EVENT_LBUTTONDOWN:
        # Create a circular seed with a diameter of 7 pixels
        cv2.circle(seed_map, (x, y), 7, (seed_label,), -1)
        # Annotate the seed label on the image
        cv2.putText(image, str(seed_label), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def watershed():
    while True:
        global seed_map, seed_label, key, seed_colormap
        # Display the image with seeds
        cv2.imshow('Image', image)

        # Create a colormap for seed visualization
        seed_colormap = cv2.applyColorMap((seed_map * 100).astype(np.uint8), cv2.COLORMAP_JET)

        # Display the seed map
        cv2.imshow('Seed Map', seed_colormap)

        if (key >= ord('1')) and (key <= ord('9')):
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

        if key == ord('r'):
            cv2.destroyAllWindows()
            grab_image()
            initialize_image()


        # If 'q' is pressed, exit the loop
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break


grab_image()
initialize_image()
watershed()