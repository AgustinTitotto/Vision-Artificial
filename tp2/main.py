import cv2
import math
from functions import convert_to_color, on_trackbar, get_trackbar_value, denoise, get_contour, filter_contours_by_area
from joblib import load

cap = cv2.VideoCapture(0) # Use laptop camara
cv2.namedWindow('img1') # Create Window
cv2.createTrackbar('threshold', 'img1', 80, 255, on_trackbar) # Create trackbar for threshold
cv2.createTrackbar('denoise', 'img1', 7, 50, on_trackbar) # Create trackbar for denoise
cv2.createTrackbar('minArea', 'img1', 450, 10000, on_trackbar) # Create trackbars for area
cv2.createTrackbar('maxArea', 'img1', 95000, 99999, on_trackbar)
#cv2.createTrackbar('tolerance', 'img1', 20, 100, on_trackbar) # Create trackbar for distance, decimal
saved_contours = []
biggest_contour = None


sorter = load('filename.joblib')

while True:     
    ret, frame = cap.read()
    flip_frame = cv2.flip(frame, 1) # Flip image so its correct
    grey_frame = convert_to_color(frame=flip_frame, color=cv2.COLOR_BGR2GRAY) # Convert frame to gray
    
    
    threshold_value = get_trackbar_value('threshold', 'img1') # Grab trackbar value for threshold
    kernel_radius = get_trackbar_value('denoise', 'img1') # Grab trackbar value for denoise
    min_area = get_trackbar_value('minArea', 'img1')
    max_area = get_trackbar_value('maxArea', 'img1')
    #distance = get_trackbar_value('tolerance', 'img1')/100 # Grab trackbar value for distance, then convert to decimal
    
    ret1, thresh1 = cv2.threshold(grey_frame, threshold_value, 255, cv2.THRESH_BINARY) # Apply threshold with trackbar value
    denoise_frame = denoise(thresh1, cv2.MORPH_ELLIPSE, kernel_radius) # Apply denoise
    cv2.imshow('grey_frame', denoise_frame)
    contours, hierarchy = cv2.findContours(denoise_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # Draw contours
    filtered_contours = filter_contours_by_area(contours, min_area, max_area)

    predicted_labels = []

    for contour in filtered_contours:   
       
        hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        for i in range(0, 7):
            hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i])) # Mapeo para agrandar la escala.
    
        predicted_label = sorter.predict([hu_moments])[0]
        # min_score = min(square_score, circle_score, star_score)

        x, y, w, h = cv2.boundingRect(contour)

        # if min_score > distance: # If the score is bigger than distance selected, it doesnt match any shape
        #     cv2.drawContours(flip_frame, [contour], -1, (0, 0, 255), 5) # If it doesnt match any shape, paint contour in red
        # else: 
        if predicted_label == 1:
            cv2.drawContours(flip_frame, [contour], -1, (0, 255, 255), 5) # Paint square yellow
            cv2.putText(flip_frame, 'Square', (x - 20, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
        elif predicted_label == 2:
            cv2.drawContours(flip_frame, [contour], -1, (255, 0, 0), 5) # Paint circle blue
            cv2.putText(flip_frame, 'Triangle', (x - 20, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        elif predicted_label == 3:
            cv2.drawContours(flip_frame, [contour], -1, (0, 255, 0), 5) # Paint star green
            cv2.putText(flip_frame, 'Star', (x - 20, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow('img1', flip_frame) # Show image

    if cv2.waitKey(1) == ord('z'): # Waits () amount of time, if the key 'z' is pressed, it stops the loop
        break

cap.release()
cv2.destroyAllWindows()



