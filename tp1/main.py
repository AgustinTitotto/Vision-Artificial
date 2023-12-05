import cv2
from functions import convert_to_color, on_trackbar, get_trackbar_value, denoise, get_contour, filter_contours_by_area

cap = cv2.VideoCapture(0) # Use laptop camara
cv2.namedWindow('img1') # Create Window
cv2.createTrackbar('threshold', 'img1', 184, 255, on_trackbar) # Create trackbar for threshold
cv2.createTrackbar('denoise', 'img1', 7, 50, on_trackbar) # Create trackbar for denoise
cv2.createTrackbar('minArea', 'img1', 450, 10000, on_trackbar) # Create trackbars for area
cv2.createTrackbar('maxArea', 'img1', 95000, 99999, on_trackbar)
cv2.createTrackbar('tolerance', 'img1', 20, 100, on_trackbar) # Create trackbar for distance, decimal
saved_contours = []
biggest_contour = None

square_contour = get_contour('./static/square.png')
circle_contour = get_contour('./static/circle.png') # Get figures contours
star_contour = get_contour('./static/star.png')


while True:     
    ret, frame = cap.read()
    flip_frame = cv2.flip(frame, 1) # Flip image so its correct
    grey_frame = convert_to_color(frame=flip_frame, color=cv2.COLOR_BGR2GRAY) # Convert frame to gray

    threshold_value = get_trackbar_value('threshold', 'img1') # Grab trackbar value for threshold
    kernel_radius = get_trackbar_value('denoise', 'img1') # Grab trackbar value for denoise
    min_area = get_trackbar_value('minArea', 'img1')
    max_area = get_trackbar_value('maxArea', 'img1')
    distance = get_trackbar_value('tolerance', 'img1')/100 # Grab trackbar value for distance, then convert to decimal
    
    ret1, thresh1 = cv2.threshold(grey_frame, threshold_value, 255, cv2.THRESH_BINARY) # Apply threshold with trackbar value
    denoise_frame = denoise(thresh1, cv2.MORPH_ELLIPSE, kernel_radius) # Apply denoise

    contours, hierarchy = cv2.findContours(denoise_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # Draw contours
    filtered_contours = filter_contours_by_area(contours, min_area, max_area)

    for contour in filtered_contours:
        square_score = cv2.matchShapes(contour, square_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        circle_score = cv2.matchShapes(contour, circle_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        star_score = cv2.matchShapes(contour, star_contour, cv2.CONTOURS_MATCH_I1, 0.0)

        min_score = min(square_score, circle_score, star_score)

        x, y, w, h = cv2.boundingRect(contour)

        if min_score > distance: # If the score is bigger than distance selected, it doesnt match any shape
            cv2.drawContours(flip_frame, [contour], -1, (0, 0, 255), 5) # If it doesnt match any shape, paint contour in red
        else: 
            if min_score == square_score:
                cv2.drawContours(flip_frame, [contour], -1, (0, 255, 255), 5) # Paint square yellow
                cv2.putText(flip_frame, 'Square', (x - 20, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
            if min_score == circle_score:
                cv2.drawContours(flip_frame, [contour], -1, (255, 0, 0), 5) # Paint circle blue
                cv2.putText(flip_frame, 'Circle', (x - 20, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            if min_score == star_score:
                cv2.drawContours(flip_frame, [contour], -1, (0, 255, 0), 5) # Paint star green
                cv2.putText(flip_frame, 'Star', (x - 20, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow('img1', flip_frame) # Show image

    if cv2.waitKey(1) == ord('z'): # Waits () amount of time, if the key 'z' is pressed, it stops the loop
        break

cap.release()
cv2.destroyAllWindows()



