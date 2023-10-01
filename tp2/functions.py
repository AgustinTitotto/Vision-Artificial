import cv2

def convert_to_color(frame, color): # Convert frame's color to parameter color 
    return cv2.cvtColor(frame, color)

def on_trackbar(val): # Cuando hay un cambio, que no haga nada. Ahora estamos pidiendo siempre el valor. Podriamos pasarle una funcion que cambie la imagen. 
    pass


def get_trackbar_value(trackbar_name, window_name): # Get tackbar value to change image threshold
    return cv2.getTrackbarPos(trackbar_name, window_name) + 1 # No puede ser 0, porque crashea

def denoise(frame, method, radius): # Method to eliminate noise
    kernel = cv2.getStructuringElement(method, (radius, radius)) 
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel) # Erosión - dilatación, elimina ruido true (puntos blancos)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) # Dilatación - erosión, elimina ruido false (puntos negros)
    return closing

def get_contours(frame, mode, method):
    contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours

def get_biggest_contour(contours):
    max_cnt = contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt) > cv2.contourArea(max_cnt):
            max_cnt = cnt
    return max_cnt

def compare_contours(contour_to_compare, saved_contours, max_diff):
    for contour in saved_contours:
        if cv2.matchShapes(contour_to_compare, contour, cv2.CONTOURS_MATCH_I2, 0) < max_diff:
            return True
    return False

def draw_contours(frame, contours, color, thickness):
    # -1 for all contours
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame

def get_contour(path): # Grab image contour 
    image = cv2.imread(path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret1, contour = cv2.threshold(grey_image, 120, 255, cv2.THRESH_BINARY)
    return get_contours(contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0] # (1 because 0 is the whole image contour)

def filter_contours_by_area(contours, min_area, max_area):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)
    return filtered_contours