import cv2
import numpy as np

cap = cv2.VideoCapture(0) # Use laptop camara
cv2.namedWindow('img1') # Create Window

while True:
    ret, frame = cap.read()
    flip_frame = cv2.flip(frame, 1) # Flip image so its correct
    cv2.imshow('img1', flip_frame)  # Show image

    if cv2.waitKey(1) == ord(' '): # waits for spacebar to be pressed
        ret1, img = cap.read()
        flip_img = cv2.flip(img, 1) # Flip image so its correct
        cv2.namedWindow('img2') # Create Window
        cv2.imshow('img2', flip_img) # Show image

        # si usamos el metodo de GC_INIT_WITH_RECT no es necesario camara por eso hacemos una matriz de 0
        #[:2] se utiliza para tomar solo los dos primeros elementos de la tupla, que representan las dimensiones espaciales de la imagen (ancho y alto).
        mask = np.zeros(flip_img.shape[:2], np.uint8) # Creamos la mascara del tamaño de la imagen

        # These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # usamos roi para agarrar el rect
        box = cv2.selectROI("img2", flip_img, fromCenter=False, showCrosshair=False)

        # hacemos el grabcut (imagen, mascara, rectangulo, bgdModel, fgdModel, iteraciones, metodo)
        cv2.grabCut(flip_img, mask, box, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

        # Esta operación suele utilizarse para generar una máscara binaria en la que los valores 0 y 2 representan el fondo y los valores distintos de 0 y 2 representan el objeto de interés.
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        flip_img = flip_img * mask2[:, :, np.newaxis]

        cv2.namedWindow('Grabcut')  # Create Window
        cv2.imshow('Grabcut', flip_img)  # Show image



    if cv2.waitKey(1) == ord('q'): # Waits () amount of time, if the key 'q' is pressed, it stops the loop
        break