import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Impossible d'accéder à la webcam.")

cv2.namedWindow("Trackbars")

def nothing(x):
    pass

cv2.createTrackbar("H min", "Trackbars", 15, 179, nothing)
cv2.createTrackbar("S min", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("V min", "Trackbars", 212, 255, nothing)
cv2.createTrackbar("H max", "Trackbars", 50, 179, nothing)
cv2.createTrackbar("S max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V max", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de lecture de la webcam.")
        break

    # Réduction taille pour optimiser
    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H min", "Trackbars")
    s_min = cv2.getTrackbarPos("S min", "Trackbars")
    v_min = cv2.getTrackbarPos("V min", "Trackbars")
    h_max = cv2.getTrackbarPos("H max", "Trackbars")
    s_max = cv2.getTrackbarPos("S max", "Trackbars")
    v_max = cv2.getTrackbarPos("V max", "Trackbars")

    lower_yellow = np.array([h_min, s_min, v_min])
    upper_yellow = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #nettoyage du masque
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()
    found = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if radius > 10:  #moins strict pour webcam
                cv2.circle(result, center, radius, (0, 255, 0), 2)
                cv2.circle(result, center, 3, (0, 0, 255), -1)
                found = True

    cv2.imshow("Webcam", frame)
    cv2.imshow("Masque Jaune", mask)
    cv2.imshow("Masque Nettoyé", mask_clean)
    if found:
        cv2.imshow("Détection", result)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
