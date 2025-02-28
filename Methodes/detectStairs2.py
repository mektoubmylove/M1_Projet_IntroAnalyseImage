import cv2
import imutils
import numpy as np


def detect_stairs2(image_path):
    """
    Détecte et compte les marches d'un escalier en fusionnant les lignes proches.

    :param image_path: Chemin de l'image contenant l'escalier.
    """
    # Charger l'image
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    # Convertir en niveaux de gris et appliquer un flou gaussien
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Détection des contours avec Sobel
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Fusionner les gradients
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Seuil Otsu pour binariser
    _, binary = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=150, maxLineGap=10)

    # Liste pour stocker les positions Y des lignes détectées
    detected_lines_y = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = abs(y2 - y1) / (abs(x2 - x1) + 1e-6)  # Éviter la division par zéro

            # Garder seulement les lignes horizontales
            if 0 < slope < 1:
                detected_lines_y.append((y1 + y2) // 2)  # Moyenne de y1 et y2 pour éviter les variations

    # Trier les lignes en fonction de leur position Y
    detected_lines_y.sort()

    # Fusionner les lignes proches
    merged_lines_y = []
    threshold = 30 # Seuil pour considérer deux lignes comme une seule marche

    for y in detected_lines_y:
        if not merged_lines_y or abs(y - merged_lines_y[-1]) > threshold:
            merged_lines_y.append(y)

    # Dessiner les lignes fusionnées sur une image vierge
    output_image = np.zeros_like(binary)

    for y in merged_lines_y:
        cv2.line(output_image, (0, y), (image.shape[1], y), 255, 2)

    # Affichage du nombre de marches
    stair_count = len(merged_lines_y)
    cv2.putText(output_image, f"Marches: {stair_count}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    # Affichage des résultats
    cv2.namedWindow("Image Originale", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Originale", 640, 640)
    cv2.imshow("Image Originale", image)

    cv2.namedWindow("Contours Sobel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contours Sobel", 640, 640)
    cv2.imshow("Contours Sobel", sobel_combined)

    cv2.namedWindow("Lignes fusionnées", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lignes fusionnées", 640, 640)
    cv2.imshow("Lignes fusionnées", output_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Nombre de marches détectées : {stair_count}")

detect_stairs2("../data/train/Groupe1_Image3.jpg")