import os

import cv2
import imutils
import numpy as np


def detect_stairs2(image_path):
    """
    Détecte et compte les marches d'un escalier en fusionnant les lignes proches.

    :param image_path: Chemin de l'image contenant l'escalier.
    """
    image_path = os.path.abspath(image_path)

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")


    image = imutils.resize(image, width=500)

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit et améliorer la détection des contours
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours avec les filtres Sobel (dérivées en X et Y)
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)  # Dérivée en X
    sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)  # Dérivée en Y

    # Convertir les gradients en valeurs absolues pour obtenir des contours visibles
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Fusionner les gradients Sobel X et Y pour obtenir une meilleure détection des contours
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Appliquer un seuillage d'Otsu pour binariser l'image et accentuer les contours
    _, binary = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=150, maxLineGap=10)

    # Liste pour stocker les coordonnées Y des lignes détectées
    detected_lines_y = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pente = abs(y2 - y1) / (abs(x2 - x1) + 1e-6)  # Calcul de la pente

            # Garder seulement les lignes horizontales
            if 0 < pente < 3:
                detected_lines_y.append((y1 + y2) // 2)

    # Trier les lignes détectées en fonction de leur coordonnée Y
    detected_lines_y.sort()

    # Fusionner les lignes proches pour éviter de compter plusieurs fois la même marche
    merged_lines_y = []
    threshold = 30  # Seuil pour considérer que deux lignes sont la même marche

    for y in detected_lines_y:
        # Si la liste `merged_lines_y` est vide ou si la nouvelle ligne détectée est suffisamment éloignée
        # de la dernière ligne ajoutée (distance > threshold), alors on l'ajoute
        if not merged_lines_y or abs(y - merged_lines_y[-1]) > threshold:
            merged_lines_y.append(y)  # Ajouter la position Y de la nouvelle ligne fusionnée

    # Création d'une image vierge pour dessiner les lignes détectées
    output_image = np.zeros_like(binary)

    # Dessiner les lignes fusionnées représentant les marches
    for y in merged_lines_y:
        cv2.line(output_image, (0, y), (image.shape[1], y), 255, 2)

    # Compter le nombre de marches détectées
    stair_count = len(merged_lines_y)

    # Afficher le nombre de marches sur l'image de sortie
    cv2.putText(output_image, f"Marches: {stair_count}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    #0Affichage des résultats
    
    # cv2.namedWindow("Image Originale", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Image Originale", 640, 640)
    # cv2.imshow("Image Originale", image)

    # cv2.namedWindow("Contours Sobel", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Contours Sobel", 640, 640)
    # cv2.imshow("Contours Sobel", sobel_combined)

    # cv2.namedWindow("Lignes fusionnées", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Lignes fusionnées", 640, 640)
    # cv2.imshow("Lignes fusionnées", output_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(f"Nombre de marches détectées {image_path} : {stair_count}")

    return stair_count

detect_stairs2("C:/Users/Pepit/Documents/GitHub/M1_Projet_IntroAnalyseImage/data/train/grp7img12.jpg")