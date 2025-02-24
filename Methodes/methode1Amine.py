import cv2
import imutils
import numpy as np


def count_stairs(image_path):
    """
    Détecte et compte le nombre de marches d'un escalier sur une image.

    :param image_path: Chemin de l'image à analyser.
    """
    # Charger l'image
    img = cv2.imread(image_path)
    img = imutils.resize(img, width=800)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    # Appliquer un flou gaussien pour améliorer la détection des contours
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Détection des contours avec Canny
    edges = cv2.Canny(blurred, 80, 240, apertureSize=3)

    # Convertir en couleur pour affichage des lignes détectées
    out_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    control = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=5)

    # Liste pour stocker les positions des lignes détectées
    y_keeper_for_lines = []

    # Affichage des lignes détectées sur l'image de contrôle
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(control, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

        # Prendre la première ligne détectée comme référence
        x1, y1, x2, y2 = lines[0][0]
        cv2.line(out_img, (0, y1), (img.shape[1], y1), (0, 0, 255), 3, cv2.LINE_AA)
        y_keeper_for_lines.append(y1)

        stair_counter = 1  # Initialiser le compteur de marches

        # Vérifier les autres lignes détectées
        for line in lines[1:]:
            x1, y1, x2, y2 = line[0]
            okey = True  # Variable pour éviter les doublons

            for y in y_keeper_for_lines:
                if abs(y - y1) < 15:  # Vérifier si la ligne est proche d'une ligne existante
                    okey = False
                    break

            if okey:
                cv2.line(out_img, (0, y1), (img.shape[1], y1), (0, 0, 255), 3, cv2.LINE_AA)
                y_keeper_for_lines.append(y1)
                stair_counter += 1

        # Ajouter le texte du nombre de marches détectées
        cv2.putText(out_img, f"Nombre de marches : {stair_counter}",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Affichage des images avec des fenêtres redimensionnées
    cv2.namedWindow("Image Originale", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Originale", 640, 640)
    cv2.imshow("Image Originale", img)

    cv2.namedWindow("Contours détectés", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contours détectés", 640, 640)
    cv2.imshow("Contours détectés", control)

    cv2.namedWindow("Marches détectées", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Marches détectées", 640, 640)
    cv2.imshow("Marches détectées", out_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
#count_stairs("../data/train/Groupe1_Image3.jpg")

def detect_stairs_rectangles(image_path):
    """
    Détecte et compte les marches d'un escalier en identifiant les formes rectangulaires.

    :param image_path: Chemin de l'image contenant l'escalier.
    """
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path)
    image=imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour lisser l'image et réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des bords avec Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Détection des contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Liste pour stocker les rectangles détectés
    detected_rectangles = []

    # Parcourir tous les contours détectés
    for contour in contours:
        # Approximer le contour par un polygone
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Vérifier si le polygone a 4 sommets => Potentiellement un rectangle
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # Filtrer les petits rectangles (bruit)
            if w > 50 and h > 20:  # Ajuster selon la taille des marches
                detected_rectangles.append((x, y, w, h))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Nombre total de rectangles détectés
    num_stairs = len(detected_rectangles)
    cv2.putText(image, f"Nombre de marches : {num_stairs}",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher les résultats
    cv2.namedWindow("Contours détectés", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contours détectés", 640, 640)
    cv2.imshow("Contours détectés", edges)

    cv2.namedWindow("Marches détectées", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Marches détectées", 640, 640)
    cv2.imshow("Marches détectées", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
#detect_stairs_rectangles("../data/train/Groupe1_Image3.jpg")

def detect_stairs(image_path):
    """
    Détecte et compte les marches d'un escalier en utilisant des contours et des lignes de Hough.

    :param image_path: Chemin de l'image contenant l'escalier.
    """
    # Charger l'image
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    # Si l'image est trop grande, on la recadre
    if image.shape[1] > 1000 and image.shape[0] > 1000:
        image = image[100: image.shape[0] - 100, 100: image.shape[1] - 100]

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Détection des contours avec Laplacien
    laplace = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
    laplace = cv2.convertScaleAbs(laplace)

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
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=180, maxLineGap=10)

    # Dessiner les lignes détectées
    output_image = np.zeros_like(binary)
    detected_points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = abs(y2 - y1) / (abs(x2 - x1) + 1e-6)  # Éviter la division par zéro
            if 0 < slope < 1:  # On garde seulement les lignes presque horizontales
                cv2.line(output_image, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)
                detected_points.append((x1, y1))

    # Filtrer les points pour éviter les doublons
    filtered_points = []
    x_thresh, y_thresh = 5, 8  # Seuils pour fusionner les points proches

    for i, point in enumerate(detected_points):
        if i > 0:
            if abs(point[0] - detected_points[i - 1][0]) < x_thresh or abs(point[1] - detected_points[i - 1][1]) < y_thresh:
                continue
        filtered_points.append(point)

    # Affichage du nombre de marches
    stair_count = len(filtered_points)
    cv2.putText(output_image, f"Nombre de marches: {stair_count}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    # Affichage des résultats
    cv2.namedWindow("Image Originale", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Originale", 640, 640)
    cv2.imshow("Image Originale", image)

    cv2.namedWindow("Contours Sobel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contours Sobel", 640, 640)
    cv2.imshow("Contours Sobel", sobel_combined)

    cv2.namedWindow("Lignes détectées", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lignes détectées", 640, 640)
    cv2.imshow("Lignes détectées", output_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Nombre de marches détectées : {stair_count}")

#detect_stairs("../data/train/Groupe1_Image3.jpg")

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