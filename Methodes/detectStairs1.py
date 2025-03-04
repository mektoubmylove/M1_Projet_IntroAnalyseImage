import cv2
import imutils
import numpy as np

def detect_stairs1(image_path):
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
    """
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
    """
    print(f"Nombre de marches détectées {image_path}: {stair_count}")
    return stair_count

#detect_stairs("../data/train/Groupe1_Image3.jpg")