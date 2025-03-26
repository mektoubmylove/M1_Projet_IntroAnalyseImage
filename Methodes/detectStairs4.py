import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

def find_optimal_canny_threshold(image):
    """Trouve les seuils optimaux pour Canny en utilisant Otsu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    return int(otsu_threshold * 0.5), int(otsu_threshold * 1.5)


def preprocess_image(image):
    """Convertit en niveaux de gris, applique un flou et détecte les contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours avec Sobel
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=5))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=5))
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Binarisation avec Otsu
    _, binary = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return binary


def detect_dominant_angle(binary):
    """Détecte l'angle dominant des lignes dans l'image."""
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=150, maxLineGap=10)
    angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for line in lines for x1, y1, x2, y2 in
              [line[0]]] if lines is not None else []
    return np.median(angles) if angles else 0


def rotate_image(image, angle):
    """Effectue une rotation de l'image pour corriger son orientation."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def detect_stairs(image):
    """Détecte les escaliers dans l'image et affiche les marches sur l'image originale."""
    binary = preprocess_image(image)

    # Détection des contours avec Canny
    low_threshold, high_threshold = find_optimal_canny_threshold(binary)
    edges = cv2.Canny(binary, low_threshold, high_threshold, apertureSize=5)

    # Détection des lignes avec Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    detected_lines_y = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = abs(y2 - y1) / (abs(x2 - x1) + 1e-6)
            if 0 < slope < 3:
                detected_lines_y.append((y1 + y2) // 2)

    detected_lines_y.sort()

    # Fusionner les lignes proches
    merged_lines_y = []
    threshold = 50
    for y in detected_lines_y:
        if not merged_lines_y or abs(y - merged_lines_y[-1]) > threshold:
            merged_lines_y.append(y)

    stair_count = len(merged_lines_y)

    # Dessiner les marches détectées sur l'image originale
    output_image = image.copy()
    for y in merged_lines_y:
        cv2.line(output_image, (0, y), (image.shape[1], y), (0, 0, 255), 2)  # Rouge pour les marches détectées

    # Ajouter le nombre de marches détectées sur l'image
    cv2.putText(output_image, f"Marches: {stair_count}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    """
    # Affichage des résultats avec subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Afficher les images dans les subplots
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image Originale')
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Contours Canny')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Marches Détectées')
    axes[2].axis('off')

    # Afficher la fenêtre
    plt.tight_layout()
    plt.show()
    """
    return stair_count



def detect_stairs_with_homography(image_path):
    """Corrige l'orientation de l'image si nécessaire avant la détection des escaliers."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    image = imutils.resize(image, width=500)
    binary = preprocess_image(image)

    # Détection et correction de l'orientation
    angle = detect_dominant_angle(binary)
    print(f"Angle dominant détecté : {angle}°")
    if abs(angle) > 50:
        image = rotate_image(image, -angle)

    # Détection finale des escaliers avec affichage
    return detect_stairs(image)


#  Exécution du script
#detect_stairs_with_homography("../data/train/Groupe5_image03.jpg")
