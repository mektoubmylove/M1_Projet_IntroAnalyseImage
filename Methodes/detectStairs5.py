import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

def find_optimal_canny_threshold(image):
    """Détermine les seuils optimaux pour Canny en utilisant Otsu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(otsu_threshold * 0.5), int(otsu_threshold * 1.5)


def preprocess_image(image):
    """Prétraitement de l'image : conversion en niveaux de gris, flou et seuillage."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=5)
    sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

    _, binary = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def detect_dominant_angle(binary):
    """Détecte l'angle dominant des lignes trouvées par la transformée de Hough."""
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=150, maxLineGap=10)
    if lines is None:
        return 0
    angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for line in lines for x1, y1, x2, y2 in [line[0]]]
    return np.median(angles) if angles else 0


def rotate_image(image, angle):
    """Fait pivoter l'image selon un angle donné."""
    (h, w) = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def invert_image(image):
    """Inverse les couleurs de l'image."""
    return cv2.bitwise_not(image)


def detect_stairs(image):
    """Détecte les escaliers et affiche les marches sur l'image originale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reverse_gray = invert_image(gray)
    low_threshold, high_threshold = find_optimal_canny_threshold(reverse_gray)
    edges = cv2.Canny(reverse_gray, low_threshold, high_threshold, apertureSize=5)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)
    detected_lines_y = sorted([(y1 + y2) // 2 for line in lines for x1, y1, x2, y2 in [line[0]] if
                               0 < abs(y2 - y1) / (abs(x2 - x1) + 1e-6) < 3]) if lines is not None else []

    # Fusionner les lignes proches
    merged_lines_y = []
    for y in detected_lines_y:
        if not merged_lines_y or abs(y - merged_lines_y[-1]) > 55:
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
    """Détecte les escaliers en appliquant une correction d'orientation si nécessaire."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    image = imutils.resize(image, width=400)
    binary = preprocess_image(image)
    angle = detect_dominant_angle(binary)
    print(f"Angle dominant détecté : {angle}°")

    rotated_image = rotate_image(image, -angle) if abs(angle) > 10 else image

    # Détection finale avec affichage
    return detect_stairs(rotated_image)


#  Exécution du script
#detect_stairs_with_homography("../data/train/Groupe5_image16.jpeg")
