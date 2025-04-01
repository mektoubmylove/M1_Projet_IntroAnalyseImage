import cv2
import numpy as np
import imutils


def merge_close_lines(lines, min_distance=30):
    """ Fusionne les lignes proches en une seule ligne moyenne. """
    if lines is None:
        return []

    lines_y = sorted([(y1 + y2) // 2 for line in lines for x1, y1, x2, y2 in [line[0]]])
    merged_lines = []

    for y in lines_y:
        if not merged_lines or abs(y - merged_lines[-1]) > min_distance:
            merged_lines.append(y)

    return merged_lines


def detect_dominant_angle(image):
    """Détecte l'angle dominant des lignes pour corriger l'inclinaison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)
    if lines is None:
        return 0

    angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for line in lines for x1, y1, x2, y2 in [line[0]]]
    return np.median(angles) if angles else 0


def rotate_image(image, angle):
    """Fait pivoter l'image autour de son centre."""
    (h, w) = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def findContours(image_path):
    """
    Détecte les marches d'escalier avec correction de l'inclinaison.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    image = imutils.resize(image, width=500)

    # Détecter l'angle dominant et corriger l'orientation
    angle = detect_dominant_angle(image)
    print(f"Angle de correction détecté : {angle:.2f}°")
    corrected = rotate_image(image, -angle) if abs(angle) > 75 else image

    # Conversion en niveaux de gris et détection des bords
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Détection des lignes avec Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=75, maxLineGap=10)

    # Fusion des lignes détectées pour trouver les marches
    merged_lines = merge_close_lines(lines, 55)

    # Dessin des marches détectées sur l'image corrigée
    result = corrected.copy()
    for y in merged_lines:
        cv2.line(result, (0, y), (result.shape[1], y), (0, 255, 0), 2)  # Lignes vertes pour les marches

    cv2.putText(result, f"Marches: {len(merged_lines)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    """
    cv2.imshow('Détection avec correction', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return len(merged_lines)

# Exécution du script
#findContours("../data/train/t3i2.jpg")
