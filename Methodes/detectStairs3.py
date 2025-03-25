import cv2
import numpy as np
import imutils


def find_optimal_canny_threshold(image):
    """Détermine les seuils optimaux pour Canny en utilisant Otsu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(otsu_threshold * 0.5), int(otsu_threshold * 1.5)


def preprocess_image(image):
    """Prétraitement : conversion en gris, flou gaussien et seuillage Sobel."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=5)
    sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

    _, binary = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def detect_dominant_angle(binary):
    """Détecte l'angle dominant des lignes trouvées avec Hough."""
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=150, maxLineGap=10)
    if lines is None:
        return 0
    angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for line in lines for x1, y1, x2, y2 in [line[0]]]
    return np.median(angles) if angles else 0


def rotate_image(image, angle):
    """Fait pivoter l'image autour de son centre."""
    (h, w) = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def detect_stairs(image):
    """Détecte uniquement les marches en filtrant strictement les lignes horizontales et en fusionnant les lignes proches."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    low_threshold, high_threshold = find_optimal_canny_threshold(gray)
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=5)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)
    if lines is None:
        return 0, 0, 0

    detected_lines_y = []
    validated_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        pente = abs(y2 - y1) / (abs(x2 - x1) + 1e-6)

        # Filtrer uniquement les lignes horizontales (pente ≈ 0)
        if pente < 2.5:  # Plus le seuil est bas, plus la détection est stricte
            mid_y = (y1 + y2) // 2
            detected_lines_y.append(mid_y)
            validated_lines.append((x1, y1, x2, y2))

    #  Fusionner les lignes proches (évite les doublons)
    detected_lines_y.sort()
    merged_lines_y = []
    for y in detected_lines_y:
        if not merged_lines_y or abs(y - merged_lines_y[-1]) > 40:
            merged_lines_y.append(y)

    #  Dessiner uniquement les lignes fusionnées (celles qui restent après fusion)
    output_image = image.copy()

    # Dessiner les lignes fusionnées sur l'image (après fusion des lignes proches)
    for y in merged_lines_y:
        cv2.line(output_image, (0, y), (image.shape[1], y), (0, 255, 0), 2)  # Vert pour les lignes fusionnées

    """
    cv2.imshow("Lignes détectées", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return len(merged_lines_y), np.mean(np.diff(merged_lines_y)) if len(merged_lines_y) > 1 else 0, np.max(np.diff(merged_lines_y)) if len(merged_lines_y) > 1 else 0


def detect_stairs_with_homography(image_path):
    """Détecte les escaliers avec correction d'orientation si nécessaire."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    image = imutils.resize(image, width=400)
    binary = preprocess_image(image)
    angle = detect_dominant_angle(binary)
    print(f"Angle dominant détecté : {angle}°")

    rotated_image = rotate_image(image, -angle) if abs(angle) > 5 else image
    return detect_stairs(rotated_image)


#  Exécution du script
#stairs_count, _, _ = detect_stairs_with_homography("../data/train/Groupe1_Image7.jpg")
#print(f"Nombre de marches détectées : {stairs_count}")
