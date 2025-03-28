import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

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
        return 0  # Pas de correction si aucune ligne trouvée

    angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for line in lines for x1, y1, x2, y2 in [line[0]]]
    return np.median(angles) if angles else 0


def rotate_image(image, angle):
    """Fait pivoter l'image autour de son centre."""
    (h, w) = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def apply_homography(image):
    """Applique une transformation homographique pour redresser la perspective."""
    h, w = image.shape[:2]

    # Points sources (ex: coins approximatifs des marches) - À ajuster selon les images
    #src_pts = np.float32([[50, 200], [450, 200], [30, 400], [470, 400]])
    #src_pts = np.float32([[60, 180], [460, 180], [50, 420], [460, 420]])
    #src_pts = np.float32([[80, 220], [420, 220], [100, 360], [420, 360]])
    src_pts = np.float32([[100, 180], [460, 180], [70, 450], [480, 450]])  #perspective forte
    #src_pts = np.float32([[100, 200], [450, 200], [70, 450], [480, 450]])

    #src_pts = np.float32([[150, 200], [400, 200], [150, 400], [400, 400]])
    #src_pts = np.float32([[20, 150], [480, 150], [20, 500], [480, 500]])
    #src_pts = np.float32([[120, 220], [400, 220], [130, 380], [410, 380]])
    #src_pts = np.float32([[70, 250], [450, 250], [80, 370], [460, 370]])

    # Position cible après redressement
    #dst_pts = np.float32([[100, 200], [450, 200], [100, 450], [450, 450]])  # Image redressée
    dst_pts = np.float32([[50, 200], [450, 200], [50, 450], [450, 450]])
    #dst_pts = np.float32([[70, 180], [450, 180], [70, 460], [450, 460]])
    #dst_pts = np.float32([[90, 200], [430, 200], [90, 400], [430, 400]])
    #dst_pts = np.float32([[50, 220], [470, 220], [50, 460], [470, 460]])
    #dst_pts = np.float32([[60, 200], [460, 200], [60, 420], [460, 420]])
    #dst_pts = np.float32([[70, 210], [450, 210], [70, 430], [450, 430]])
    #dst_pts = np.float32([[100, 180], [460, 180], [100, 450], [460, 450]])

    # Calcul de la matrice d’homographie
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Application de la transformation
    warped_image = cv2.warpPerspective(image, H, (w, h))

    return warped_image


def findContours2(image_path):
    """
    Détecte les marches avec correction de perspective et affiche les résultats.
    """
    # Chargement et redimensionnement
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    image = imutils.resize(image, width=500)

    # Application de l'homographie
    homography_image = apply_homography(image)

    # Correction de l'inclinaison
    angle = detect_dominant_angle(homography_image)
    print(f"Angle détecté : {angle:.2f}°")
    corrected = rotate_image(homography_image, -angle) if abs(angle) > 100 else homography_image

    # Détection des bords et des lignes
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Convertir l'image des contours Canny en image couleur pour pouvoir dessiner en rouge
    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Appliquer HoughLinesP pour détecter les lignes
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=75, maxLineGap=10)

    # Dessiner les lignes en rouge sur l'image couleur des contours Canny
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(color_edges, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Lignes rouges

    # Fusion des lignes détectées pour repérer les marches
    merged_lines = merge_close_lines(lines, 55)

    # Dessin des marches détectées
    result = corrected.copy()
    for y in merged_lines:
        cv2.line(result, (0, y), (result.shape[1], y), (0, 255, 0), 2)

    cv2.putText(result, f"Marches: {len(merged_lines)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage des résultats avec Matplotlib
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Affichage de l'image originale
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image Originale')
    axes[0].axis('off')

    # Affichage de l'image après homographie
    axes[1].imshow(cv2.cvtColor(homography_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Après Homographie')
    axes[1].axis('off')

    # Affichage de l'image avec les contours détectés par Canny et les lignes de Hough en rouge
    axes[2].imshow(cv2.cvtColor(color_edges, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Contours Canny + Lignes Hough (rouge)')
    axes[2].axis('off')

    # Affichage des marches détectées
    axes[3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Marches Détectées')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    return len(merged_lines)


# Exécution du script
findContours2("../data/train/t3i22.png")