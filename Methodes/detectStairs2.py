import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

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

    image = imutils.resize(image, width=300)

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit et améliorer la détection des contours
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours avec Sobel
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Fusionner les gradients Sobel X et Y
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Seuil Otsu pour binariser l'image
    _, binary = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=150, maxLineGap=10)

    # Liste pour stocker les coordonnées Y des lignes détectées
    detected_lines_y = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = abs(y2 - y1) / (abs(x2 - x1) + 1e-6)
            if 0 < slope < 3:  # Conserver les lignes horizontales
                detected_lines_y.append((y1 + y2) // 2)

    # Trier les lignes détectées par leur coordonnée Y
    detected_lines_y.sort()

    # Fusionner les lignes proches
    merged_lines_y = []
    threshold = 40  # Seuil pour considérer que deux lignes sont la même marche

    for y in detected_lines_y:
        if not merged_lines_y or abs(y - merged_lines_y[-1]) > threshold:
            merged_lines_y.append(y)

    # Création d'une image  pour dessiner les lignes fusionnées
    output_image = np.zeros_like(binary)

    # Dessiner les lignes fusionnées représentant les marches
    for y in merged_lines_y:
        cv2.line(output_image, (0, y), (image.shape[1], y), 255, 2)

    # Compter le nombre de marches détectées
    stair_count = len(merged_lines_y)

    # Afficher le nombre de marches sur l'image de sortie
    cv2.putText(output_image, f"Marches: {stair_count}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sobel_combined_rgb = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2RGB)
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    """
    # Affichage avec subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(sobel_combined_rgb, cmap="gray")
    axes[1].set_title("Contours Sobel")
    axes[1].axis("off")

    axes[2].imshow(output_image_rgb, cmap="gray")
    axes[2].set_title(f"Lignes fusionnées ({stair_count} marches)")
    axes[2].axis("off")

    plt.show()
    """
    print(f"Nombre de marches détectées {image_path}: {stair_count}")
    return stair_count

#detect_stairs2("../data/train/Groupe5_image05.jpg")
