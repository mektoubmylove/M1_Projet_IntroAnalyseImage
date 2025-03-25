import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt  # Import de matplotlib pour l'affichage

def detect_stairs_rectangles(image_path):
    """
    Détecte et compte les marches d'un escalier en identifiant les formes rectangulaires.

    :param image_path: Chemin de l'image contenant l'escalier.
    """
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=800)
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
            if w > 30 and h > 15:
                detected_rectangles.append((x, y, w, h))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Nombre total de rectangles détectés
    num_stairs = len(detected_rectangles)
    cv2.putText(image, f"Nombre de marches : {num_stairs}",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    """
    # Affichage des résultats avec Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Image des contours détectés
    axes[0].imshow(edges, cmap='gray')
    axes[0].set_title("Contours détectés")
    axes[0].axis("off")

    # Image des marches détectées avec rectangles
    axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Marches détectées")
    axes[1].axis("off")

    # Afficher la fenêtre
    plt.tight_layout()
    plt.show()
    """
    return num_stairs

# Test avec une image
#detect_stairs_rectangles("../data/train/Groupe1_Image3.jpg")
