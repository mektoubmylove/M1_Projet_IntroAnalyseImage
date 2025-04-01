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
                if abs(y - y1) < 15:  
                    okey = False
                    break

            if okey:
                cv2.line(out_img, (0, y1), (img.shape[1], y1), (0, 0, 255), 3, cv2.LINE_AA)
                y_keeper_for_lines.append(y1)
                stair_counter += 1

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