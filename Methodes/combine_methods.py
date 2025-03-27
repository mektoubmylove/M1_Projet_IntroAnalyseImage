import numpy as np
import cv2
import os
from collections import Counter

from Methodes.findContours import findContours
from Methodes.detectStairs2 import detect_stairs2 as dt2
from Methodes.detectStairs3 import detect_stairs_with_homography as dt3
from Methodes.detectStairs4 import detect_stairs_with_homography as dt4
from Methodes.detectStairs5 import detect_stairs_with_homography as dt5
from Methodes.detectStairs6 import detect_stairs_with_homography as dt6
from Methodes.find_Contours_Homographie import findContours2


def compute_average_stairs(image_path, methods):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    predictions = []

    for method in methods:
        try:
            count = method(image_path)

            if isinstance(count, tuple):
                count = count[0]

            if isinstance(count, (int, float)):
                predictions.append(count)
            else:
                print(f" {method.__name__} a retourné un format invalide : {count}")

        except Exception as e:
            print(f"Erreur avec {method.__name__} : {e}")

    print("Résultats des méthodes :", predictions)

    return np.mean(predictions) if predictions else 0

def compute_majority_vote(image_path, methods):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    predictions = []

    for method in methods:
        try:
            count = method(image_path)

            if isinstance(count, tuple):
                count = count[0]

            if isinstance(count, (int, float)):
                predictions.append(int(count))
            else:
                print(f" {method.__name__} a retourné un format invalide : {count}")

        except Exception as e:
            print(f"Erreur avec {method.__name__} : {e}")

    print("Résultats des méthodes :", predictions)  # Debugging

    if not predictions:
        return 0

    # Appliquer le vote majoritaire
    counter = Counter(predictions)
    majority_vote = counter.most_common(1)[0][0]  # Prend la valeur la plus fréquente

    return majority_vote

def hybrid(image_path, methods, threshold=2):
    """
    Combine le vote majoritaire et une moyenne filtrée pour une estimation robuste.

    :param image_path: Chemin de l'image à analyser.
    :param methods: Liste des fonctions de détection.
    :param threshold: Écart maximum accepté autour du vote majoritaire.
    :return: Estimation finale du nombre de marches.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    predictions = []

    for method in methods:
        try:
            count = method(image_path)
            if isinstance(count, tuple):
                count = count[0]  # On ne garde que le nombre de marches
            if isinstance(count, (int, float)):
                predictions.append(int(count))
        except Exception as e:
            print(f"Erreur avec {method.__name__} : {e}")

    print("Prédictions des méthodes :", predictions)

    if not predictions:
        return 0

    # Étape 1 : Vote majoritaire
    counter = Counter(predictions)
    majority_vote = counter.most_common(1)[0][0]

    # Étape 2 : Filtrage des valeurs trop éloignées
    filtered_predictions = [p for p in predictions if abs(p - majority_vote) <= threshold]

    # Étape 3: Moyenne filtrée
    final_prediction = np.mean(filtered_predictions) if filtered_predictions else majority_vote

    return round(final_prediction)
# Exemple d'utilisation
#image_path = "../data/train/Groupe1_Image2.jpg"  # Remplace avec ton image
#methods = [dt2,dt3, dt4, dt5, dt6, findContours,findContours2]  # Liste des méthodes à tester

#average_stairs = compute_average_stairs(image_path, methods)
#print(f"Nombre moyen de marches détectées : {average_stairs:.2f}")
