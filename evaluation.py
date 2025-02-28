import json
import cv2
import numpy as np

from Methodes.detectStairs2 import detect_stairs2


def update_json_with_predictions(json_file):
    """
    Met à jour le fichier JSON avec les prédictions de marches.

    :param json_file: Chemin du fichier JSON contenant les images et la vérité terrain.
    """
    # Charger le fichier JSON
    with open(json_file, "r") as file:
        data = json.load(file)

    # Parcourir chaque image dans les données
    for entry in data["data"]:
        image_path = entry["image"]
        actual_count = entry["actual_count"]

        # Obtenir la prédiction du nombre de marches
        predicted_count = detect_stairs2(image_path)

        # Calculer l'erreur absolue
        absolute_error = abs(predicted_count - actual_count)

        # Mettre à jour les valeurs dans le JSON
        entry["predicted_count"] = predicted_count
        entry["absolute_error"] = absolute_error

    # Sauvegarder les mises à jour dans le fichier JSON
    with open(json_file, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Fichier JSON mis à jour avec les prédictions.")


# update_json_with_predictions("stairs_data.json")

def evaluate_predictions(json_file):
    """
    Calcule des métriques d'évaluation basées sur les prédictions du JSON.

    :param json_file: Chemin du fichier JSON mis à jour.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    # Extraire les erreurs absolues
    absolute_errors = [entry["absolute_error"] for entry in data["data"]]

    # Calcul des métriques
    mae = np.mean(absolute_errors)  # Mean Absolute Error
    rmse = np.sqrt(np.mean(np.square(absolute_errors)))  # Root Mean Squared Error

    print(f"📊 Évaluation des performances :")
    print(f"🔹 Erreur Absolue Moyenne (MAE) : {mae:.2f} marches")
    print(f"🔹 Erreur Quadratique Moyenne (RMSE) : {rmse:.2f} marches")

    return mae, rmse

# evaluate_predictions("stairs_data.json")