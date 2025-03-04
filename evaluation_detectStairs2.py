import json
import os

import cv2
import numpy as np

from Methodes.detectStairs2 import detect_stairs2


def update_json_with_predictions(train_dir, json_file, output_file):
    """
    Parcourt le dossier train, applique detect_stairs2(), et met à jour gt.json avec les résultats.

    :param train_dir: Chemin du dossier contenant les images d'entraînement.
    :param json_file: Fichier JSON contenant la vérité terrain (gt.json).
    :param output_file: Fichier JSON mis à jour avec les prédictions.
    """
    # Charger le fichier JSON (ground truth)
    with open(json_file, "r") as file:
        data = json.load(file)

    # Convertir en dictionnaire pour un accès rapide (image -> actual_count)
    gt_dict = {entry["image"]: entry["actual_count"] for entry in data["data"]}

    total_images = len(gt_dict)
    failed_images = 0  # Compteur d'images non traitées

    # Parcourir toutes les images du dossier train
    for image_name in os.listdir(train_dir):
        image_path = os.path.join(train_dir, image_name)

        if image_name in gt_dict:  # Vérifier si l'image est bien référencée dans gt.json
            actual_count = gt_dict[image_name]

            try:
                # Obtenir la prédiction du nombre de marches
                predicted_count = detect_stairs2(image_path)

                if predicted_count is not None:
                    absolute_error = abs(predicted_count - actual_count)
                else:
                    raise ValueError("Détection échouée")

            except Exception as e:
                print(f"❌ Erreur avec {image_name}: {e}")
                predicted_count, absolute_error = None, None
                failed_images += 1

            # Mettre à jour les valeurs dans le JSON
            for entry in data["data"]:
                if entry["image"] == image_name:
                    entry["predicted_count"] = predicted_count
                    entry["absolute_error"] = absolute_error
                    break  # Sortir dès que l'image est trouvée dans le JSON

    # Sauvegarder les mises à jour dans un fichier JSON
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

    print(f"\n✅ Fichier JSON mis à jour : {output_file}")
    print(f"📊 Images traitées : {total_images - failed_images}/{total_images}")
    print(f"❌ Images non traitées : {failed_images}")


def evaluate_predictions(json_file):
    """
    Calcule des métriques d'évaluation basées sur les prédictions enregistrées dans le JSON.

    :param json_file: Chemin du fichier JSON mis à jour.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    absolute_errors = []
    valid_entries = 0  # Compteur pour éviter les erreurs en cas de valeurs nulles

    for entry in data["data"]:
        if entry["absolute_error"] is not None:
            absolute_errors.append(entry["absolute_error"])
            valid_entries += 1

    if valid_entries > 0:
        # Calcul des métriques seulement si des prédictions valides existent
        mae = np.mean(absolute_errors)  # Mean Absolute Error
        rmse = np.sqrt(np.mean(np.square(absolute_errors)))  # Root Mean Squared Error

        print("\n📊 **Évaluation des performances :**")
        print(f"🔹 Erreur Absolue Moyenne (MAE) : {mae:.2f} marches")
        print(f"🔹 Erreur Quadratique Moyenne (RMSE) : {rmse:.2f} marches")
        return mae, rmse
    else:
        print("❌ Aucune prédiction valide disponible pour évaluation.")
        return None, None


# Exécution du script
train_directory = "data/train"
ground_truth_json = "gt.json"
updated_json = "gt_result_train_detectStairs2.json"

# 1️⃣ Mise à jour des prédictions
update_json_with_predictions(train_directory, ground_truth_json, updated_json)

# 2️⃣ Calcul des métriques
evaluate_predictions(updated_json)