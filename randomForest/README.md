#  Détection et Comptage de Marches d’Escalier
## Groupe 3 : Amine AISSAOUI, Emmanuel Cattan, Abdelkader SOUAYAH


### projetImageDetecStairs3

#### 1. Prétraitement de l’image

    L’image est convertie en niveau de gris pour simplifier l’analyse.

    Un flou gaussien est appliqué pour réduire le bruit.

    Une détection de contours est effectuée en combinant les filtres de Sobel et la méthode d’Otsu.

#### 2. Détection de l’angle dominant

    La transformée de Hough est utilisée pour détecter les lignes dans l’image.

    Les angles des lignes détectées sont analysés, et l’angle médian est retenu pour corriger l’orientation de l’image.

#### 3. Correction de l’orientation

    Si l’angle dominant est >5°, l’image est tournée pour réaligner les escaliers horizontalement.

#### 4. Détection des marches

    Une nouvelle détection des contours est effectuée sur l’image corrigée.

    La transformée de Hough est à nouveau utilisée pour repérer les lignes horizontales, correspondant aux marches.

    Un regroupement des lignes est effectué pour éviter les doublons et mieux estimer le nombre de marches.

#### 5. Extraction des caractéristiques

    Pour chaque image, les informations suivantes sont extraites
    - Nombre de marches détectées
    - Distance moyenne entre les marches
    - Distance maximale entre deux marches

Les résultats sont enregistrés dans un fichier CSV ("features.csv").
#### 6. Entraînement du modèle

    Les caractéristiques extraites sont utilisées pour entraîner un modèle Random Forest Regressor.

    Le modèle apprend à prédire le nombre réel de marches en comparant les résultats aux données annotées dans gt.json.

    L’erreur absolue moyenne (MAE) est calculée pour évaluer la précision du modèle.

    Le modèle entraîné est sauvegardé sous "stair_detector_model.pkl".

#### 7. Test et validation

    Le modèle est testé sur de nouvelles images.

    Pour chaque image testée, il effectue la même détection et prédit le nombre de marches.

    Si une vérité terrain est disponible, une erreur absolue est calculée entre la valeur réelle et la valeur prédite.

    Les résultats finaux sont sauvegardés dans "gt_results.json".

### projetImageDetectStairs5

#### 1 Traitement d'Image et Prétraitement

    Le code commence par appliquer une série de transformations d'image pour détecter les escaliers présents dans les images.

    Conversion en niveaux de gris : Les images sont converties en niveaux de gris pour simplifier le traitement.

    Flou Gaussien : Un flou est appliqué pour réduire le bruit de l'image, ce qui facilite la détection des contours.

    Détection de contours : On utilise l'opérateur de Sobel pour détecter les bords, et on combine les résultats des axes X et Y. Ensuite, on applique la méthode Otsu pour déterminer un seuil optimal et binariser l'image.

#### 2 Détection des Lignes et Orientation de l'Image

    Ensuite, l'algorithme détecte les lignes dans l'image, qui sont censées correspondre aux marches d'escalier.

    Détection d'angle dominant : La fonction detect_dominant_angle() utilise la transformée de Hough pour identifier les lignes dans l'image, puis calcule l'angle dominant de ces lignes. Si l'angle est significatif, l'image est pivotée pour corriger l'orientation (en la tournant si nécessaire).

#### 3 Détection des Marches

    Après l'ajustement de l'orientation de l'image, la détection des marches s'effectue en plusieurs étapes :

    Appliquer Canny : Utilisation de la méthode Canny pour détecter les bords dans l'image.

    HoughLinesP : Cette méthode détecte les lignes dans les bords de l'image, en particulier celles représentant les marches d'escalier.

    Fusionner les lignes proches : Une fois les lignes détectées, celles qui sont proches les unes des autres (selon une distance donnée) sont fusionnées pour obtenir une estimation du nombre de marches.

    Calculer les distances entre les marches : On calcule la distance moyenne et maximale entre les lignes détectées pour caractériser l'espacement des marches.

#### 4 Extraction des Caractéristiques et Entraînement du Modèle

    Une fois que les caractéristiques des marches (nombre de marches, distance moyenne, distance maximale) sont extraites, elles sont sauvegardées dans un fichier CSV pour être utilisées par le modèle de machine learning.
    Caractéristiques extraites :

    Nombre de marches détectées.

    Distance moyenne entre les marches.

    Distance maximale entre les marches.

    Entraînement du modèle RandomForest

    Chargement des caractéristiques : Le fichier features.csv contient les caractéristiques extraites de l'étape précédente.

    Préparation des données : Le code sépare les caractéristiques des données d'entraînement de la vérité terrain (nombre réel de marches).

    Entraînement du modèle : Un RandomForestRegressor est entraîné pour prédire le nombre de marches sur un ensemble de données d'entraînement. Le modèle est ensuite évalué avec l'erreur absolue moyenne (MAE).

#### 5 Prédiction et Évaluation du Modèle
    Test du modèle sur de nouvelles images :

    Une fois le modèle entraîné, il est testé sur de nouvelles images pour vérifier sa capacité à prédire le nombre de marches d'escalier.

    Le modèle applique les mêmes étapes de prétraitement et de détection des marches sur les nouvelles images.

    Les résultats de la prédiction sont comparés à la vérité terrain (si elle est disponible), et l'erreur absolue est calculée pour évaluer la performance du modèle.

    Les résultats sont ensuite sauvegardés dans un fichier gt_results.json, qui contient :

    Le nombre réel de marches.

    Le nombre prédit par le modèle.

    L'erreur absolue entre la valeur réelle et la prédite.

### Comparaison des performances des modèles

| Projet                        | MAE après entraînement | MAE sur ensemble de test |
|-------------------------------|------------------------|--------------------------|
| projectImageDetectStairs3     | 4.08                   | 5.16                     |
| projectImageDetectStairs5     | 1.30                   | 5.61                     |

###  Analyse :
- **projectImageDetectStairs5** a une meilleure précision après entraînement (**MAE = 1.30** contre **4.08** pour projectImageDetectStairs3).
- Cependant, il se généralise légèrement moins bien en test (**MAE = 5.61** contre **5.16**), ce qui peut indiquer un surajustement.

📌 **Conclusion** : Bien que projectImageDetectStairs5 soit plus précis sur l’entraînement, son score de test suggère une  perte de généralisation.
