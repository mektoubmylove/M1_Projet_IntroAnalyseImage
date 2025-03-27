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

### extract_features
Extraction des caractéristiques des images
- L'image est convertie en niveaux de gris pour simplifier l'analyse.
- Une détection des contours est effectuée à l'aide de l'algorithme Canny pour identifier les bords dans l'image. 
- Ensuite, la transformée de Hough est utilisée pour détecter les lignes dans l'image. En particulier, les lignes horizontales (qui pourraient correspondre aux marches de l'escalier) sont isolées. 
- Plusieurs statistiques sont calculées à partir des lignes détectées : le nombre de lignes horizontales, leur position moyenne sur l'axe vertical, et l'écart-type de ces positions. Ces statistiques servent de caractéristiques pour le modèle.

Préparation des données d'entraînement
- Les données extraites des images, c'est-à-dire les caractéristiques (comme le nombre de lignes horizontales et leur position) et les annotations réelles (le nombre de marches réel), sont collectées dans deux tableaux : X pour les caractéristiques et y pour les étiquettes (nombre de marches). 
Normalisation des caractéristiques
- Avant de procéder à l'entraînement du modèle, les caractéristiques extraites sont normalisées à l'aide de la méthode StandardScaler de la bibliothèque scikit-learn.

Entraînement du modèle avec Leave-One-Out Cross-Validation (LOO)

- Une approche de validation croisée Leave-One-Out (LOO) est utilisée pour entraîner et évaluer le modèle. L'idée de LOO est que pour chaque itération, une seule image est utilisée comme test, tandis que le reste des images est utilisé pour entraîner le modèle. Cela permet de tester la capacité du modèle à généraliser à de nouvelles données tout en utilisant efficacement toutes les images disponibles pour l'entraînement.

- Pendant chaque itération :

  - Le modèle Random Forest est entraîné sur l'ensemble des données d'entraînement et testé sur une seule image (celle qui a été mise de côté).

  - L'erreur entre la prédiction du modèle et la vérité terrain (le nombre réel de marches) est calculée. L'erreur absolue est utilisée comme mesure de la performance du modèle.

### extract_histogram_features

Extraction des caractéristiques des images
- Redimensionnement de l'image : Chaque image est redimensionnée à une hauteur de 1024 pixels. Cela permet de standardiser les tailles d'images avant leur analyse.
- Conversion en niveaux de gris : L'image est convertie en une image en niveaux de gris, ce qui est une étape essentielle avant toute détection de contours ou de lignes.
- Détection des contours : La méthode cv2.Canny est utilisée pour détecter les contours dans l'image. Cela permet de repérer les zones de changement significatif dans l'image.
- Détection des lignes : Ensuite, la méthode cv2.HoughLinesP est utilisée pour détecter les lignes dans l'image (en particulier les lignes horizontales représentant les marches d'escalier).
- Création d'un histogramme : Un histogramme est construit pour chaque image. Pour chaque ligne détectée, la position en y (coordonnée verticale) des lignes est utilisée pour déterminer dans quel "bin" de l'histogramme cette ligne doit être comptabilisée. Le but est de créer un résumé de la répartition verticale des lignes détectées.

La sortie de cette fonction est un histogramme, qui est un vecteur de caractéristiques représentant l'image.

Normalisation des données :
- La normalisation des caractéristiques est effectuée à l'aide de StandardScaler de sklearn. Cela garantit que chaque caractéristique a une moyenne de 0 et un écart-type de 1

Entraînement du modèle avec Leave-One-Out Cross-Validation (LOO) :

- La méthode Leave-One-Out Cross-Validation (LOO) est utilisée pour évaluer la performance du modèle.
- Dans chaque itération, une image est utilisée comme ensemble de test, tandis que toutes les autres sont utilisées pour entraîner le modèle.
- Un Random Forest Regressor est créé avec 1000 arbres (n_estimators=1000), et le modèle est entraîné sur les données d'entraînement à chaque itération.
- Le modèle fait ensuite des prédictions sur l'image test et l'erreur absolue est calculée (la différence entre le nombre réel de marches et la prédiction).

Calcul de l'erreur absolue moyenne (MAE) :

- À la fin de toutes les itérations de la validation croisée, l'erreur absolue moyenne (MAE) est calculée en moyennant toutes les erreurs absolues obtenues pour chaque test. 
- Cette métrique permet de mesurer la précision du modèle : plus la MAE est faible, mieux le modèle prédit le nombre de marches.

### Comparaison des performances des modèles

| Projet                     | MAE après entraînement | MAE sur ensemble de test |
|----------------------------|------------------------|--------------------------|
| projectImageDetectStairs3  | 4.08                   | 5.16                     |
| projectImageDetectStairs5  | 1.30                   | 5.61                     |
| extract_features           | 1.74                   | 5.17                     |
| extract_histogram_features | 1.76                   | 3.89                     |

###  Analyse :
- **projectImageDetectStairs5** a une meilleure précision après entraînement (**MAE = 1.30** contre **4.08** pour projectImageDetectStairs3).
- Cependant, il se généralise légèrement moins bien en test (**MAE = 5.61** contre **5.16**), ce qui peut indiquer un surajustement.
- extract_features et extract_histogram_features montrent des résultats assez proches, avec des MAE respectivement de 1.74 et 1.76 après l'entraînement, mais une meilleure généralisation sur l'ensemble de test, où leurs MAE sont respectivement 5.17 et 3.89. 

 **Conclusion** : Bien que projectImageDetectStairs5 soit plus précis sur l’entraînement, son score de test plus élevé indique une perte de généralisation (surajustement).

Les méthodes comme extract_features et extract_histogram_features semblent être plus équilibrées en termes de généralisation, offrant des résultats plus stables sur l’ensemble de test, même si elles ne sont pas aussi précises que projectImageDetectStairs5 sur les données d’entraînement.
