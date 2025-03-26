#  Détection et Comptage de Marches d’Escalier
## Groupe 3 : Amine AISSAOUI, Emmanuel Cattan, Abdelkader SOUAYAH
## 📌 Contexte
Ce projet vise à concevoir un programme capable de détecter et compter le nombre de marches d’un escalier.

## 🎯 Objectif
L’objectif principal est de développer une méthode fiable pour :
- Détecter les marches sur une image d’escalier.
- Compter le nombre de marches.
- Évaluer les performances du système sur un jeu de données construit pour l’occasion.
- Identifier les limites de la méthode et proposer des améliorations.

### Exemple d'images

| Image 1 | Image 2 | Image 3 |
|---------|---------|---------|
| ![Image 1](assets/img1preview.jpg) | ![Image 2](assets/img2preview.png) | ![Image 3](assets/img3preview.jpg) |


## 📂 Structure du Projet
- **📁 data/** : Contient les images d’escalier avec annotations du nombre de marches.
- **📁 Méthodes/** : Plusieurs approches pour la détection et le comptage des marches (readme inclus)
- **📁 evaluations/** : Résultats des tests et évaluations sur les ensembles train et val (readme inclus)
- **📁 randomForest/** : entrainement de modeles à partir des features qu'on a obtenu via nos méthodes
- **📜 split.py**: divise un ensemble d'images en train (60%), validation (20%) et test (20%) 
- **📜 utils.py**: fonctions implémentés sans opencv (composante connexe, gaussianblur etc )

## 🛠️ Méthodologie
1. **Construction d’un jeu de données**
   - Acquisition d’images de différents escaliers sous divers angles et conditions lumineuses.
   - Annotation manuelle du nombre de marches pour chaque image (vérité terrain).

2. **Détection et Comptage des Marches**
   - TODO

3. **Évaluation des Performances**
   - Comparaison des résultats du programme avec les annotations manuelles.
   - Calcul de métriques comme la MAE.

4. **Critique et Amélioration**
   - TODO

## 🚀 Installation et Utilisation
### 📥 Prérequis
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

### Exécution TODO
- 

