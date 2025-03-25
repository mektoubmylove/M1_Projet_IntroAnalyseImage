#  Détection et Comptage de Marches d’Escalier
## Groupe 3 : Amine AISSAOUI, Emmanuel Cattan, Abdelkader SOUAYAH

## Résultats MAE

| Méthode                     | MAE Train | MAE Validation |
|-----------------------------|----------:|--------------:|
| **detectStairs1**           | 18.33     | 10.05        |
| **detectStairs2**           | 4.57      | 4.50         |
| **detectStairs3**           | 4.48      | 4.35         |
| **detectStairs4**           | 4.36      | 3.60         |
| **detectStairs5**           | 4.28      | 3.80         |
| **detectStairs6**           | 4.57      | 3.45         |
| **findContours**            | 3.81      | 3.85         |
| **findContoursRectangle1**  | 5.03      | 4.95         |

La méthode findContours offre les meilleures performances avec une erreur moyenne absolue (MAE) de 3.81 en entraînement et 3.85 en validation, surpassant les autres approches. Les méthodes detectStairs3, detectStairs4, detectStairs5, et detectStairs6 obtiennent également de bons résultats, tandis que detectStairs1 est nettement moins performante avec une MAE beaucoup plus élevée.