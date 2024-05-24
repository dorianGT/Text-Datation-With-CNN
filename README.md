# Datation de Texte avec CNN

## Description
Ce projet vise à prédire la date de rédaction de textes en utilisant des réseaux de neurones convolutifs (CNN). Notre modèle analyse des séries temporelles de données textuelles pour extraire des caractéristiques utiles et faire des prédictions précises. L'architecture multi-branche utilise des couches de convolution profondes pour capter des motifs locaux. Le rapport détaille les étapes du projet, y compris la conception du réseau, l'entraînement du modèle, et les tests de performance. Les résultats montrent que notre approche basée sur les CNN permet des prédictions précises mais peut être optimisée davantage.

## Structures
Le projet est divisé en 2 parties :
- Le fichier Main_TER.ipynb contient le code du projet.
- Les fichiers test, train et validation contiennent les différents livres pour notre jeu de données.

## Executable
Vous pouvez accéder à un exécutable pour essayer les modèles obtenus en suivant ce lien :

C'est un simple script Python utilisable en ligne de commande comme suit :
```bash
python script_predict.py

Vous aurez ensuite deux champs à remplir, la localisation du model puis la localisation du ou des fichiers dont vous souhaités prédire la date.
