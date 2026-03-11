BlinkCounter

Outil Python permettant de compter automatiquement les clignements d’yeux dans des vidéos à partir des landmarks faciaux MediaPipe.

Le logiciel est conçu pour traiter de grandes cohortes de sujets où chaque sujet possède :

une vidéo étalon

plusieurs vidéos expérimentales

Le programme produit un tableau CSV du nombre de clignements par sujet et par condition.

Principe de fonctionnement

Le logiciel utilise MediaPipe Face Landmarker pour détecter les points du visage.

Pour chaque frame :

détection du visage

extraction des landmarks des paupières

calcul du Eye Aspect Ratio (EAR)

𝐸
𝐴
𝑅
=
∣
∣
𝑝
2
−
𝑝
6
∣
∣
+
∣
∣
𝑝
3
−
𝑝
5
∣
∣
2
×
∣
∣
𝑝
1
−
𝑝
4
∣
∣
EAR=
2×∣∣p1−p4∣∣
∣∣p2−p6∣∣+∣∣p3−p5∣∣
	​


Le EAR mesure l’ouverture de l’œil.

EAR	état
élevé	œil ouvert
faible	œil fermé
Normalisation par sujet

La morphologie des yeux varie selon les individus.

Pour rendre les résultats comparables, une normalisation est réalisée à partir d’une vidéo étalon.

Sur la vidéo étalon :

extraction de tous les EAR

calcul de :

EAR_open_ref   = percentile 95
EAR_closed_ref = percentile 5

Puis le EAR est normalisé :

EAR_norm = (EAR - EAR_closed_ref) / (EAR_open_ref - EAR_closed_ref)

Cela permet de comparer les clignements indépendamment de la morphologie.

Détection des clignements

La détection repose sur une hystérésis :

fermeture si  EAR_norm < close_threshold
réouverture si EAR_norm > open_threshold

Un clignement est validé si la fermeture dure entre :

min_closed_frames
max_closed_frames
Calibration automatique

Chaque dossier sujet contient un fichier :

attendu.txt

Ce fichier contient le nombre réel de clignements observés dans la vidéo étalon.

Le script teste plusieurs couples de seuils :

close_threshold
open_threshold

et choisit ceux qui reproduisent le mieux ce nombre.

Ces seuils sont ensuite utilisés pour toutes les vidéos du sujet.

Structure des données

Le dossier video/ doit contenir un dossier par sujet.

video/
├── dry/
│   └── test.mp4
├── sujet_001/
│   ├── étalon.mp4
│   ├── attendu.txt
│   ├── normal.mp4
│   └── chaussette.mp4
├── sujet_002/
│   ├── étalon.mp4
│   ├── attendu.txt
│   └── normal.mp4

Contenu de attendu.txt :

17
Résultat

Le programme produit un CSV :

subject,normal,chaussette
sujet_001,18,11
sujet_002,15,
sujet_003,,9

lignes → sujets

colonnes → conditions expérimentales

Modes d'exécution
Dry run

Analyse une seule vidéo dans :

video/dry/

Utilisé pour :

vérifier la détection

visualiser le EAR

vérifier les seuils

Real run

Analyse complète :

lecture de étalon.mp4

calibration des seuils

analyse des vidéos expérimentales

génération du CSV final

Arguments du script
--model

Chemin vers le modèle MediaPipe.

--model face_landmarker.task
--dry-run

Mode test.

Par défaut :

true

Pour lancer l’étude complète :

--dry-run false
--show

Affiche la vidéo annotée pendant l’analyse.

Affiche :

EAR brut

EAR normalisé

nombre de clignements

⚠️ disponible uniquement en dry run

--csv

Nom du fichier CSV de sortie.

Par défaut :

results.csv

Exemple :

--csv etude.csv
--min-closed-frames

Durée minimale d’une fermeture pour valider un clignement.

Défaut :

2
--max-closed-frames

Durée maximale d’une fermeture pour valider un clignement.

Défaut :

12
Exemples d'utilisation
Test simple
python betterBlink.py --model face_landmarker.task
Test avec affichage
python betterBlink.py --model face_landmarker.task --show
Lancer l'étude complète
python betterBlink.py --model face_landmarker.task --dry-run false
Export CSV personnalisé
python betterBlink.py --model face_landmarker.task --dry-run false --csv etude.csv
Dépendances

Python ≥ 3.10

Installer les bibliothèques :

pip install mediapipe opencv-python numpy
Performances

Chaque vidéo est lue une seule fois.

Le coût principal est la détection des landmarks faciaux via MediaPipe.

Le pipeline est conçu pour traiter de grandes cohortes sans relire les vidéos inutilemen