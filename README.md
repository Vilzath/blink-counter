# BlinkCounter

BlinkCounter est un outil Python permettant de compter automatiquement les clignements d’yeux dans des vidéos à partir des landmarks faciaux MediaPipe.

Le logiciel est conçu pour traiter des cohortes importantes de sujets avec :
- une vidéo étalon
- plusieurs vidéos expérimentales

Le programme produit plusieurs fichiers CSV exploitables pour analyse scientifique.


========================================================
PRINCIPE DE FONCTIONNEMENT
========================================================

Le pipeline fonctionne en trois étapes principales :

1. Analyse de la vidéo étalon
2. Calibration automatique des seuils
3. Analyse des vidéos expérimentales


--------------------------------------------------------
1. EXTRACTION DU SIGNAL EAR
--------------------------------------------------------

Pour chaque frame :

- détection du visage via MediaPipe
- extraction des points des paupières
- calcul du Eye Aspect Ratio (EAR)


--------------------------------------------------------
DÉFINITION DU EAR (Eye Aspect Ratio)
--------------------------------------------------------

Le Eye Aspect Ratio (EAR) est une mesure géométrique de l’ouverture de l’œil basée sur la position de six points (landmarks) autour de l’œil.

Ces points sont fournis par le modèle MediaPipe Face Landmarker.

Pour un œil donné, on définit :

- p1 et p4 : coins horizontaux de l’œil
- p2 et p6 : points verticaux externes
- p3 et p5 : points verticaux internes

La formule du EAR est :

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

où ||pi - pj|| représente la distance euclidienne entre deux points.


--------------------------------------------------------
INTERPRÉTATION DU EAR
--------------------------------------------------------

Le EAR est une grandeur sans unité qui décrit le rapport entre :

- la hauteur de l’œil (verticale)
- sa largeur (horizontale)

Comportement typique :

- œil ouvert :
  les distances verticales sont élevées
  → EAR élevé

- œil fermé :
  les distances verticales deviennent proches de zéro
  → EAR faible

Ainsi :

- EAR diminue fortement lors d’un clignement
- EAR remonte lors de la réouverture


--------------------------------------------------------
PROPRIÉTÉS IMPORTANTES
--------------------------------------------------------

1. Invariance relative à l’échelle :
   le EAR est un ratio → peu sensible à la distance caméra/sujet

2. Sensibilité aux landmarks :
   la précision dépend directement de la qualité du tracking facial

3. Sensibilité au bruit :
   de petites variations de landmarks peuvent créer du jitter dans le EAR

4. Variabilité inter-individuelle :
   le EAR moyen dépend de la morphologie des yeux

→ d’où la nécessité de la normalisation par sujet (EAR_norm)


--------------------------------------------------------
UTILISATION DANS LE PROJET
--------------------------------------------------------

Dans BlinkCounter :

- le EAR est calculé pour chaque frame
- puis normalisé (EAR_norm)
- la détection de clignement repose sur les variations temporelles de EAR_norm


--------------------------------------------------------
2. NORMALISATION PAR SUJET
--------------------------------------------------------

Pour compenser les différences morphologiques :

Sur la vidéo étalon :

- EAR_open_ref   = percentile 95
- EAR_closed_ref = percentile 5

Puis :

EAR_norm = (EAR - EAR_closed_ref) / (EAR_open_ref - EAR_closed_ref)

Toutes les détections sont effectuées sur EAR_norm.


--------------------------------------------------------
3. CALIBRATION AUTOMATIQUE
--------------------------------------------------------

Chaque sujet possède un fichier :

attendu.txt

Ce fichier contient le nombre réel de clignements sur l’étalon.

Le script teste plusieurs couples :

- close_threshold
- open_threshold

Contraintes :

- close ∈ [0.10 ; 0.45]
- open ∈ [0.30 ; 0.85]
- open > close + 0.08

Le couple optimal minimise :

|clignements_prédits - clignements_attendus|


========================================================
DÉTECTION DES CLIGNEMENTS
========================================================

Le système utilise une machine à états avec hystérésis.

États :
- o → neutre
- - → fermeture
- + → réouverture


--------------------------------------------------------
LOGIQUE
--------------------------------------------------------

1. Entrée en fermeture :

EAR_norm < close_threshold

2. Phase fermée :

in_closed = True
closed_len += 1

3. Validation du clignement :

EAR_norm > open_threshold

ET

min_closed_frames ≤ closed_len ≤ max_closed_frames


--------------------------------------------------------
POINT CRITIQUE : DURÉE DU CLIGNEMENT
--------------------------------------------------------

La durée mesurée n’est PAS uniquement le temps œil fermé.

Elle correspond à :

temps entre :
- première frame < close_threshold
- première frame > open_threshold

Donc inclut :
- fermeture
- plateau
- réouverture

Conséquence importante :

Un clignement peut être rejeté si :

closed_len > max_closed_frames

même s’il est visuellement valide.


--------------------------------------------------------
EFFETS POSSIBLES
--------------------------------------------------------

- clignements lents rejetés
- deux clignements proches fusionnés
- dépendance forte aux seuils


========================================================
STRUCTURE DES DONNÉES
========================================================

video/
  dry/
    test.mp4
  sujet_001/
    étalon.mp4
    attendu.txt
    Coloriage.mp4
    Jeu SANS chrono.mp4
    ...
  sujet_002/
    ...


========================================================
FICHIERS GÉNÉRÉS
========================================================

results.csv :
- nombre de clignements
- intervalle moyen
- min / max
- clignements par minute
- Face Detect Rate

Essential.csv :
- clignements par minute uniquement (sans étalon)

Erreur relative.txt :
- erreur moyenne relative
- biais moyen
- écart type

details/ :
- fichiers EAR détaillés par sujet

seuil.csv :
- seuils et statistiques du signal EAR


========================================================
PARAMÈTRES PRINCIPAUX
========================================================

DEFAULT_MIN_CLOSED_FRAMES = 1
DEFAULT_MAX_CLOSED_FRAMES = 16
DEEPDATA_STEP_FRAMES = 5


========================================================
LIMITATIONS
========================================================

- rejet des clignements longs
- fusion possible
- dépendance aux seuils
- dépendance au tracking visage


========================================================
UTILISATION
========================================================

python betterBlink.py --model face_landmarker.task
python betterBlink.py --model face_landmarker.task --show
python betterBlink.py --model face_landmarker.task --dry-run false


========================================================
CONCLUSION
========================================================

Le système est :

- automatisé
- scalable
- reproductible

Mais :

- basé sur une logique déterministe stricte
- nécessite une interprétation prudente des résultats