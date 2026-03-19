# BlinkCounter

BlinkCounter est un outil Python permettant de **compter automatiquement les clignements d’yeux dans des vidéos** à partir des **landmarks faciaux MediaPipe**.

Le logiciel est conçu pour traiter **des cohortes importantes de sujets** où chaque sujet possède :

- une **vidéo étalon**
- plusieurs **vidéos expérimentales**

Le programme produit un **fichier CSV contenant le nombre de clignements par sujet et par condition expérimentale**.

---

# Principe de fonctionnement

Le logiciel utilise **MediaPipe Face Landmarker** pour détecter les points du visage.

Pour chaque image de la vidéo :

1. détection du visage  
2. extraction des points des paupières  
3. calcul du **Eye Aspect Ratio (EAR)**

Formule :

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Le EAR mesure l’ouverture de l’œil :

| EAR | Interprétation |
|----|----|
| élevé | œil ouvert |
| faible | œil fermé |

---

# Normalisation par sujet

La morphologie des yeux varie selon les individus.  
Pour rendre les résultats comparables, le logiciel normalise le EAR à partir d’une **vidéo étalon**.

Sur la vidéo étalon :

```
EAR_open_ref   = percentile 95
EAR_closed_ref = percentile 5
```

Puis :

```
EAR_norm = (EAR - EAR_closed_ref) / (EAR_open_ref - EAR_closed_ref)
```

Le clignement est détecté sur **EAR_norm**, pas sur le EAR brut.

---

# Détection des clignements

Le système utilise une **hystérésis** :

```
fermeture si  EAR_norm < close_threshold
réouverture si EAR_norm > open_threshold
```

Un clignement est validé si la durée de fermeture est comprise entre :

```
min_closed_frames
max_closed_frames
```

Cette logique évite :

- les faux positifs
- les oscillations autour d’un seuil unique

---

# Calibration automatique

Chaque dossier sujet contient un fichier :

```
attendu.txt
```

Ce fichier contient le **nombre réel de clignements observés dans la vidéo étalon**.

Le programme teste plusieurs couples de seuils :

```
close_threshold
open_threshold
```

et choisit ceux qui reproduisent le mieux ce nombre.

Les seuils calibrés sont ensuite utilisés pour les vidéos expérimentales.

---

# Structure des données

Le dossier `video/` doit contenir un dossier par sujet.

```
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
```

Contenu de `attendu.txt` :

```
17
```

---

# Résultat

Le programme produit un CSV :

```
subject,normal,chaussette
sujet_001,18,11
sujet_002,15,
sujet_003,,9
```

- lignes → sujets  
- colonnes → conditions expérimentales  

---

# Modes d'exécution

## Dry run

Analyse une seule vidéo située dans :

```
video/dry/
```

Utilisé pour :

- vérifier la détection
- visualiser le EAR
- tester les paramètres

---

## Real run

Analyse complète :

1. analyse de `étalon.mp4`
2. calibration des seuils
3. analyse des vidéos expérimentales
4. génération du CSV final

---

# Arguments du script

## `--model`

Chemin vers le modèle MediaPipe.

Exemple :

```
--model face_landmarker.task
```

---

## `--dry-run`

Mode test.

Par défaut :

```
true
```

Pour lancer l’étude complète :

```
--dry-run false
```

---

## `--show`

Affiche la vidéo annotée pendant l’analyse.

Informations affichées :

- EAR brut
- EAR normalisé
- nombre de clignements

⚠️ disponible uniquement en **dry run**

---

## `--csv`

Nom du fichier CSV de sortie.

Par défaut :

```
results.csv
```

Exemple :

```
--csv etude.csv
```

---

## `--min-closed-frames`

Durée minimale d’une fermeture pour valider un clignement.

Défaut :

```
2
```

---

## `--max-closed-frames`

Durée maximale d’une fermeture pour valider un clignement.

Défaut :

```
12
```

---

# Exemples d’utilisation

### Test simple

```
python betterBlink.py --model face_landmarker.task
```

### Test avec affichage

```
python betterBlink.py --model face_landmarker.task --show
```

### Lancer l'étude complète

```
python betterBlink.py --model face_landmarker.task --dry-run false
```

### Export CSV personnalisé

```
python betterBlink.py --model face_landmarker.task --dry-run false --csv etude.csv
```

---

# Dépendances

Python ≥ 3.10

Installer :

```
pip install mediapipe opencv-python numpy
```

---

# Performances

Chaque vidéo est **lue une seule fois**.

Le coût principal est la détection des landmarks faciaux via MediaPipe.

Le pipeline est conçu pour **traiter de grandes cohortes de sujets** sans relecture inutile des vidéos.

# Calcul de l'erreur

Le script calcule en `real_run` trois métriques globales d'erreur à partir des **vidéos étalon disponibles** :

- **Erreur moyenne relative**
- **Biais moyen**
- **Écart type de l'erreur**

Ces valeurs sont écrites dans le fichier :

```text
Erreur relative.txt
```

Ce fichier est **remplacé à chaque exécution** du `real_run`.

---

## Ce que mesurent ces métriques

Pour chaque sujet disposant d'une vidéo étalon valide :

- `attendu` = nombre réel de clignements indiqué dans `attendu.txt`
- `prédit` = nombre de clignements trouvé par le script sur l'étalon après calibration

À partir de ces valeurs, le script calcule :

### Erreur moyenne relative

Elle mesure l'écart moyen entre le nombre attendu et le nombre prédit, **rapporté au nombre attendu**.

Forme utilisée :

```text
|prédit - attendu| / attendu
```

Puis la moyenne est calculée sur l'ensemble des étalons disponibles.

Interprétation :

- plus cette valeur est faible, plus le comptage sur les étalons est proche de la référence
- c'est une mesure globale de précision de comptage sur les vidéos étalon

---

### Biais moyen

Il mesure la tendance moyenne du modèle à surcompter ou sous compter dans les vidéos.

Forme utilisée :

```text
prédit - attendu
```

Interprétation :

- biais moyen **positif** : tendance à trouver trop de clignements
- biais moyen **négatif** : tendance à en manquer
- biais proche de zéro : pas de dérive moyenne forte

---

### Écart type de l'erreur

Il mesure la **dispersion des erreurs** entre sujets.

Interprétation :

- faible écart type : comportement relativement stable entre sujets
- fort écart type : comportement plus variable selon les vidéos étalon

---

## Peut-on généraliser ces résultats aux vidéos lors d'activité ?

Ces métriques donnent une **tendance générale du comportement du modèle sur les étalons analysés**.

Ce que l'on peut dire :

- elles décrivent le comportement sur les **vidéos étalon**
- elles donnent une idée du niveau d'erreur global observé sur les sujets disponibles au moment du run

Donc, il est possible de généraliser aux videos d'activités mais uniquement si celle ci sont dans les memes conditions
- même cadrage
- même résolution
- même luminosité
- même sujet
- même posture



---

## Interprétation recommandée

Les métriques du fichier `Erreur relative.txt` doivent être interprétées comme :

> une estimation globale de l'erreur du modèle sur les vidéos étalon disponibles pendant l'exécution

Elles sont utiles pour :

- suivre la qualité globale du pipeline
- vérifier si le modèle tend à sous-compter ou surcompter
- comparer plusieurs versions du script ou des réglages

---

## Limite importante

Ces métriques sont calculées uniquement à partir de ce qui est disponible dans le pipeline actuel :

- `attendu.txt`
- comptage prédit sur l'étalon
- agrégation globale de ces erreurs

Elles ne constituent donc **pas** une mesure directe de l'erreur sur les vidéos de catégorie, mais une **référence globale issue des étalons**.