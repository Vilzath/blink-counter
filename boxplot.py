import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# CONFIGURATION
# =========================

CSV_FILE = "plotbox.csv"
OUTPUT_FILE = "boxplots.png"

# Séparateur du CSV :
# - "\t" si ton fichier est tabulé
# - ";" si ton fichier est séparé par des points-virgules
# - "," si ton fichier est séparé par des virgules
SEPARATOR = "\t"

# Taille de l'image
FIG_WIDTH = 12
FIG_HEIGHT = 6

# Noms affichés pour chaque boîte.
# Si tu mets None, le script utilisera Série 1, Série 2, etc.
# Exemple :
BOX_LABELS = ["Etalon", "Sans Chrono", "Avec Chrono", "Coloriage", "P5 Ordi", "P5 Papier"]


# =========================
# LECTURE DU CSV
# =========================

csv_path = Path(CSV_FILE)

if not csv_path.exists():
    raise FileNotFoundError(f"Fichier introuvable : {CSV_FILE}")

df = pd.read_csv(
    csv_path,
    sep=SEPARATOR,
    header=None,
    decimal=","
)

# La première colonne contient les noms : Max, Min, Q1, Mediane, etc.
df = df.set_index(0)

# Nettoyage des noms de lignes
df.index = df.index.str.strip()

# Conversion des valeurs en nombres
df = df.apply(pd.to_numeric, errors="coerce")


# =========================
# VÉRIFICATION DES DONNÉES
# =========================

required_rows = ["Max", "Min", "Q1", "Mediane", "Q3", "Moyenne"]

missing_rows = [row for row in required_rows if row not in df.index]

if missing_rows:
    raise ValueError(
        "Lignes manquantes dans le CSV : "
        + ", ".join(missing_rows)
    )

number_of_boxes = df.shape[1]

if BOX_LABELS is None:
    BOX_LABELS = [f"Série {i + 1}" for i in range(number_of_boxes)]

if len(BOX_LABELS) != number_of_boxes:
    raise ValueError(
        f"BOX_LABELS contient {len(BOX_LABELS)} noms, "
        f"mais le CSV contient {number_of_boxes} séries."
    )


# =========================
# PRÉPARATION DES BOÎTES
# =========================

boxplot_stats = []

for i, column in enumerate(df.columns):
    stats = {
        "label": BOX_LABELS[i],
        "whislo": df.loc["Min", column],
        "q1": df.loc["Q1", column],
        "med": df.loc["Mediane", column],
        "q3": df.loc["Q3", column],
        "whishi": df.loc["Max", column],
        "mean": df.loc["Moyenne", column],
        "fliers": []
    }

    # Vérification de cohérence simple
    if not (
        stats["whislo"]
        <= stats["q1"]
        <= stats["med"]
        <= stats["q3"]
        <= stats["whishi"]
    ):
        print(
            f"Attention : valeurs incohérentes pour {BOX_LABELS[i]} : "
            f"Min <= Q1 <= Mediane <= Q3 <= Max n'est pas respecté."
        )

    boxplot_stats.append(stats)


# =========================
# CRÉATION DU GRAPHIQUE
# =========================

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

bp = ax.bxp(
    boxplot_stats,
    vert=False,
    showmeans=True,
    patch_artist=True,
    widths=0.6,
    meanprops={
        "marker": "o",
        "markerfacecolor": "yellow",
        "markeredgecolor": "black",
        "markersize": 6
    },
    boxprops={
        "facecolor": "#6fc17d",
        "edgecolor": "#1f7a3a",
        "linewidth": 1.5,
        "alpha": 0.85
    },
    medianprops={
        "color": "#1f7a3a",
        "linewidth": 1.8
    },
    whiskerprops={
        "color": "#1f7a3a",
        "linewidth": 1.4
    },
    capprops={
        "color": "#1f7a3a",
        "linewidth": 1.4
    }
)

# Grille verticale comme sur ton exemple
ax.grid(axis="x", linestyle="-", alpha=0.25)
ax.set_axisbelow(True)

# Titre et axes
ax.set_title("Boîtes à moustaches comparatives des ecarts entre les clignements", fontsize=14)
ax.set_xlabel("Valeur")
ax.set_ylabel("Séries")

# Échelle commune basée sur les min/max de toutes les séries
global_min = df.loc["Min"].min()
global_max = df.loc["Max"].max()
margin = (global_max - global_min) * 0.05

ax.set_xlim(global_min - margin, global_max + margin)

# Ligne esthétique
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()


# =========================
# AFFICHAGE ET SAUVEGARDE
# =========================

plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Graphique sauvegardé dans : {OUTPUT_FILE}")