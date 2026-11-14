# Optimisation du Placement d'Éoliennes

Ce projet est issu du cours **Optimisation Avancée** et s'appuie sur le framework **AMON** (Wind Farm Layout Optimization Problem), un environnement de benchmarking et d'optimisation boîte noire basé sur le modèle de sillage PyWake. L'objectif est de déterminer les coordonnées $(x_i, y_i)$ optimales d'un parc de $N$ éoliennes afin de maximiser la production énergétique annuelle (AEP en GWh) en minimisant les pertes aérodynamiques dues aux effets de sillage, tout en respectant strictement les contraintes géométriques du site (inclusion dans la zone constructible, contournement des zones d'exclusion et maintien d'une distance minimale de sécurité entre turbines). Pour relever ce défi d'optimisation non linéaire et coûteuse en évaluations, ce dépôt permet de générer des solutions initiales admissibles sur grille adaptative et de les optimiser grâce à l'algorithme **CMA-ES**.

## Installation

Avant d'exécuter les scripts, créez un environnement Conda avec **Python 3.9+** (ou votre version compatible) :

```bash
conda create -n windfarm_opt python=3.9.19
conda activate windfarm_opt
```

Installez ensuite les dépendances requises (incluant `py_wake`, `cma`, `geopandas`, `shapely`, `matplotlib`, etc.) :

```bash
pip install -r requirements.txt
```

> **Note :** Assurez-vous que la bibliothèque `cma` (`pip install cma`) est bien installée pour l'étape de pré-optimisation par stratégie d'évolution.

---

## Architecture et Nouvelles Fonctionnalités

En plus du cœur d'évaluation boîte noire d'origine, le dépôt intègre désormais un pipeline complet de génération d'instances, d'initialisation sous contraintes et de résolution :

- **`algo.py`** : Génère automatiquement des jeux d'instances (`instances_varying/`) en faisant varier le nombre d'éoliennes (`nb_wt`). Pour chaque configuration, le script :
  1. Crée un dossier `instance_X_wt/` et met à jour dynamiquement la dimension dans `param.txt`.
  2. Calcule une solution initiale admissible et pré-optimisée via `initial_solution.py`.
  3. Vérifie le respect strict des contraintes géométriques (inclusion dans la zone constructible et distance minimale entre turbines).
  4. Évalue la production énergétique annuelle (**AEP** en GWh) et sauvegarde automatiquement la carte du terrain avec les éoliennes placées (`terrain_X_wt.png`).
  5. Exporte le vecteur de coordonnées `[x_0, y_0, ..., x_n, y_n]` dans `x0.txt`.
- **`initial_solution.py`** : Implémente le pipeline d'initialisation en deux étapes :
  1. `initial_sol_grid` : Discrétise la zone constructible par une grille régulière adaptative (espacement initial de $5D$, réduit progressivement si nécessaire) afin de garantir une faisabilité géométrique immédiate.
  2. `run_cma_es` : Affine les coordonnées via l'algorithme évolutionnaire **CMA-ES** avec pénalisation rapide des contraintes hors PyWake pour maximiser l'AEP initiale.
- **`glouton.py` & `recuit.py`** : Heuristiques et métaheuristiques (algorithme glouton et recuit simulé) développées pour résoudre le problème de placement.
- **`plotting_functions.py`** : Outils de visualisation cartographique (zone constructible, zones d'exclusion hachurées, rose des vents / direction dominante, positions des turbines et cartes de sillage).

---

## Utilisation

### 1. Générer des instances et visualiser le placement (`algo.py`)

Pour générer une nouvelle instance (par exemple pour 50 éoliennes) et obtenir le rendu graphique du parc dans `instances_varying/instance_50_wt/terrain_50_wt.png` :

```bash
python algo.py
```

_(Pour ouvrir également la fenêtre graphique interactive à l'écran, commentez `matplotlib.use('Agg')` en tête de `algo.py` et `initial_solution.py` avant l'exécution)._

### 2. Validation de l'installation (NOMAD)

Pour vérifier le bon fonctionnement de l'environnement avec le solveur de référence :

```bash
python ./windfarm_nomad.py
```

Les résultats sont enregistrés dans `instances_results/` ainsi que dans le fichier récapitulatif `output_instances.txt`.

### 3. Évaluation directe de la fonction boîte noire (`windfarm_eval.py`)

Vous pouvez évaluer n'importe quelle configuration `X.txt` associée à un fichier `param.txt` directement en ligne de commande :

```bash
python ./windfarm_eval.py 'instances/1/param.txt' 'instances/1/x0.txt'
```

Ou depuis un script Python :

```python
import windfarm_eval

aep, spacing_constraint, placing_constraint = windfarm_eval.windfarm_eval(
    'instances/1/param.txt',
    'instances/1/x0.txt'
)
```

---

## Format des fichiers d'entrée

### Fichier de paramètres (`param.txt`)

```text
DIMENSION               Nombre d'éoliennes dans le parc
WT_DIAMETER             Diamètre du rotor (en mètres)
WT_HUB_HEIGHT           Hauteur du moyeu (en mètres)
SCALE_FACTOR            Facteur d'échelle du terrain (généralement 1)
POWER_CURVE             Chemin vers la courbe de puissance de l'éolienne
BOUNDARY_FILE           Chemin vers le shapefile des limites du terrain
EXCLUSION_ZONE_FILE     Chemin vers le shapefile des zones d'exclusion (na si aucune)
WIND_SPEED              Chemin vers les données CSV de vitesse du vent
WIND_DIRECTION          Chemin vers les données CSV de direction du vent
```

### Fichier de coordonnées (`x0.txt`)

Liste Python aplatie contenant les paires $(x_i, y_i)$ de chaque éolienne :

```text
[x_0, y_0, x_1, y_1, ..., x_n, y_n]
```
