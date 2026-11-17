"""
Ce module regroupe les fonctions permettant de générer une solution initiale admissible
sur une grille adaptative, puis de l'affiner par stratégie d'évolution (CMA-ES) afin
de maximiser la production énergétique annuelle (AEP) tout en respectant les contraintes.
"""

import matplotlib
matplotlib.use('Agg')  # Désactive l'interface graphique Tkinter pour éviter les blocages en exécution batch

import warnings
warnings.filterwarnings("ignore")

import itertools
import random
import sys
import numpy as np
import shapely.geometry as sg
from shapely.geometry import Point
import matplotlib.pyplot as plt
import cma

import blackbox as bb
import constraints as cst
import data as d
import windfarm_setting as wf


def run_cma_es(
    nb_wt,
    buildable_zone,
    D,
    lb,
    ub,
    x_init,
    y_init,
    fmGROSS,
    WS_BB,
    WD_BB,
    maxfevals=50,
    sigma0=15.0,
    penalty_weight=1e5,
):
    """
    Optimise les positions (x, y) des éoliennes à l'aide de l'algorithme CMA-ES
    (Covariance Matrix Adaptation Evolution Strategy).

    Les contraintes géométriques sont gérées par pénalisation exacte avant même d'appeler
    le simulateur PyWake, ce qui accélère fortement l'évaluation des solutions non admissibles.

    Paramètres
    ----------
    nb_wt : int
        Nombre d'éoliennes à placer.
    buildable_zone : shapely.Polygon ou list
        Polygone représentant la zone autorisée pour l'implantation.
    D : float
        Diamètre du rotor d'une éolienne (en mètres), utilisé pour la distance minimale inter-turbines.
    lb, ub : list
        Bornes inférieure et supérieure [x_min, y_min] et [x_max, y_max] du domaine.
    x_init, y_init : list
        Coordonnées initiales des éoliennes servant de point de départ à CMA-ES.
    fmGROSS : WindFarmModel
        Modèle de sillage PyWake configuré pour le site.
    WS_BB, WD_BB : array-like
        Données de vitesses et directions du vent issues des fichiers CSV.
    maxfevals : int
        Budget maximal d'évaluations de la fonction objectif.
    sigma0 : float
        Écart-type initial (en mètres) régissant l'amplitude d'exploration autour de `x_init`.
    penalty_weight : float
        Coefficient de pénalité appliqué aux violations de contraintes géométriques.

    Retourne
    --------
    tuple (opt_x, opt_y, best_aep)
        Listes des coordonnées optimisées et valeur de la meilleure AEP atteinte (en GWh).
    """
    # Mise à plat des coordonnées [x1, y1, x2, y2, ...] et duplication des bornes pour chaque turbine
    x0 = list(itertools.chain(*zip(x_init, y_init)))
    lb_full = lb * nb_wt
    ub_full = ub * nb_wt

    # Extraction de l'objet géométrique Shapely si encapsulé dans une liste
    zone_geom = buildable_zone[0] if isinstance(buildable_zone, (list, np.ndarray)) else buildable_zone

    def fitness_function(x_vec):
        """
        Fonction coût à minimiser par CMA-ES :
        - Retourne une forte pénalité (> 10^6) si au moins une éolienne sort de la zone ou est trop proche d'une autre.
        - Retourne -AEP (en GWh) si la configuration est géométriquement valide.
        """
        x_coords = x_vec[0::2]
        y_coords = x_vec[1::2]

        # 1. Vérification rapide des contraintes géométriques (sans simulation de sillage)
        sum_dist = cst.placing_constraint(x_coords, y_coords, zone_geom)
        s_d = cst.spacing_constraint_min(x_coords, y_coords, D)

        if sum_dist > 0 or s_d > 0:
            return 1e6 + penalty_weight * (sum_dist + s_d)

        # 2. Simulation aérodynamique via PyWake uniquement pour les solutions admissibles
        if fmGROSS is not None:
            try:
                _, aep = bb.aep_func(x_coords, y_coords, fmGROSS, WS_BB, WD_BB)
                aep_val = float(aep)
                aep_gwh = aep_val if aep_val < 1e5 else aep_val * 1e-6
                return -aep_gwh  # Signe négatif car CMA-ES minimise l'objectif
            except Exception:
                return 1e6
            
        return 0.0

    print(f"--> Lancement CMA-ES ({nb_wt} éoliennes | budget: {maxfevals} evals)...")

    # Configuration du solveur : matrice de covariance diagonale activée au-delà de 30 turbines
    # pour réduire la complexité en O(n^2) vers O(n)
    opts = {
        'bounds': [lb_full, ub_full],
        'maxfevals': maxfevals,
        'CMA_diagonal': True if nb_wt >= 30 else False,
        'verbose': -9,
        'seed':42,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    eval_count = 0
    best_aep = 0.0
    best_x_vec = list(x0)

    # Boucle évolutionnaire : génération d'une population, évaluation et mise à jour de la distribution
    while not es.stop():
        solutions = es.ask()
        fitnesses = [fitness_function(sol) for sol in solutions]
        es.tell(solutions, fitnesses)

        eval_count += len(solutions)

        # Mémorisation de la meilleure solution admissible rencontrée
        for sol, fit in zip(solutions, fitnesses):
            if fit < 0:
                aep_gwh = -fit
                if aep_gwh > best_aep:
                    best_aep = aep_gwh
                    best_x_vec = list(sol)
                    print(f"   [Eval {eval_count}] Nouvelle AEP : {best_aep:.4f} GWh")

    opt_x = list(best_x_vec[0::2])
    opt_y = list(best_x_vec[1::2])

    print(f"--> Fin CMA-ES | Meilleure AEP : {best_aep:.4f} GWh")
    plt.close('all')
    return opt_x, opt_y, best_aep


def initial_sol_grid(nb_wt, buildable_zone, D, lb, ub):
    """
    Génère un placement initial admissible en discrétisant le terrain par une grille régulière.

    L'espacement de la grille démarre à 5*D (pour limiter d'emblée les effets de sillage)
    et est progressivement réduit par pas de 0.1*D jusqu'à ce que suffisamment de nœuds
    se trouvent à l'intérieur de la zone constructible.

    Paramètres
    ----------
    nb_wt : int
        Nombre d'éoliennes à placer.
    buildable_zone : shapely.Polygon ou list
        Polygone représentant la zone constructible.
    D : float
        Diamètre du rotor (en mètres).
    lb, ub : list
        Bornes inférieure et supérieure du domaine.

    Retourne
    --------
    tuple (list, list)
        Listes des coordonnées `x` et `y` sélectionnées aléatoirement parmi les nœuds valides.
    """
    zone_geom = buildable_zone[0] if isinstance(buildable_zone, (list, np.ndarray)) else buildable_zone
    factor = 5.0

    # Recherche de la maille la plus large pouvant accueillir les `nb_wt` éoliennes
    while factor >= 0.5:
        spacing = factor * D
        x_coords = np.arange(lb[0] + spacing / 2, ub[0], spacing)
        y_coords = np.arange(lb[1] + spacing / 2, ub[1], spacing)

        # Filtrage des sommets de la grille situés strictement dans la zone constructible
        valid_points = []
        for x in x_coords:
            for y in y_coords:
                p = Point(x, y)
                if zone_geom.contains(p):
                    valid_points.append((x, y))

        # Tirage sans remise de `nb_wt` emplacements si la grille est assez dense
        if len(valid_points) >= nb_wt:
            chosen = random.sample(valid_points, nb_wt)
            return [p[0] for p in chosen], [p[1] for p in chosen]

        factor -= 0.1

    # Méthode de secours par tirage aléatoire uniforme si aucune grille régulière n'a suffi
    valid_points = []
    attempts = 0
    while len(valid_points) < nb_wt and attempts < 10000:
        rx = random.uniform(lb[0], ub[0])
        ry = random.uniform(lb[1], ub[1])
        p = Point(rx, ry)
        if zone_geom.contains(p) and (rx, ry) not in valid_points:
            valid_points.append((rx, ry))
        attempts += 1

    return [p[0] for p in valid_points], [p[1] for p in valid_points]


def initial_sol_nomad(nb_wt, buildable_zone, D, lb, ub, spacing=True, param_file_path="instances/1/param.txt"):
    """
    Pipeline complet de génération d'une solution initiale de haute qualité :
    1. Charge l'environnement aérodynamique du site (PyWake et données de vent).
    2. Construit une configuration géométrique admissible via `initial_sol_grid`.
    3. Affine les positions des turbines via un court run de `run_cma_es` (40 évaluations).

    Paramètres
    ----------
    nb_wt : int
        Nombre d'éoliennes à placer.
    buildable_zone : shapely.Polygon ou list
        Polygone de la zone constructible.
    D : float
        Diamètre du rotor (en mètres).
    lb, ub : list
        Bornes du domaine spatial.
    spacing : bool
        Paramètre conservé pour compatibilité d'interface avec l'appelant.
    param_file_path : str
        Chemin vers le fichier `param.txt` de l'instance à initialiser.

    Retourne
    --------
    tuple (list, list)
        Listes `x_opt` et `y_opt` des coordonnées prêtes à être écrites dans `x0.txt`.
    """
    fmGROSS, WS_BB, WD_BB = None, None, None
    try:
        # Chargement des paramètres spécifiques de l'instance pour l'évaluation AEP
        _, D_file, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_path)
        fmGROSS, WS, WD, max_index, max_ws = wf.site_setting(power_curve, D_file, hub_height, wind_speed, wind_direction, "results_init")
        WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    except Exception as e:
        print(f"Avertissement : Impossible de charger PyWake ({e}).")

    # Étape 1 : Placement géométrique initial sur grille
    x_init, y_init = initial_sol_grid(nb_wt, buildable_zone, D, lb, ub)
    
    # Étape 2 : Optimisation par CMA-ES pour maximiser l'AEP initiale
    x_opt, y_opt, _ = run_cma_es(
        nb_wt, buildable_zone, D, lb, ub, x_init, y_init, fmGROSS, WS_BB, WD_BB, maxfevals=40, sigma0=15.0
    )

    plt.close('all')
    return x_opt, y_opt
