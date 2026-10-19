"""

This script contains the functions to generate initial solutions respecting the constraints.

This script requires multiple libraries that are written in the `requirements.txt` to be installed in your Python environnement. 
Make sure to install them properly with the right version.

"""

import itertools
import sys
import numpy as np
import windfarm_setting as wf
import constraints as cst
import data as d
import PyNomad
import plotting_functions as plot_f
import random
from shapely.geometry import Point

def initial_sol_nomad(nb_wt, buildable_zone, D, lb, ub, spacing=True):
    """Script to compute the eap and constraint values with the NOMAD solver."""

    def initial_sol(x):
        """Script to evaluate the initial solution constraints."""
        try:
            # Get coords
            dim = x.size()
            x_list = [x.get_coord(i) for i in range(dim)]
            x_coords = x_list[0::2]
            y_coords = x_list[1::2]

            # Calculate constraints
            sum_dist = cst.placing_constraint(x_coords, y_coords, buildable_zone)

            if spacing:
                s_d = cst.spacing_constraint_min(x_coords, y_coords, D)
                rawBBO = f"0 {s_d} {sum_dist}"
                x.setBBO(rawBBO.encode("UTF-8"))
            else:
                rawBBO = f"0 {sum_dist}"
                x.setBBO(rawBBO.encode("UTF-8"))

        except Exception as e:
            print("Unexpected eval error:", e)
            return 0
        return 1

    # Augmentation du budget et ajustement des paramètres NOMAD
    params = [
        "BB_INPUT_TYPE * R",
        "STOP_IF_FEASIBLE true",
        "MAX_BB_EVAL 5000",  # Budget suffisant pour les grands N_T
        "DISPLAY_DEGREE 0",
        "DISPLAY_ALL_EVAL false",
        "DIRECTION_TYPE ORTHO 2N"
    ]

    if spacing:
        params += ["BB_OUTPUT_TYPE OBJ PB EB"]
    else:
        params += ["BB_OUTPUT_TYPE OBJ PB"]

    # Création des groupes de variables pour les coordonnées (x,y)
    for i in range(0, 2 * nb_wt, 2):
        params += [f"VARIABLE_GROUP {i} {i+1}"]

    # Tirage initial dans les bornes
    x_init = np.random.uniform(lb[0], ub[0], nb_wt)
    y_init = np.random.uniform(lb[1], ub[1], nb_wt)
    X0 = list(itertools.chain(*zip(x_init, y_init)))

    # Bornes du domaine pour NOMAD
    lb_full = lb * nb_wt
    ub_full = ub * nb_wt

    ## NOMAD optimization 
    result = PyNomad.optimize(initial_sol, X0, lb_full, ub_full, params)
    
    # Récupération sécurisée des coordonnées
    if isinstance(result, dict) and 'x_best' in result and len(result['x_best']) > 0:
        x_best = result['x_best'][0::2]
        y_best = result['x_best'][1::2]
    else:
        print(f"⚠️ Avertissement : NOMAD n'a pas trouvé de point faisable pour {nb_wt} turbines.")
        print("Conservation de X0 initial irréalisable en attendant une meilleure méthode d'initialisation.")
        x_best = x_init.tolist()
        y_best = y_init.tolist()

    return x_best, y_best


def initial_sol_test(param_file_name, x0_file_name):
    """Script to compute the eap and constraint values with the NOMAD solver.

        Parameters
        ----------
        param_file_name : str
            Path to the parameter text file.
        x0_file_name : str
            Path where the function will create the x0.txt with the coordinates of the initial solution.
        
        Returns
        -------
        x0_file_name/x0.txt : text file
            A text file with the coordinates of the initial solution.
    """
    
    # Initializing site and boundary files
    nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_name)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)
    
    x, y = initial_sol_nomad(nb_wt, buildable_zone, D, lb, ub, spacing=True)
    
    sum_dist = cst.placing_constraint(x, y, buildable_zone)
    s_d = cst.spacing_constraint_min(x, y, D)
    
    print("Coordonnées uniques :", cst.checking_same_coords(x, y))
    print("Placing constraint (hors zone) :", sum_dist, "| Spacing constraint :", s_d)
    
    X0 = list(itertools.chain(*zip(x, y)))
    x0_file_path = x0_file_name + "/x0.txt"
    
    with open(x0_file_path, 'w+') as f:
        f.write(str(X0))

def initial_sol_grid(nb_wt, buildable_zone, D, lb, ub):
    """
    Génère une solution géométrique admissible (placement et espacement)
    en échantillonnant une grille, sans faire appel à un optimiseur.
    """
    # Extraction de la géométrie Shapely unique si c'est une liste ou un ndarray
    if isinstance(buildable_zone, (list, np.ndarray)):
        zone_geom = buildable_zone[0]
    else:
        zone_geom = buildable_zone

    factor = 5.0  # On commence avec un espacement large (5 * D)
    
    while factor >= 2.0:  # 2.0 correspond à la contrainte stricte (2D)
        spacing = factor * D
        
        # Création de la grille de points dans les bornes du terrain
        x_coords = np.arange(lb[0] + spacing/2, ub[0], spacing)
        y_coords = np.arange(lb[1] + spacing/2, ub[1], spacing)
        
        valid_points = []
        for x in x_coords:
            for y in y_coords:
                p = Point(x, y)
                # Vérification de l'appartenance à la zone constructible
                if zone_geom.contains(p):
                    valid_points.append((x, y))
        
        # Si la grille génère assez de points valides
        if len(valid_points) >= nb_wt:
            chosen = random.sample(valid_points, nb_wt)
            x_best = [p[0] for p in chosen]
            y_best = [p[1] for p in chosen]
            return x_best, y_best
            
        # Si le nombre de points est insuffisant, on resserre la grille
        factor -= 0.1
        
    print(f"Erreur : Impossible de placer {nb_wt} turbines avec un espacement >= 2D.")
    # Sécurité au cas où le terrain est trop petit
    chosen = valid_points + random.choices(valid_points, k=max(0, nb_wt - len(valid_points)))
    return [p[0] for p in chosen], [p[1] for p in chosen]