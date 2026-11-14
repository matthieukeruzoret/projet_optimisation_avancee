"""
This script contains the functions to generate initial solutions respecting the constraints
and optimizing wind farm layouts using fast CMA-ES.
"""

import matplotlib
matplotlib.use('Agg')  # Évite les erreurs de thread Tkinter

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
    maxfevals=400,
    sigma0=15.0,
    penalty_weight=1e5,
):
    x0 = list(itertools.chain(*zip(x_init, y_init)))
    lb_full = lb * nb_wt
    ub_full = ub * nb_wt

    zone_geom = buildable_zone[0] if isinstance(buildable_zone, (list, np.ndarray)) else buildable_zone

    def fitness_function(x_vec):
        x_coords = x_vec[0::2]
        y_coords = x_vec[1::2]

        # Contraintes géométriques sans PyWake
        sum_dist = cst.placing_constraint(x_coords, y_coords, zone_geom)
        s_d = cst.spacing_constraint_min(x_coords, y_coords, D)

        if sum_dist > 0 or s_d > 0:
            return 1e6 + penalty_weight * (sum_dist + s_d)

        # Calcul PyWake si réalisable
        if fmGROSS is not None:
            try:
                _, aep = bb.aep_func(x_coords, y_coords, fmGROSS, WS_BB, WD_BB)
                aep_val = float(aep)
                aep_gwh = aep_val if aep_val < 1e5 else aep_val * 1e-6
                return -aep_gwh
            except Exception:
                return 1e6
        return 0.0

    print(f"--> Lancement CMA-ES ({nb_wt} éoliennes | budget: {maxfevals} evals)...")

    opts = {
        'bounds': [lb_full, ub_full],
        'maxfevals': maxfevals,
        'CMA_diagonal': True if nb_wt >= 30 else False,
        'verbose': -9,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    eval_count = 0
    best_aep = 0.0
    best_x_vec = list(x0)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [fitness_function(sol) for sol in solutions]
        es.tell(solutions, fitnesses)

        eval_count += len(solutions)

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
    """Génère une solution géométrique sans doublons dans la zone constructible."""
    zone_geom = buildable_zone[0] if isinstance(buildable_zone, (list, np.ndarray)) else buildable_zone
    factor = 5.0

    while factor >= 0.5:
        spacing = factor * D
        x_coords = np.arange(lb[0] + spacing / 2, ub[0], spacing)
        y_coords = np.arange(lb[1] + spacing / 2, ub[1], spacing)

        valid_points = []
        for x in x_coords:
            for y in y_coords:
                p = Point(x, y)
                if zone_geom.contains(p):
                    valid_points.append((x, y))

        if len(valid_points) >= nb_wt:
            chosen = random.sample(valid_points, nb_wt)
            return [p[0] for p in chosen], [p[1] for p in chosen]

        factor -= 0.1

    # Secours : génération aléatoire d'un ensemble de points uniques
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
    """Pipeline d'initialisation et affinage par CMA-ES."""
    fmGROSS, WS_BB, WD_BB = None, None, None
    try:
        _, D_file, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_path)
        fmGROSS, WS, WD, max_index, max_ws = wf.site_setting(power_curve, D_file, hub_height, wind_speed, wind_direction, "results_init")
        WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    except Exception as e:
        print(f"Avertissement : Impossible de charger PyWake ({e}).")

    x_init, y_init = initial_sol_grid(nb_wt, buildable_zone, D, lb, ub)

    x_opt, y_opt, _ = run_cma_es(
        nb_wt, buildable_zone, D, lb, ub, x_init, y_init, fmGROSS, WS_BB, WD_BB, maxfevals=400, sigma0=15.0
    )

    plt.close('all')
    return x_opt, y_opt