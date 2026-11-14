import matplotlib 

import warnings
warnings.filterwarnings("ignore")

import itertools
import os
import matplotlib.pyplot as plt
import blackbox as bb
import constraints as cst
import data as d
import initial_solution as init_sol
import plotting_functions as pf
import windfarm_setting as wf


def generate_instances_varying_turbines(
    base_param_path="instances/1/param.txt",
    output_root="instances_varying",
    min_turbines=20,
    max_turbines=70,
    step=5,
):
    """
    Génère un jeu d'instances d'optimisation de parc éolien pour différents nombres d'éoliennes.

    Pour chaque valeur de `nb_wt` entre `min_turbines` et `max_turbines` (par pas de `step`) :
    1. Crée un dossier dédié et met à jour le fichier `param.txt` avec la nouvelle dimension.
    2. Calcule une solution initiale admissible et pré-optimisée via Grille + CMA-ES.
    3. Vérifie le respect des contraintes géométriques (zone constructible et distance minimale).
    4. Évalue la production énergétique annuelle (AEP) et génère la carte du terrain (`terrain_X_wt.png`).
    5. Sauvegarde le vecteur de coordonnées initiales dans `x0.txt`.

    Paramètres
    ----------
    base_param_path : str
        Chemin vers le fichier `param.txt` de référence contenant les caractéristiques du site.
    output_root : str
        Dossier racine où seront créés les sous-dossiers d'instances.
    min_turbines : int
        Nombre minimal d'éoliennes à générer.
    max_turbines : int
        Nombre maximal d'éoliennes à générer.
    step : int
        Pas d'incrémentation du nombre d'éoliennes entre deux instances.
    """
    # 1. Extraction des paramètres physiques et géographiques de l'instance de base
    (
        orig_nb_wt,
        diameter,
        hub_height,
        scale_factor,
        power_curve,
        boundary_file,
        exclusion_zone_file,
        wind_speed,
        wind_direction,
    ) = d.read_param_file(base_param_path)

    # 2. Modélisation géométrique du terrain (bornes, frontières extérieures et zones interdites)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(
        boundary_file, exclusion_zone_file, scale_factor=scale_factor
    )
    # Soustraction des zones d'exclusion à la frontière globale pour obtenir le polygone constructible
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

    # 3. Initialisation du modèle aérodynamique (PyWake) et chargement de la rose des vents
    fmGROSS, WS, WD, max_index, max_ws = wf.site_setting(
        power_curve, diameter, hub_height, wind_speed, wind_direction, "results_init"
    )
    WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)

    # 4. Lecture du contenu brut de param.txt afin de pouvoir modifier dynamiquement la dimension
    with open(base_param_path, "r") as f:
        param_lines = f.read().splitlines()

    os.makedirs(output_root, exist_ok=True)

    # 5. Génération séquentielle des instances pour chaque taille de parc
    for nb_wt in range(min_turbines, max_turbines + 1, step):
        target_dir = os.path.join(output_root, f"instance_{nb_wt}_wt")
        os.makedirs(target_dir, exist_ok=True)

        # Mise à jour de la première ligne du fichier param.txt (ex: "DIMENSION 50")
        new_param_lines = param_lines.copy()
        tag = param_lines[0].split()[0]
        new_param_lines[0] = f"{tag} {nb_wt}"

        param_file_path = os.path.join(target_dir, "param.txt")
        with open(param_file_path, "w") as f:
            f.write("\n".join(new_param_lines) + "\n")

        # Génération d'un placement initial respectant les contraintes et affiné par CMA-ES
        x, y = init_sol.initial_sol_nomad(
            nb_wt,
            buildable_zone,
            diameter,
            lb,
            ub,
            spacing=True,
            param_file_path=param_file_path,
        )

        # Contrôle de la faisabilité géométrique de la solution obtenue (0.0 m = aucune violation)
        sum_dist = cst.placing_constraint(x, y, buildable_zone)
        s_d = cst.spacing_constraint_min(x, y, diameter)
        print(
            f"Admissibilité ({nb_wt} turbines) -> Hors zone : {sum_dist:.2f} m | Conflit espacement : {s_d:.2f} m"
        )

        # Évaluation de l'AEP (Annual Energy Production) et conversion systématique en GWh
        _, aep = bb.aep_func(x, y, fmGROSS, WS_BB, WD_BB)
        aep_val = float(aep)
        aep_gwh = aep_val if aep_val < 1e5 else aep_val * 1e-6

        # Génération et sauvegarde du graphique du terrain avec les éoliennes et le vent dominant
        plot_path = os.path.join(target_dir, f"terrain_{nb_wt}_wt.png")
        pf.plot_terrain(
            x=x,
            y=y,
            obj_function="AEP",
            units="GWh",
            obj_function_value=aep_gwh,
            n_wt=nb_wt,
            ub=ub,
            boundary_shapely=boundary_shapely,
            exclusion_zones_shapely=exclusion_zones_shapely,
            max_index=max_index,
            max_ws=max_ws,
            save=True,
            save_name=plot_path,
        )
        plt.show()
        plt.close('all')

        # Formatage des coordonnées en liste plate [x1, y1, x2, y2, ...] attendue par les solveurs
        x0 = list(itertools.chain(*zip(x, y)))
        x0_file_path = os.path.join(target_dir, "x0.txt")
        with open(x0_file_path, "w") as f:
            f.write(str(x0))

    print(f"\nToutes les instances ont été générées avec succès dans : {output_root}")


if __name__ == "__main__":
    # Exécution sur un cas test de 50 éoliennes
    generate_instances_varying_turbines(
        base_param_path="instances/1/param.txt",
        output_root="instances_varying",
        min_turbines=50,
        max_turbines=50,
        step=1,
    )