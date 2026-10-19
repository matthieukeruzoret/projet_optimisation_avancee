import itertools
import os
import shutil
import constraints as cst
import data as d
import initial_solution as init_sol
import windfarm_setting as wf

def generate_instances_varying_turbines(
    base_param_path="instances/1/param.txt",
    output_root="instances_varying",
    min_turbines=20,
    max_turbines=70,
    step=5,
):
    """Génère des dossiers d'instances pour différentes valeurs de nb_wt

    avec leur fichier param.txt et x0.txt admissible.
    """
    # 1. Lecture des métadonnées du fichier source
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

    # 2. Construction de la zone constructible et des bornes du domaine
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(
        boundary_file, exclusion_zone_file, scale_factor=scale_factor
    )
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

    # 3. Lecture du contenu brut pour mettre à jour la ligne DIMENSION
    with open(base_param_path, "r") as f:
        param_lines = f.read().splitlines()

    os.makedirs(output_root, exist_ok=True)

    # 4. Boucle sur le nombre de turbines
    for nb_wt in range(min_turbines, max_turbines + 1, step):
        target_dir = os.path.join(output_root, f"instance_{nb_wt}_wt")
        os.makedirs(target_dir, exist_ok=True)
        print(f"--- Génération de la solution initiale pour {nb_wt} éoliennes ---")

       # Génération via la méthode de la grille géométrique
        x, y = init_sol.initial_sol_grid(
            nb_wt, buildable_zone, diameter, lb, ub
        )

        # Vérification de faisabilité géométrique
        sum_dist = cst.placing_constraint(x, y, buildable_zone)
        s_d = cst.spacing_constraint_min(x, y, diameter)
        print(
            f"Admissibilité ({nb_wt} turbines) -> Hors zone : {sum_dist:.2f} m | Conflit espacement : {s_d:.2f} m"
        )

        # Construction du vecteur aplati [x1, y1, x2, y2, ...]
        x0 = list(itertools.chain(*zip(x, y)))

        # Écriture du fichier x0.txt
        x0_file_path = os.path.join(target_dir, "x0.txt")
        with open(x0_file_path, "w") as f:
            f.write(str(x0))

        # Mise à jour et écriture du param.txt (la ligne 0 contient le nombre d'éoliennes)
        new_param_lines = param_lines.copy()
        tag = param_lines[0].split()[0]
        new_param_lines[0] = f"{tag} {nb_wt}"

        param_file_path = os.path.join(target_dir, "param.txt")
        with open(param_file_path, "w") as f:
            f.write("\n".join(new_param_lines) + "\n")

    print(f"\nToutes les instances ont été générées dans : {output_root}")

if __name__ == "__main__":
    # Ajuste le paramètre step (ex. step=5 ou step=10) selon le nombre de cas voulus
    generate_instances_varying_turbines(
        base_param_path="instances/1/param.txt",
        output_root="instances_varying",
        min_turbines=20,
        max_turbines=70,
        step=5,
    )