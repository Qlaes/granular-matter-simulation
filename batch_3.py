import time

import numpy as np
import matplotlib.pyplot as plt



###############################################################################
# Batch 3 (FAST) — Extreme interaction cases (system size × grain-size distribution)
#
# Research question:
#   How do system size and grain-size distribution affect packing density
#   and stress distribution in a simulated granular medium?
#
# Lecture anchors (conceptual):
#   - Lecture 5: overdamped / highly dissipative relaxation:
#       x_{n+1} = x_n + (F / gamma) dt
#   - Lecture 1: neighbour acceleration (cell list idea)
#   - Lecture 12: contact structure exists, but Batch 3 plots focus on stress
#
# Output: EXACTLY 2 plots
#   (1) 2×2 panel: normalized stress-map magnitude (sigma_yy / <p>)
#   (2) Interaction plot: pressure proxy vs PSD for small vs large N
###############################################################################



###############################################################################
# 0) Grain-size distributions (extremes only)
###############################################################################


def generate_particle_radii_monodisperse(number_of_particles,
                                         base_radius):
    particle_radii = np.full(number_of_particles, float(base_radius), dtype=float)

    return particle_radii



def generate_particle_radii_polydisperse_uniform(number_of_particles,
                                                 base_radius,
                                                 random_number_generator,
                                                 minimum_radius_factor,
                                                 maximum_radius_factor):
    minimum_radius = float(minimum_radius_factor * base_radius)
    maximum_radius = float(maximum_radius_factor * base_radius)

    particle_radii = random_number_generator.uniform(minimum_radius,
                                                     maximum_radius,
                                                     size=number_of_particles).astype(float)

    return particle_radii



###############################################################################
# 1) Geometry + initialization
###############################################################################


def compute_box_size_from_target_packing_fraction(particle_radii,
                                                  target_packing_fraction,
                                                  box_aspect_ratio):
    total_particle_area = float(np.sum(np.pi * particle_radii ** 2))

    box_area = total_particle_area / float(target_packing_fraction)

    box_height = float(np.sqrt(box_area / float(box_aspect_ratio)))

    box_width = float(box_aspect_ratio * box_height)

    return box_width, box_height



def create_grid_initial_positions(number_of_particles,
                                  box_width,
                                  box_height,
                                  maximum_particle_radius,
                                  extra_spacing_factor):
    minimum_spacing = float(extra_spacing_factor * (2.0 * maximum_particle_radius))

    number_of_grid_points_x = int(np.floor(box_width / minimum_spacing))
    number_of_grid_points_y = int(np.floor(box_height / minimum_spacing))

    if number_of_grid_points_x * number_of_grid_points_y < number_of_particles:
        raise ValueError("Grid too small. Increase box or reduce N.")

    x_coordinates = np.linspace(maximum_particle_radius,
                                box_width - maximum_particle_radius,
                                number_of_grid_points_x)

    y_coordinates = np.linspace(maximum_particle_radius,
                                box_height - maximum_particle_radius,
                                number_of_grid_points_y)

    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)

    flattened_x = grid_x.flatten()[:number_of_particles]
    flattened_y = grid_y.flatten()[:number_of_particles]

    particle_positions = np.zeros((number_of_particles, 2), dtype=float)

    particle_positions[:, 0] = flattened_x
    particle_positions[:, 1] = flattened_y

    return particle_positions



def add_position_jitter(particle_positions,
                        particle_radii,
                        random_number_generator,
                        jitter_fraction_of_local_radius):
    jitter_scales = (jitter_fraction_of_local_radius * particle_radii).reshape(-1, 1)

    jitter = random_number_generator.uniform(-1.0, +1.0, size=particle_positions.shape) * jitter_scales

    particle_positions = particle_positions + jitter

    return particle_positions



###############################################################################
# 2) Cell list (neighbour acceleration)
###############################################################################


def build_cell_list(particle_positions,
                    box_width,
                    box_height,
                    cell_size):
    number_of_cells_x = int(np.floor(box_width / cell_size)) + 1
    number_of_cells_y = int(np.floor(box_height / cell_size)) + 1

    cell_x_indices = np.floor(particle_positions[:, 0] / cell_size).astype(int)
    cell_y_indices = np.floor(particle_positions[:, 1] / cell_size).astype(int)

    cell_x_indices = np.clip(cell_x_indices, 0, number_of_cells_x - 1)
    cell_y_indices = np.clip(cell_y_indices, 0, number_of_cells_y - 1)

    cells = {}

    for particle_index in range(particle_positions.shape[0]):

        cell_key = (int(cell_x_indices[particle_index]), int(cell_y_indices[particle_index]))

        if cell_key not in cells:
            cells[cell_key] = []

        cells[cell_key].append(int(particle_index))

    return cells, number_of_cells_x, number_of_cells_y



def generate_candidate_pairs_from_cells(cells,
                                       number_of_cells_x,
                                       number_of_cells_y):
    neighbour_offsets = [
        (-1, -1), (-1,  0), (-1, +1),
        ( 0, -1), ( 0,  0), ( 0, +1),
        (+1, -1), (+1,  0), (+1, +1),
    ]

    visited_cell_pairs = set()

    for (cell_x, cell_y), particle_indices in cells.items():

        for dx, dy in neighbour_offsets:

            neighbour_cell_x = cell_x + dx
            neighbour_cell_y = cell_y + dy

            if neighbour_cell_x < 0 or neighbour_cell_x >= number_of_cells_x:
                continue

            if neighbour_cell_y < 0 or neighbour_cell_y >= number_of_cells_y:
                continue

            neighbour_key = (neighbour_cell_x, neighbour_cell_y)

            if neighbour_key not in cells:
                continue

            ordered_cell_pair = tuple(sorted([(cell_x, cell_y), neighbour_key]))

            if ordered_cell_pair in visited_cell_pairs:
                continue

            visited_cell_pairs.add(ordered_cell_pair)

            neighbour_particle_indices = cells[neighbour_key]

            for i in particle_indices:
                for j in neighbour_particle_indices:

                    if j <= i:
                        continue

                    yield int(i), int(j)



###############################################################################
# 3) Forces + contacts (polydisperse-capable)
###############################################################################


def compute_forces_and_contacts_polydisperse(particle_positions,
                                            particle_radii,
                                            box_width,
                                            box_height,
                                            contact_spring_constant,
                                            wall_spring_constant,
                                            cutoff_radius,
                                            cached_cells,
                                            cached_number_of_cells_x,
                                            cached_number_of_cells_y):
    number_of_particles = particle_positions.shape[0]

    total_forces = np.zeros((number_of_particles, 2), dtype=float)

    contact_pairs = []
    contact_force_vectors = []
    contact_branch_vectors = []

    cutoff_radius_squared = float(cutoff_radius ** 2)



    for i, j in generate_candidate_pairs_from_cells(cached_cells,
                                                    cached_number_of_cells_x,
                                                    cached_number_of_cells_y):

        displacement_vector = particle_positions[j] - particle_positions[i]

        distance_squared = float(displacement_vector[0] ** 2 + displacement_vector[1] ** 2)

        if distance_squared > cutoff_radius_squared:
            continue

        center_distance = float(np.sqrt(distance_squared))

        if center_distance <= 1e-15:
            continue

        contact_distance = float(particle_radii[i] + particle_radii[j])

        overlap_distance = contact_distance - center_distance

        if overlap_distance <= 0.0:
            continue

        unit_normal_vector = displacement_vector / center_distance

        normal_force_magnitude = float(contact_spring_constant * overlap_distance)

        force_vector_on_j = normal_force_magnitude * unit_normal_vector

        total_forces[i] -= force_vector_on_j
        total_forces[j] += force_vector_on_j

        contact_pairs.append((int(i), int(j)))
        contact_force_vectors.append(force_vector_on_j)
        contact_branch_vectors.append(displacement_vector)



    # Soft wall forces (each particle uses its own radius)
    x = particle_positions[:, 0]
    y = particle_positions[:, 1]

    overlap_left = particle_radii - x
    overlap_right = x + particle_radii - box_width

    overlap_bottom = particle_radii - y
    overlap_top = y + particle_radii - box_height

    mask_left = overlap_left > 0.0
    mask_right = overlap_right > 0.0

    mask_bottom = overlap_bottom > 0.0
    mask_top = overlap_top > 0.0

    total_forces[mask_left, 0] += wall_spring_constant * overlap_left[mask_left]
    total_forces[mask_right, 0] -= wall_spring_constant * overlap_right[mask_right]

    total_forces[mask_bottom, 1] += wall_spring_constant * overlap_bottom[mask_bottom]
    total_forces[mask_top, 1] -= wall_spring_constant * overlap_top[mask_top]



    contact_force_vectors = np.array(contact_force_vectors, dtype=float)
    contact_branch_vectors = np.array(contact_branch_vectors, dtype=float)

    return total_forces, contact_pairs, contact_force_vectors, contact_branch_vectors



###############################################################################
# 4) Overdamped relaxation step (fast)
###############################################################################


def overdamped_relaxation_step(particle_positions,
                               total_forces,
                               drag_coefficient_gamma,
                               time_step_size):
    drift_velocities = total_forces / float(drag_coefficient_gamma)

    new_positions = particle_positions + drift_velocities * float(time_step_size)

    return new_positions, drift_velocities



###############################################################################
# 5) Stress measurement (global + coarse-grained map)
###############################################################################


def compute_global_stress_tensor(contact_force_vectors,
                                 contact_branch_vectors,
                                 box_width,
                                 box_height):
    if contact_force_vectors.size == 0:
        return 0.0, 0.0

    box_area = float(box_width * box_height)

    r_x = contact_branch_vectors[:, 0]
    r_y = contact_branch_vectors[:, 1]

    f_x = contact_force_vectors[:, 0]
    f_y = contact_force_vectors[:, 1]

    sigma_xx = float(np.sum(r_x * f_x) / box_area)
    sigma_yy = float(np.sum(r_y * f_y) / box_area)

    return sigma_xx, sigma_yy



def coarse_grain_sigma_yy_map_magnitude(contact_force_vectors,
                                        contact_branch_vectors,
                                        particle_positions,
                                        contact_pairs,
                                        box_width,
                                        box_height,
                                        number_of_bins_x,
                                        number_of_bins_y):
    """
    For poster-quality visualization, we accumulate the *magnitude* of the local
    vertical stress contribution: |r_y * f_y| per contact into bins.

    This avoids near-cancellation when positive/negative contributions appear
    locally (common in contact networks), which can otherwise make maps look blank.
    """
    sigma_yy_map = np.zeros((number_of_bins_y, number_of_bins_x), dtype=float)

    if contact_force_vectors.size == 0:
        return sigma_yy_map

    bin_width_x = float(box_width / number_of_bins_x)
    bin_width_y = float(box_height / number_of_bins_y)

    bin_area = float(bin_width_x * bin_width_y)

    for contact_index, (i, j) in enumerate(contact_pairs):

        midpoint = 0.5 * (particle_positions[i] + particle_positions[j])

        bin_x = int(np.floor(midpoint[0] / bin_width_x))
        bin_y = int(np.floor(midpoint[1] / bin_width_y))

        bin_x = int(np.clip(bin_x, 0, number_of_bins_x - 1))
        bin_y = int(np.clip(bin_y, 0, number_of_bins_y - 1))

        r_y = float(contact_branch_vectors[contact_index, 1])
        f_y = float(contact_force_vectors[contact_index, 1])

        sigma_yy_map[bin_y, bin_x] += abs(r_y * f_y) / bin_area

    return sigma_yy_map



###############################################################################
# 6) One extreme case: fast compress + relax
###############################################################################


def run_one_extreme_case_fast(number_of_particles,
                              particle_radii,
                              initial_packing_fraction,
                              box_aspect_ratio,
                              contact_spring_constant,
                              wall_spring_constant,
                              neighbour_cutoff_factor,
                              drag_coefficient_gamma,
                              overdamped_time_step_size,
                              compression_strain_per_stage,
                              max_number_of_compression_stages,
                              max_relaxation_steps_per_stage,
                              drift_speed_tolerance,
                              pressure_threshold,
                              minimum_mean_degree_for_jam_proxy,
                              cell_list_rebuild_interval,
                              max_displacement_fraction_of_cell,
                              number_of_bins_for_stress_map,
                              random_seed,
                              verbose):
    rng = np.random.default_rng(random_seed)

    particle_radii = np.asarray(particle_radii, dtype=float)

    maximum_particle_radius = float(np.max(particle_radii))



    box_width, box_height = compute_box_size_from_target_packing_fraction(
        particle_radii=particle_radii,
        target_packing_fraction=initial_packing_fraction,
        box_aspect_ratio=box_aspect_ratio
    )

    particle_positions = create_grid_initial_positions(
        number_of_particles=number_of_particles,
        box_width=box_width,
        box_height=box_height,
        maximum_particle_radius=maximum_particle_radius,
        extra_spacing_factor=1.05
    )

    particle_positions = add_position_jitter(
        particle_positions=particle_positions,
        particle_radii=particle_radii,
        random_number_generator=rng,
        jitter_fraction_of_local_radius=0.25
    )



    cutoff_radius = float(neighbour_cutoff_factor * (2.0 * maximum_particle_radius))

    cell_size = float(cutoff_radius)

    max_displacement_before_rebuild = float(max_displacement_fraction_of_cell * cell_size)



    latest_contact_pairs = []
    latest_contact_forces = np.zeros((0, 2), dtype=float)
    latest_contact_branches = np.zeros((0, 2), dtype=float)

    latest_pressure_proxy = 0.0



    for compression_stage in range(max_number_of_compression_stages):

        scale_factor = float(1.0 - compression_strain_per_stage)

        box_width *= scale_factor
        box_height *= scale_factor

        particle_positions *= scale_factor



        cached_cells, cached_nx, cached_ny = build_cell_list(
            particle_positions=particle_positions,
            box_width=box_width,
            box_height=box_height,
            cell_size=cell_size
        )

        reference_positions_for_cell_list = particle_positions.copy()

        last_drift_rms = 1e9



        for relaxation_step in range(max_relaxation_steps_per_stage):

            rebuild_due_to_interval = (relaxation_step % cell_list_rebuild_interval == 0)

            max_displacement = float(np.max(np.sqrt(np.sum((particle_positions - reference_positions_for_cell_list) ** 2,
                                                           axis=1))))

            rebuild_due_to_motion = (max_displacement > max_displacement_before_rebuild)

            if rebuild_due_to_interval or rebuild_due_to_motion:

                cached_cells, cached_nx, cached_ny = build_cell_list(
                    particle_positions=particle_positions,
                    box_width=box_width,
                    box_height=box_height,
                    cell_size=cell_size
                )

                reference_positions_for_cell_list = particle_positions.copy()



            total_forces, contact_pairs, contact_force_vectors, contact_branch_vectors = compute_forces_and_contacts_polydisperse(
                particle_positions=particle_positions,
                particle_radii=particle_radii,
                box_width=box_width,
                box_height=box_height,
                contact_spring_constant=contact_spring_constant,
                wall_spring_constant=wall_spring_constant,
                cutoff_radius=cutoff_radius,
                cached_cells=cached_cells,
                cached_number_of_cells_x=cached_nx,
                cached_number_of_cells_y=cached_ny
            )



            # IMPORTANT FIX:
            # Always store the most recent contact state, even if we don't "jam" immediately.
            latest_contact_pairs = contact_pairs
            latest_contact_forces = contact_force_vectors
            latest_contact_branches = contact_branch_vectors



            particle_positions, drift_velocities = overdamped_relaxation_step(
                particle_positions=particle_positions,
                total_forces=total_forces,
                drag_coefficient_gamma=drag_coefficient_gamma,
                time_step_size=overdamped_time_step_size
            )



            particle_positions[:, 0] = np.clip(particle_positions[:, 0],
                                               0.0 + particle_radii,
                                               box_width - particle_radii)

            particle_positions[:, 1] = np.clip(particle_positions[:, 1],
                                               0.0 + particle_radii,
                                               box_height - particle_radii)



            last_drift_rms = float(np.sqrt(np.mean(np.sum(drift_velocities ** 2, axis=1))))

            sigma_xx, sigma_yy = compute_global_stress_tensor(
                contact_force_vectors=contact_force_vectors,
                contact_branch_vectors=contact_branch_vectors,
                box_width=box_width,
                box_height=box_height
            )

            latest_pressure_proxy = float(0.5 * (sigma_xx + sigma_yy))

            mean_degree_proxy = 2.0 * len(contact_pairs) / float(number_of_particles)



            settled = (last_drift_rms < drift_speed_tolerance)

            jammed = (latest_pressure_proxy > pressure_threshold
                      and mean_degree_proxy > minimum_mean_degree_for_jam_proxy
                      and settled)

            if settled:
                # Fast exit if already settled at this stage; continuing relax steps is wasted.
                if jammed:
                    break

            if jammed:
                break



        if verbose and (compression_stage % 10 == 0 or compression_stage < 4):
            print(f"  stage={compression_stage:3d}   "
                  f"p~{latest_pressure_proxy: .3e}   "
                  f"v_rms={last_drift_rms: .3e}   "
                  f"contacts={len(latest_contact_pairs)}")

        if latest_pressure_proxy > pressure_threshold and last_drift_rms < drift_speed_tolerance:
            break



    sigma_yy_map = coarse_grain_sigma_yy_map_magnitude(
        contact_force_vectors=latest_contact_forces,
        contact_branch_vectors=latest_contact_branches,
        particle_positions=particle_positions,
        contact_pairs=latest_contact_pairs,
        box_width=box_width,
        box_height=box_height,
        number_of_bins_x=number_of_bins_for_stress_map,
        number_of_bins_y=number_of_bins_for_stress_map
    )

    if latest_pressure_proxy > 0.0:
        sigma_yy_map_normalized = sigma_yy_map / latest_pressure_proxy
    else:
        sigma_yy_map_normalized = sigma_yy_map



    results = {
        "pressure_proxy": float(latest_pressure_proxy),
        "sigma_yy_map_normalized": sigma_yy_map_normalized,
    }

    return results



###############################################################################
# 7) Plot helpers (human look)
###############################################################################


def apply_human_axis_style():
    current_axes = plt.gca()

    current_axes.spines["top"].set_visible(False)
    current_axes.spines["right"].set_visible(False)

    current_axes.grid(True, which="major", alpha=0.25, linewidth=0.7)
    current_axes.minorticks_on()
    current_axes.grid(True, which="minor", alpha=0.10, linewidth=0.4)



###############################################################################
# 8) Main — run the 4 extreme cases and make 2 plots
###############################################################################


if __name__ == "__main__":

    # Extreme sizes (from your Batch 1)
    small_system_size_N = 200
    large_system_size_N = 3200



    # PSD extremes
    base_radius = 0.02

    polydisperse_min_radius_factor = 0.70
    polydisperse_max_radius_factor = 1.30



    # FAST protocol (compress harder + relax less)
    initial_packing_fraction = 0.25

    box_aspect_ratio = 1.0

    contact_spring_constant = 2.0e4
    wall_spring_constant = 2.0e4

    neighbour_cutoff_factor = 1.25

    # Faster overdamped evolution (Lecture 5 style)
    overdamped_time_step_size = 1.2e-3
    drag_coefficient_gamma = 120.0

    # Bigger compression per stage => fewer stages
    compression_strain_per_stage = 0.02
    max_number_of_compression_stages = 90

    # Fewer relax steps per stage
    max_relaxation_steps_per_stage = 90

    # Slightly looser "settled" tolerance for speed
    drift_speed_tolerance = 2.5e-5

    # Jam thresholds (keep modest)
    pressure_threshold = 200.0
    minimum_mean_degree_for_jam_proxy = 3.0

    # Cell list caching
    cell_list_rebuild_interval = 12
    max_displacement_fraction_of_cell = 0.35

    # Fewer bins => faster maps
    number_of_bins_for_stress_map = 20



    cases_to_run = [
        {"label": "Small N, monodisperse", "N": small_system_size_N, "psd": "monodisperse"},
        {"label": "Small N, polydisperse", "N": small_system_size_N, "psd": "polydisperse_uniform"},
        {"label": "Large N, monodisperse", "N": large_system_size_N, "psd": "monodisperse"},
        {"label": "Large N, polydisperse", "N": large_system_size_N, "psd": "polydisperse_uniform"},
    ]



    results_by_case_label = {}

    overall_start_time = time.time()



    for case_index, case in enumerate(cases_to_run):

        case_label = case["label"]
        number_of_particles = int(case["N"])

        random_seed = 10_000 + 77 * case_index + number_of_particles

        random_number_generator = np.random.default_rng(random_seed)



        if case["psd"] == "monodisperse":

            particle_radii = generate_particle_radii_monodisperse(
                number_of_particles=number_of_particles,
                base_radius=base_radius
            )

        else:

            particle_radii = generate_particle_radii_polydisperse_uniform(
                number_of_particles=number_of_particles,
                base_radius=base_radius,
                random_number_generator=random_number_generator,
                minimum_radius_factor=polydisperse_min_radius_factor,
                maximum_radius_factor=polydisperse_max_radius_factor
            )



        print(f"\nRunning case: {case_label}")
        print(f"  N={number_of_particles}")
        print(f"  radii: min={np.min(particle_radii):.4f}  mean={np.mean(particle_radii):.4f}  max={np.max(particle_radii):.4f}")

        start_time = time.time()

        results = run_one_extreme_case_fast(
            number_of_particles=number_of_particles,
            particle_radii=particle_radii,
            initial_packing_fraction=initial_packing_fraction,
            box_aspect_ratio=box_aspect_ratio,
            contact_spring_constant=contact_spring_constant,
            wall_spring_constant=wall_spring_constant,
            neighbour_cutoff_factor=neighbour_cutoff_factor,
            drag_coefficient_gamma=drag_coefficient_gamma,
            overdamped_time_step_size=overdamped_time_step_size,
            compression_strain_per_stage=compression_strain_per_stage,
            max_number_of_compression_stages=max_number_of_compression_stages,
            max_relaxation_steps_per_stage=max_relaxation_steps_per_stage,
            drift_speed_tolerance=drift_speed_tolerance,
            pressure_threshold=pressure_threshold,
            minimum_mean_degree_for_jam_proxy=minimum_mean_degree_for_jam_proxy,
            cell_list_rebuild_interval=cell_list_rebuild_interval,
            max_displacement_fraction_of_cell=max_displacement_fraction_of_cell,
            number_of_bins_for_stress_map=number_of_bins_for_stress_map,
            random_seed=random_seed,
            verbose=False
        )

        elapsed = time.time() - start_time

        sigma_map = results["sigma_yy_map_normalized"]

        print(f"  pressure proxy p~ {results['pressure_proxy']:.3e}")
        print(f"  sigma-map: min={np.min(sigma_map):.3e}  max={np.max(sigma_map):.3e}")
        print(f"  time: {elapsed:.1f}s")

        results_by_case_label[case_label] = results



    print(f"\nTotal wall time: {time.time() - overall_start_time:.1f}s")



    ###############################################################################
    # Plot 1 (of 2): 2×2 panel of normalized σyy-map magnitudes for extremes
    ###############################################################################

    figure_1 = plt.figure(figsize=(10.2, 8.0))

    panel_order = [
        "Small N, monodisperse",
        "Small N, polydisperse",
        "Large N, monodisperse",
        "Large N, polydisperse",
    ]

    global_vmax = 0.0
    for label in panel_order:
        global_vmax = max(global_vmax, float(np.max(results_by_case_label[label]["sigma_yy_map_normalized"])))

    image_handles = []

    for subplot_index, label in enumerate(panel_order):

        sigma_yy_map_normalized = results_by_case_label[label]["sigma_yy_map_normalized"]

        plt.subplot(2, 2, subplot_index + 1)

        image_handle = plt.imshow(sigma_yy_map_normalized,
                                  origin="lower",
                                  aspect="auto",
                                  vmin=0.0,
                                  vmax=global_vmax)

        image_handles.append(image_handle)

        plt.title(label, fontsize=11)

        plt.xlabel("x-bin")
        plt.ylabel("y-bin")

        apply_human_axis_style()

    plt.suptitle(r"Batch 3 — Stress distribution extremes: normalized $|\sigma_{yy}| / \langle p \rangle$",
                 y=0.98)

    plt.tight_layout(rect=[0.0, 0.0, 0.92, 0.95])

    colorbar_axis = figure_1.add_axes([0.93, 0.13, 0.02, 0.73])
    cbar = plt.colorbar(image_handles[-1], cax=colorbar_axis)
    cbar.set_label(r"$|\sigma_{yy}| / \langle p \rangle$")

    plt.show()


    ###############################################################################
    # Plot 2 (of 2): Interaction plot — pressure proxy vs PSD for small vs large N
    ###############################################################################

    small_mono_pressure = results_by_case_label["Small N, monodisperse"]["pressure_proxy"]
    small_poly_pressure = results_by_case_label["Small N, polydisperse"]["pressure_proxy"]

    large_mono_pressure = results_by_case_label["Large N, monodisperse"]["pressure_proxy"]
    large_poly_pressure = results_by_case_label["Large N, polydisperse"]["pressure_proxy"]



    psd_positions = np.array([0, 1], dtype=float)
    psd_labels = ["monodisperse", "polydisperse"]

    plt.figure(figsize=(7.8, 4.8))

    plt.plot(psd_positions,
             [small_mono_pressure, small_poly_pressure],
             marker="o",
             linewidth=1.6,
             alpha=0.9,
             label=f"Small N = {small_system_size_N}")

    plt.plot(psd_positions,
             [large_mono_pressure, large_poly_pressure],
             marker="s",
             linewidth=1.6,
             alpha=0.9,
             label=f"Large N = {large_system_size_N}")

    plt.xticks(psd_positions, psd_labels, rotation=10)

    plt.ylabel("pressure proxy (a.u.)")

    plt.title("Batch 3 — Interaction plot: PSD effect depends on system size?")

    plt.legend(frameon=False)

    apply_human_axis_style()

    plt.tight_layout()
    plt.show()
