import time

import numpy as np
import matplotlib.pyplot as plt



###############################################################################
# Batch 1 (FASTER) — System-size sweep (monodisperse)
###############################################################################


def compute_box_size_from_target_packing_fraction(number_of_particles,
                                                  particle_radius,
                                                  target_packing_fraction,
                                                  box_aspect_ratio):
    total_particle_area = number_of_particles * np.pi * particle_radius ** 2

    box_area = total_particle_area / target_packing_fraction

    box_height = np.sqrt(box_area / box_aspect_ratio)

    box_width = box_aspect_ratio * box_height

    return float(box_width), float(box_height)



def create_grid_initial_positions(number_of_particles,
                                  box_width,
                                  box_height,
                                  particle_radius,
                                  extra_spacing_factor):
    minimum_spacing = extra_spacing_factor * (2.0 * particle_radius)

    number_of_grid_points_x = int(np.floor(box_width / minimum_spacing))
    number_of_grid_points_y = int(np.floor(box_height / minimum_spacing))

    if number_of_grid_points_x * number_of_grid_points_y < number_of_particles:
        raise ValueError("Grid too small for requested N (increase box or reduce N).")

    x_coordinates = np.linspace(particle_radius,
                                box_width - particle_radius,
                                number_of_grid_points_x)

    y_coordinates = np.linspace(particle_radius,
                                box_height - particle_radius,
                                number_of_grid_points_y)

    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)

    flattened_x = grid_x.flatten()[:number_of_particles]
    flattened_y = grid_y.flatten()[:number_of_particles]

    particle_positions = np.zeros((number_of_particles, 2), dtype=float)

    particle_positions[:, 0] = flattened_x
    particle_positions[:, 1] = flattened_y

    return particle_positions



def add_small_position_jitter(particle_positions,
                              particle_radius,
                              random_number_generator,
                              jitter_fraction_of_radius):
    jitter_scale = jitter_fraction_of_radius * particle_radius

    particle_positions += random_number_generator.uniform(-jitter_scale,
                                                         +jitter_scale,
                                                         size=particle_positions.shape)

    return particle_positions



def compute_packing_fraction(number_of_particles,
                             particle_radius,
                             box_width,
                             box_height):
    total_particle_area = number_of_particles * np.pi * particle_radius ** 2

    box_area = box_width * box_height

    return float(total_particle_area / box_area)



###############################################################################
# 1) Cell list (cached / rebuilt occasionally)
###############################################################################


def build_cell_list(particle_positions,
                    box_width,
                    box_height,
                    cell_size):
    number_of_cells_x = int(np.floor(box_width / cell_size)) + 1
    number_of_cells_y = int(np.floor(box_height / cell_size)) + 1

    cell_x = np.floor(particle_positions[:, 0] / cell_size).astype(int)
    cell_y = np.floor(particle_positions[:, 1] / cell_size).astype(int)

    cell_x = np.clip(cell_x, 0, number_of_cells_x - 1)
    cell_y = np.clip(cell_y, 0, number_of_cells_y - 1)

    cells = {}

    for particle_index in range(particle_positions.shape[0]):

        key = (int(cell_x[particle_index]), int(cell_y[particle_index]))

        if key not in cells:
            cells[key] = []

        cells[key].append(particle_index)

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

                    yield i, j



###############################################################################
# 2) Forces + contacts
###############################################################################


def compute_forces_and_contacts_with_cached_cells(particle_positions,
                                                  box_width,
                                                  box_height,
                                                  particle_radius,
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

    maximum_overlap_distance = 0.0

    cutoff_radius_squared = cutoff_radius ** 2

    for i, j in generate_candidate_pairs_from_cells(cached_cells,
                                                    cached_number_of_cells_x,
                                                    cached_number_of_cells_y):

        displacement_vector = particle_positions[j] - particle_positions[i]

        distance_squared = displacement_vector[0] ** 2 + displacement_vector[1] ** 2

        if distance_squared > cutoff_radius_squared:
            continue

        center_distance = np.sqrt(distance_squared)

        if center_distance == 0.0:
            continue

        overlap_distance = 2.0 * particle_radius - center_distance

        if overlap_distance <= 0.0:
            continue

        if overlap_distance > maximum_overlap_distance:
            maximum_overlap_distance = float(overlap_distance)

        unit_normal_vector = displacement_vector / center_distance

        normal_force_magnitude = contact_spring_constant * overlap_distance

        force_vector_on_j = normal_force_magnitude * unit_normal_vector

        total_forces[i] -= force_vector_on_j
        total_forces[j] += force_vector_on_j

        contact_pairs.append((i, j))
        contact_force_vectors.append(force_vector_on_j)
        contact_branch_vectors.append(displacement_vector)

    # Soft wall forces
    x = particle_positions[:, 0]
    y = particle_positions[:, 1]

    overlap_left = particle_radius - x
    overlap_right = x + particle_radius - box_width

    overlap_bottom = particle_radius - y
    overlap_top = y + particle_radius - box_height

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

    return (total_forces,
            contact_pairs,
            contact_force_vectors,
            contact_branch_vectors,
            maximum_overlap_distance)



###############################################################################
# 3) Overdamped relaxation (Lecture 5 style)
###############################################################################


def overdamped_relaxation_step(particle_positions,
                               total_forces,
                               drag_coefficient_gamma,
                               time_step_size):
    drift_velocities = total_forces / drag_coefficient_gamma

    new_positions = particle_positions + drift_velocities * time_step_size

    return new_positions, drift_velocities



###############################################################################
# 4) Measurements (stress + network)
###############################################################################


def compute_global_stress_tensor(contact_force_vectors,
                                 contact_branch_vectors,
                                 box_width,
                                 box_height):
    if contact_force_vectors.size == 0:
        return 0.0, 0.0

    box_area = box_width * box_height

    r_x = contact_branch_vectors[:, 0]
    r_y = contact_branch_vectors[:, 1]

    f_x = contact_force_vectors[:, 0]
    f_y = contact_force_vectors[:, 1]

    sigma_xx = np.sum(r_x * f_x) / box_area
    sigma_yy = np.sum(r_y * f_y) / box_area

    return float(sigma_xx), float(sigma_yy)



def build_contact_neighbour_sets(number_of_particles,
                                 contact_pairs):
    neighbour_sets = [set() for _ in range(number_of_particles)]

    for i, j in contact_pairs:
        neighbour_sets[i].add(j)
        neighbour_sets[j].add(i)

    return neighbour_sets



def compute_degrees_from_neighbour_sets(neighbour_sets):
    degrees = np.array([len(s) for s in neighbour_sets], dtype=int)

    return degrees



def compute_mean_local_clustering_coefficient(neighbour_sets):
    clustering_values = []

    for i in range(len(neighbour_sets)):

        neighbours = list(neighbour_sets[i])

        k = len(neighbours)

        if k < 2:
            continue

        links_among_neighbours = 0
        possible_links = k * (k - 1) / 2

        for a in range(k):
            u = neighbours[a]
            for b in range(a + 1, k):
                v = neighbours[b]
                if v in neighbour_sets[u]:
                    links_among_neighbours += 1

        clustering_values.append(links_among_neighbours / possible_links)

    if len(clustering_values) == 0:
        return 0.0

    return float(np.mean(clustering_values))



def coarse_grain_sigma_yy_map(contact_force_vectors,
                              contact_branch_vectors,
                              particle_positions,
                              contact_pairs,
                              box_width,
                              box_height,
                              number_of_bins_x,
                              number_of_bins_y):
    sigma_yy_map = np.zeros((number_of_bins_y, number_of_bins_x), dtype=float)

    if contact_force_vectors.size == 0:
        return sigma_yy_map

    bin_width_x = box_width / number_of_bins_x
    bin_width_y = box_height / number_of_bins_y

    bin_area = bin_width_x * bin_width_y

    for contact_index, (i, j) in enumerate(contact_pairs):

        midpoint = 0.5 * (particle_positions[i] + particle_positions[j])

        bin_x = int(np.floor(midpoint[0] / bin_width_x))
        bin_y = int(np.floor(midpoint[1] / bin_width_y))

        bin_x = int(np.clip(bin_x, 0, number_of_bins_x - 1))
        bin_y = int(np.clip(bin_y, 0, number_of_bins_y - 1))

        r_y = contact_branch_vectors[contact_index, 1]
        f_y = contact_force_vectors[contact_index, 1]

        sigma_yy_map[bin_y, bin_x] += (r_y * f_y) / bin_area

    return sigma_yy_map



###############################################################################
# 5) One run: compress + relax until jam (with faster cell-list strategy)
###############################################################################


def run_one_jamming_simulation(number_of_particles,
                               particle_radius,
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
                               overlap_tolerance_fraction_of_radius,
                               pressure_threshold,
                               minimum_mean_degree_for_jam,
                               cell_list_rebuild_interval,
                               max_displacement_fraction_of_cell,
                               compute_heavy_outputs,
                               random_seed,
                               verbose):
    rng = np.random.default_rng(random_seed)

    box_width, box_height = compute_box_size_from_target_packing_fraction(
        number_of_particles=number_of_particles,
        particle_radius=particle_radius,
        target_packing_fraction=initial_packing_fraction,
        box_aspect_ratio=box_aspect_ratio
    )

    particle_positions = create_grid_initial_positions(
        number_of_particles=number_of_particles,
        box_width=box_width,
        box_height=box_height,
        particle_radius=particle_radius,
        extra_spacing_factor=1.05
    )

    particle_positions = add_small_position_jitter(
        particle_positions=particle_positions,
        particle_radius=particle_radius,
        random_number_generator=rng,
        jitter_fraction_of_radius=0.10
    )

    cutoff_radius = neighbour_cutoff_factor * (2.0 * particle_radius)

    cell_size = cutoff_radius

    overlap_tolerance = overlap_tolerance_fraction_of_radius * particle_radius

    max_displacement_before_rebuild = max_displacement_fraction_of_cell * cell_size

    stage_log = []

    latest_contacts = {
        "contact_pairs": [],
        "contact_force_vectors": np.zeros((0, 2), dtype=float),
        "contact_branch_vectors": np.zeros((0, 2), dtype=float),
    }

    latest_max_overlap = 0.0

    for compression_stage in range(max_number_of_compression_stages):

        # --------------------------
        # (1) Small affine compression
        # --------------------------
        scale_factor = 1.0 - compression_strain_per_stage

        box_width *= scale_factor
        box_height *= scale_factor

        particle_positions *= scale_factor

        # --------------------------
        # (2) Relaxation (overdamped)
        # --------------------------
        cached_cells, cached_nx, cached_ny = build_cell_list(
            particle_positions=particle_positions,
            box_width=box_width,
            box_height=box_height,
            cell_size=cell_size
        )

        reference_positions_for_cell_list = particle_positions.copy()

        last_pressure_proxy = 0.0
        last_mean_degree = 0.0
        last_drift_rms = 0.0
        last_max_overlap = 0.0

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

            (total_forces,
             contact_pairs,
             contact_force_vectors,
             contact_branch_vectors,
             maximum_overlap_distance) = compute_forces_and_contacts_with_cached_cells(
                particle_positions=particle_positions,
                box_width=box_width,
                box_height=box_height,
                particle_radius=particle_radius,
                contact_spring_constant=contact_spring_constant,
                wall_spring_constant=wall_spring_constant,
                cutoff_radius=cutoff_radius,
                cached_cells=cached_cells,
                cached_number_of_cells_x=cached_nx,
                cached_number_of_cells_y=cached_ny
            )

            particle_positions, drift_velocities = overdamped_relaxation_step(
                particle_positions=particle_positions,
                total_forces=total_forces,
                drag_coefficient_gamma=drag_coefficient_gamma,
                time_step_size=overdamped_time_step_size
            )

            # Light clipping for safety
            particle_positions[:, 0] = np.clip(particle_positions[:, 0],
                                               0.0 + 0.25 * particle_radius,
                                               box_width - 0.25 * particle_radius)

            particle_positions[:, 1] = np.clip(particle_positions[:, 1],
                                               0.0 + 0.25 * particle_radius,
                                               box_height - 0.25 * particle_radius)

            drift_rms = float(np.sqrt(np.mean(np.sum(drift_velocities ** 2, axis=1))))

            sigma_xx, sigma_yy = compute_global_stress_tensor(
                contact_force_vectors=contact_force_vectors,
                contact_branch_vectors=contact_branch_vectors,
                box_width=box_width,
                box_height=box_height
            )

            pressure_proxy = 0.5 * (sigma_xx + sigma_yy)

            neighbour_sets = build_contact_neighbour_sets(
                number_of_particles=number_of_particles,
                contact_pairs=contact_pairs
            )

            degrees = compute_degrees_from_neighbour_sets(neighbour_sets)

            mean_degree = float(np.mean(degrees))

            last_pressure_proxy = pressure_proxy
            last_mean_degree = mean_degree
            last_drift_rms = drift_rms
            last_max_overlap = float(maximum_overlap_distance)

            settled = (drift_rms < drift_speed_tolerance)
            overlaps_ok = (maximum_overlap_distance < overlap_tolerance)

            if settled and overlaps_ok:
                break

        packing_fraction = compute_packing_fraction(
            number_of_particles=number_of_particles,
            particle_radius=particle_radius,
            box_width=box_width,
            box_height=box_height
        )

        stage_log.append({
            "stage": compression_stage,
            "packing_fraction": packing_fraction,
            "pressure_proxy": last_pressure_proxy,
            "mean_degree": last_mean_degree,
            "drift_rms": last_drift_rms,
            "max_overlap": last_max_overlap
        })

        if verbose and (compression_stage % 10 == 0 or compression_stage < 6):
            print(f"  stage={compression_stage:3d}   "
                  f"ϕ={packing_fraction:.4f}   "
                  f"p~{last_pressure_proxy: .3e}   "
                  f"<k>={last_mean_degree:.2f}   "
                  f"v_rms={last_drift_rms: .3e}   "
                  f"ov_max={last_max_overlap: .2e}")

        latest_contacts["contact_pairs"] = contact_pairs
        latest_contacts["contact_force_vectors"] = contact_force_vectors
        latest_contacts["contact_branch_vectors"] = contact_branch_vectors

        latest_max_overlap = last_max_overlap

        jammed = (last_pressure_proxy > pressure_threshold
                  and last_mean_degree > minimum_mean_degree_for_jam
                  and last_drift_rms < drift_speed_tolerance)

        if jammed:
            break

        if packing_fraction > 0.90:
            break

    # Final scalar outputs
    contact_pairs = latest_contacts["contact_pairs"]
    contact_force_vectors = latest_contacts["contact_force_vectors"]
    contact_branch_vectors = latest_contacts["contact_branch_vectors"]

    sigma_xx, sigma_yy = compute_global_stress_tensor(
        contact_force_vectors=contact_force_vectors,
        contact_branch_vectors=contact_branch_vectors,
        box_width=box_width,
        box_height=box_height
    )

    pressure_proxy = 0.5 * (sigma_xx + sigma_yy)

    neighbour_sets = build_contact_neighbour_sets(
        number_of_particles=number_of_particles,
        contact_pairs=contact_pairs
    )

    degrees = compute_degrees_from_neighbour_sets(neighbour_sets)

    mean_degree = float(np.mean(degrees))

    mean_clustering = compute_mean_local_clustering_coefficient(neighbour_sets)

    results = {
        "final_positions": particle_positions,
        "box_width": box_width,
        "box_height": box_height,
        "packing_fraction": compute_packing_fraction(number_of_particles, particle_radius, box_width, box_height),
        "pressure_proxy": float(pressure_proxy),
        "mean_degree": float(mean_degree),
        "mean_local_clustering": float(mean_clustering),
        "degrees": degrees,
        "stage_log": stage_log,
        "final_max_overlap": float(latest_max_overlap),
    }

    # Heavy outputs only if requested
    if compute_heavy_outputs:

        sigma_yy_map = coarse_grain_sigma_yy_map(
            contact_force_vectors=contact_force_vectors,
            contact_branch_vectors=contact_branch_vectors,
            particle_positions=particle_positions,
            contact_pairs=contact_pairs,
            box_width=box_width,
            box_height=box_height,
            number_of_bins_x=30,
            number_of_bins_y=30
        )

        if pressure_proxy > 0.0:
            sigma_yy_map_normalized = sigma_yy_map / pressure_proxy
        else:
            sigma_yy_map_normalized = sigma_yy_map

        results["sigma_yy_map_normalized"] = sigma_yy_map_normalized

    return results



###############################################################################
# 6) Batch 1 driver (N sweep) — Faster settings
###############################################################################


if __name__ == "__main__":

    # ------------------------------ Run control --------------------------------

    system_sizes_to_test = [200, 800, 3200]

    number_of_repeats_per_size = 2

    verbose_progress = True



    # ------------------------------ Physical-ish parameters ----------------------

    particle_radius = 0.02

    initial_packing_fraction = 0.25

    box_aspect_ratio = 1.0

    contact_spring_constant = 2.0e4
    wall_spring_constant = 2.0e4

    neighbour_cutoff_factor = 1.25



    # ------------------------------ Faster protocol parameters -------------------

    # Slightly larger overdamped step (as requested)
    overdamped_time_step_size = 5.0e-4

    # A bit larger drag to keep it stable with larger dt
    drag_coefficient_gamma = 80.0

    # Fewer, larger compression steps
    compression_strain_per_stage = 0.01
    max_number_of_compression_stages = 140

    # Fewer relaxation steps per stage
    max_relaxation_steps_per_stage = 200

    # Settlement / quality thresholds
    drift_speed_tolerance = 1.0e-5
    overlap_tolerance_fraction_of_radius = 0.02

    pressure_threshold = 200.0
    minimum_mean_degree_for_jam = 3.5



    # ------------------------------ Cell list  ---------------------------

    # Rebuild the cell list only every few relaxation steps,
    # or if particles moved too far since the last rebuild.
    cell_list_rebuild_interval = 8

    max_displacement_fraction_of_cell = 0.25



    # ------------------------------ Run -----------------------------------------

    all_results = {}

    overall_start_time = time.time()

    for N in system_sizes_to_test:

        per_run_results = []

        for repeat_index in range(number_of_repeats_per_size):

            seed = 10_000 * N + 100 + repeat_index

            compute_heavy_outputs = (repeat_index == 0)

            print(f"\nRunning N={N}  (repeat {repeat_index + 1}/{number_of_repeats_per_size})  seed={seed}\n")

            run_start = time.time()

            results = run_one_jamming_simulation(
                number_of_particles=N,
                particle_radius=particle_radius,
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
                overlap_tolerance_fraction_of_radius=overlap_tolerance_fraction_of_radius,
                pressure_threshold=pressure_threshold,
                minimum_mean_degree_for_jam=minimum_mean_degree_for_jam,
                cell_list_rebuild_interval=cell_list_rebuild_interval,
                max_displacement_fraction_of_cell=max_displacement_fraction_of_cell,
                compute_heavy_outputs=compute_heavy_outputs,
                random_seed=seed,
                verbose=verbose_progress
            )

            run_time = time.time() - run_start

            per_run_results.append(results)

            print("\n  final:")
            print(f"    ϕ={results['packing_fraction']:.4f}")
            print(f"    p~{results['pressure_proxy']:.3e}")
            print(f"    <k>={results['mean_degree']:.2f}")
            print(f"    C(local)~{results['mean_local_clustering']:.3f}")
            print(f"    max overlap={results['final_max_overlap']:.3e}")
            print(f"    time={run_time:.1f}s\n")

        all_results[N] = per_run_results

    overall_time = time.time() - overall_start_time



    # ------------------------------ Summary -------------------------------------

    print("\n\n=== Batch 1 summary (faster protocol) ===\n")

    for N in system_sizes_to_test:

        packing_fractions = np.array([r["packing_fraction"] for r in all_results[N]], dtype=float)
        pressures = np.array([r["pressure_proxy"] for r in all_results[N]], dtype=float)
        mean_degrees = np.array([r["mean_degree"] for r in all_results[N]], dtype=float)

        print(f"N={N}")
        print(f"  packing fraction: mean={np.mean(packing_fractions):.4f}   std={np.std(packing_fractions):.4f}")
        print(f"  pressure proxy:   mean={np.mean(pressures):.4e}   std={np.std(pressures):.4e}")
        print(f"  average degree:   mean={np.mean(mean_degrees):.3f}")
        print()

    print(f"Total wall time: {overall_time:.1f}s\n")



    # ------------------------------ Plots ---------------------------------------

    # 1) Packing fraction vs N
    plt.figure(figsize=(7, 4))

    for N in system_sizes_to_test:
        for r in all_results[N]:
            plt.plot(N, r["packing_fraction"], "o", markersize=6, alpha=0.8)

    N_values = np.array(system_sizes_to_test, dtype=float)

    phi_means = np.array([np.mean([r["packing_fraction"] for r in all_results[N]]) for N in system_sizes_to_test], dtype=float)
    phi_stds = np.array([np.std([r["packing_fraction"] for r in all_results[N]]) for N in system_sizes_to_test], dtype=float)

    plt.errorbar(N_values, phi_means, yerr=phi_stds, fmt="-", linewidth=1.2, capsize=4)

    plt.xscale("log")

    plt.xlabel("Number of particles, N")
    plt.ylabel("Packing fraction, ϕ")
    plt.title("Batch 1 — packing fraction at jamming vs system size")

    plt.grid(True, which="major", linewidth=0.6, alpha=0.5)
    plt.grid(True, which="minor", linewidth=0.3, alpha=0.25)

    plt.tight_layout()
    plt.show()



    # 2) Pressure proxy vs N
    plt.figure(figsize=(7, 4))

    for N in system_sizes_to_test:
        for r in all_results[N]:
            plt.plot(N, r["pressure_proxy"], "o", markersize=6, alpha=0.8)

    p_means = np.array([np.mean([r["pressure_proxy"] for r in all_results[N]]) for N in system_sizes_to_test], dtype=float)
    p_stds = np.array([np.std([r["pressure_proxy"] for r in all_results[N]]) for N in system_sizes_to_test], dtype=float)

    plt.errorbar(N_values, p_means, yerr=p_stds, fmt="-", linewidth=1.2, capsize=4)

    plt.xscale("log")

    plt.xlabel("Number of particles, N")
    plt.ylabel("Pressure proxy (a.u.)")
    plt.title("Batch 1 — pressure proxy at jamming vs system size")

    plt.grid(True, which="major", linewidth=0.6, alpha=0.5)
    plt.grid(True, which="minor", linewidth=0.3, alpha=0.25)

    plt.tight_layout()
    plt.show()



    # 3) Degree distributions P(k) (use the first repeat for each N)
    plt.figure(figsize=(8, 4))

    for N in system_sizes_to_test:

        degrees = all_results[N][0]["degrees"]

        max_degree = int(np.max(degrees)) if degrees.size else 0

        bins = np.arange(max_degree + 2) - 0.5

        hist, edges = np.histogram(degrees, bins=bins, density=True)

        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.plot(centers, hist, ".-", linewidth=1.0, markersize=8, label=f"N={N}")

    plt.xlabel("Degree k")
    plt.ylabel("P(k)")
    plt.title("Batch 1 — contact-network degree distributions")

    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()



    # 4) Normalized σ_yy maps (only if computed in first repeats)
    N_small = system_sizes_to_test[0]
    N_large = system_sizes_to_test[-1]

    if "sigma_yy_map_normalized" in all_results[N_small][0] and "sigma_yy_map_normalized" in all_results[N_large][0]:

        map_small = all_results[N_small][0]["sigma_yy_map_normalized"]
        map_large = all_results[N_large][0]["sigma_yy_map_normalized"]

        global_vmax = float(max(np.max(map_small), np.max(map_large)))

        plt.figure(figsize=(6, 5))
        plt.imshow(map_small, origin="lower", aspect="auto", vmin=0.0, vmax=global_vmax)
        plt.colorbar(label=r"$\sigma_{yy} / \langle p \rangle$ (coarse-grained)")
        plt.title(f"Batch 1 — normalized σ_yy map (N={N_small})")
        plt.xlabel("x-bin")
        plt.ylabel("y-bin")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 5))
        plt.imshow(map_large, origin="lower", aspect="auto", vmin=0.0, vmax=global_vmax)
        plt.colorbar(label=r"$\sigma_{yy} / \langle p \rangle$ (coarse-grained)")
        plt.title(f"Batch 1 — normalized σ_yy map (N={N_large})")
        plt.xlabel("x-bin")
        plt.ylabel("y-bin")
        plt.tight_layout()
        plt.show()

