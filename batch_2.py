import time

import numpy as np
import matplotlib.pyplot as plt



###############################################################################
# Batch 2 — Grain-size distribution sweep (fixed system size)
###############################################################################



###############################################################################
# 0) Grain-size distributions (PSD)
###############################################################################


def generate_particle_radii_for_distribution(distribution_name,
                                             number_of_particles,
                                             base_radius,
                                             random_number_generator,
                                             bidisperse_large_radius_factor=1.40,
                                             bidisperse_large_fraction=0.50,
                                             polydisperse_min_radius_factor=0.70,
                                             polydisperse_max_radius_factor=1.30):
    """
    Return an array radii[i] for a chosen grain-size distribution.

    distribution_name:
        "monodisperse"
        "bidisperse"
        "polydisperse_uniform"
    """

    if distribution_name == "monodisperse":

        particle_radii = np.full(number_of_particles, base_radius, dtype=float)

        distribution_metadata = {
            "distribution": "monodisperse",
            "base_radius": float(base_radius),
        }

        return particle_radii, distribution_metadata



    if distribution_name == "bidisperse":

        small_radius = float(base_radius)

        large_radius = float(bidisperse_large_radius_factor * base_radius)

        number_large = int(np.round(bidisperse_large_fraction * number_of_particles))

        number_small = number_of_particles - number_large

        particle_radii = np.array(
            [small_radius] * number_small + [large_radius] * number_large,
            dtype=float
        )

        random_number_generator.shuffle(particle_radii)

        distribution_metadata = {
            "distribution": "bidisperse",
            "small_radius": float(small_radius),
            "large_radius": float(large_radius),
            "large_fraction": float(number_large / number_of_particles),
        }

        return particle_radii, distribution_metadata



    if distribution_name == "polydisperse_uniform":

        minimum_radius = float(polydisperse_min_radius_factor * base_radius)

        maximum_radius = float(polydisperse_max_radius_factor * base_radius)

        particle_radii = random_number_generator.uniform(minimum_radius,
                                                         maximum_radius,
                                                         size=number_of_particles).astype(float)

        distribution_metadata = {
            "distribution": "polydisperse_uniform",
            "minimum_radius": float(minimum_radius),
            "maximum_radius": float(maximum_radius),
        }

        return particle_radii, distribution_metadata



    raise ValueError("Unknown distribution_name. Use: monodisperse, bidisperse, polydisperse_uniform")



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
    """
    Grid initialization with spacing based on the *largest* particle radius,
    so we avoid catastrophic overlaps at t=0.
    """

    minimum_spacing = float(extra_spacing_factor * (2.0 * maximum_particle_radius))

    number_of_grid_points_x = int(np.floor(box_width / minimum_spacing))
    number_of_grid_points_y = int(np.floor(box_height / minimum_spacing))

    if number_of_grid_points_x * number_of_grid_points_y < number_of_particles:
        raise ValueError("Grid too small for requested N (increase box or reduce N).")

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
    """
    Jitter per particle, scaled to its own radius.
    This helps break perfect crystallization from grid placement.
    """

    jitter_scales = (jitter_fraction_of_local_radius * particle_radii).reshape(-1, 1)

    jitter = random_number_generator.uniform(-1.0, +1.0, size=particle_positions.shape) * jitter_scales

    particle_positions = particle_positions + jitter

    return particle_positions



def compute_packing_fraction(particle_radii,
                             box_width,
                             box_height):
    total_particle_area = float(np.sum(np.pi * particle_radii ** 2))

    box_area = float(box_width * box_height)

    return float(total_particle_area / box_area)



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

        cells[cell_key].append(particle_index)

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
# 3) Forces + contacts (polydisperse)
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
    """
    Soft repulsive normal contact:
        overlap_ij = (r_i + r_j) - |x_j - x_i|
        F_n = k * overlap_ij    for overlap_ij > 0

    Wall overlaps use each particle's own radius.
    """

    number_of_particles = particle_positions.shape[0]

    total_forces = np.zeros((number_of_particles, 2), dtype=float)

    contact_pairs = []
    contact_force_vectors = []
    contact_branch_vectors = []

    maximum_overlap_absolute = 0.0

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

        if overlap_distance > maximum_overlap_absolute:
            maximum_overlap_absolute = float(overlap_distance)

        unit_normal_vector = displacement_vector / center_distance

        normal_force_magnitude = float(contact_spring_constant * overlap_distance)

        force_vector_on_j = normal_force_magnitude * unit_normal_vector

        total_forces[i] -= force_vector_on_j
        total_forces[j] += force_vector_on_j

        contact_pairs.append((int(i), int(j)))
        contact_force_vectors.append(force_vector_on_j)
        contact_branch_vectors.append(displacement_vector)



    # Soft wall forces (vectorized)
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

    return (total_forces,
            contact_pairs,
            contact_force_vectors,
            contact_branch_vectors,
            maximum_overlap_absolute)



###############################################################################
# 4) Overdamped relaxation (Lecture 5 style)
###############################################################################


def overdamped_relaxation_step(particle_positions,
                               total_forces,
                               drag_coefficient_gamma,
                               time_step_size):
    drift_velocities = total_forces / float(drag_coefficient_gamma)

    new_positions = particle_positions + drift_velocities * float(time_step_size)

    return new_positions, drift_velocities



###############################################################################
# 5) Measurements: stress + network
###############################################################################


def compute_global_stress_tensor(contact_force_vectors,
                                 contact_branch_vectors,
                                 box_width,
                                 box_height):
    """
    Minimal (σ_xx, σ_yy) using contact virial-like form:
        σ_ij ~ (1/A) sum_contacts r_i * f_j
    """

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
    """
    Local clustering computed from neighbour sets (sparse-friendly).
    """

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

        sigma_yy_map[bin_y, bin_x] += (r_y * f_y) / bin_area

    return sigma_yy_map



###############################################################################
# 6) One run: compress + relax until jam (polydisperse)
###############################################################################


def run_one_jamming_simulation_polydisperse(number_of_particles,
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
                                            overlap_tolerance_fraction_of_min_radius,
                                            pressure_threshold,
                                            minimum_mean_degree_for_jam,
                                            cell_list_rebuild_interval,
                                            max_displacement_fraction_of_cell,
                                            compute_heavy_outputs,
                                            random_seed,
                                            verbose):
    rng = np.random.default_rng(random_seed)

    particle_radii = np.asarray(particle_radii, dtype=float)

    maximum_particle_radius = float(np.max(particle_radii))
    minimum_particle_radius = float(np.min(particle_radii))

    overlap_tolerance_absolute = float(overlap_tolerance_fraction_of_min_radius * minimum_particle_radius)



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



    stage_log = []

    latest_contacts = {
        "contact_pairs": [],
        "contact_force_vectors": np.zeros((0, 2), dtype=float),
        "contact_branch_vectors": np.zeros((0, 2), dtype=float),
    }

    latest_max_overlap = 0.0



    for compression_stage in range(max_number_of_compression_stages):

        # --------------------------
        # (1) Affine compression
        # --------------------------
        scale_factor = float(1.0 - compression_strain_per_stage)

        box_width *= scale_factor
        box_height *= scale_factor

        particle_positions *= scale_factor



        # --------------------------
        # (2) Overdamped relaxation
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
             maximum_overlap_absolute) = compute_forces_and_contacts_polydisperse(
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



            particle_positions, drift_velocities = overdamped_relaxation_step(
                particle_positions=particle_positions,
                total_forces=total_forces,
                drag_coefficient_gamma=drag_coefficient_gamma,
                time_step_size=overdamped_time_step_size
            )



            # Keep particles inside the box (soft walls do most of the work; this is a safety clamp)
            particle_positions[:, 0] = np.clip(particle_positions[:, 0],
                                               0.0 + particle_radii,
                                               box_width - particle_radii)

            particle_positions[:, 1] = np.clip(particle_positions[:, 1],
                                               0.0 + particle_radii,
                                               box_height - particle_radii)



            drift_rms = float(np.sqrt(np.mean(np.sum(drift_velocities ** 2, axis=1))))

            sigma_xx, sigma_yy = compute_global_stress_tensor(
                contact_force_vectors=contact_force_vectors,
                contact_branch_vectors=contact_branch_vectors,
                box_width=box_width,
                box_height=box_height
            )

            pressure_proxy = float(0.5 * (sigma_xx + sigma_yy))



            neighbour_sets = build_contact_neighbour_sets(
                number_of_particles=number_of_particles,
                contact_pairs=contact_pairs
            )

            degrees = compute_degrees_from_neighbour_sets(neighbour_sets)

            mean_degree = float(np.mean(degrees))



            last_pressure_proxy = pressure_proxy
            last_mean_degree = mean_degree
            last_drift_rms = drift_rms
            last_max_overlap = float(maximum_overlap_absolute)



            settled = (drift_rms < drift_speed_tolerance)
            overlaps_ok = (maximum_overlap_absolute < overlap_tolerance_absolute)

            if settled and overlaps_ok:
                break



        packing_fraction = compute_packing_fraction(
            particle_radii=particle_radii,
            box_width=box_width,
            box_height=box_height
        )

        stage_log.append({
            "stage": int(compression_stage),
            "packing_fraction": float(packing_fraction),
            "pressure_proxy": float(last_pressure_proxy),
            "mean_degree": float(last_mean_degree),
            "drift_rms": float(last_drift_rms),
            "max_overlap_absolute": float(last_max_overlap),
        })

        if verbose and (compression_stage % 10 == 0 or compression_stage < 6):
            print(f"  stage={compression_stage:3d}   "
                  f"phi={packing_fraction:.4f}   "
                  f"p~{last_pressure_proxy: .3e}   "
                  f"<k>={last_mean_degree:.2f}   "
                  f"v_rms={last_drift_rms: .3e}   "
                  f"ov_max={last_max_overlap: .2e}")



        latest_contacts["contact_pairs"] = contact_pairs
        latest_contacts["contact_force_vectors"] = contact_force_vectors
        latest_contacts["contact_branch_vectors"] = contact_branch_vectors

        latest_max_overlap = float(last_max_overlap)



        jammed = (last_pressure_proxy > pressure_threshold
                  and last_mean_degree > minimum_mean_degree_for_jam
                  and last_drift_rms < drift_speed_tolerance)

        if jammed:
            break



        if packing_fraction > 0.92:
            break



    # Final outputs
    contact_pairs = latest_contacts["contact_pairs"]
    contact_force_vectors = latest_contacts["contact_force_vectors"]
    contact_branch_vectors = latest_contacts["contact_branch_vectors"]

    sigma_xx, sigma_yy = compute_global_stress_tensor(
        contact_force_vectors=contact_force_vectors,
        contact_branch_vectors=contact_branch_vectors,
        box_width=box_width,
        box_height=box_height
    )

    pressure_proxy = float(0.5 * (sigma_xx + sigma_yy))

    neighbour_sets = build_contact_neighbour_sets(
        number_of_particles=number_of_particles,
        contact_pairs=contact_pairs
    )

    degrees = compute_degrees_from_neighbour_sets(neighbour_sets)

    mean_degree = float(np.mean(degrees))

    mean_clustering = compute_mean_local_clustering_coefficient(neighbour_sets)



    results = {
        "final_positions": particle_positions,
        "particle_radii": particle_radii,
        "box_width": float(box_width),
        "box_height": float(box_height),
        "packing_fraction": compute_packing_fraction(particle_radii, box_width, box_height),
        "pressure_proxy": float(pressure_proxy),
        "mean_degree": float(mean_degree),
        "mean_local_clustering": float(mean_clustering),
        "degrees": degrees,
        "stage_log": stage_log,
        "final_max_overlap_absolute": float(latest_max_overlap),
    }



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
# 7) Batch 2 driver (PSD sweep at fixed N)
###############################################################################


if __name__ == "__main__":

    # ------------------------------ Run control --------------------------------

    fixed_system_size_N = 800

    number_of_repeats_per_distribution = 3

    verbose_progress = True



    # ------------------------------ PSD definitions -----------------------------

    base_radius = 0.02

    distributions_to_test = [
        {
            "name": "monodisperse",
        },
        {
            "name": "bidisperse",
            "bidisperse_large_radius_factor": 1.40,
            "bidisperse_large_fraction": 0.50,
        },
        {
            "name": "polydisperse_uniform",
            "polydisperse_min_radius_factor": 0.70,
            "polydisperse_max_radius_factor": 1.30,
        },
    ]



    # ------------------------------ Protocol parameters -------------------------

    initial_packing_fraction = 0.25

    box_aspect_ratio = 1.0

    contact_spring_constant = 2.0e4
    wall_spring_constant = 2.0e4

    neighbour_cutoff_factor = 1.25

    overdamped_time_step_size = 5.0e-4
    drag_coefficient_gamma = 80.0

    compression_strain_per_stage = 0.01
    max_number_of_compression_stages = 160

    max_relaxation_steps_per_stage = 220

    drift_speed_tolerance = 1.0e-5
    overlap_tolerance_fraction_of_min_radius = 0.02

    pressure_threshold = 200.0
    minimum_mean_degree_for_jam = 3.5



    # ------------------------------ Cell list caching ---------------------------

    cell_list_rebuild_interval = 8

    max_displacement_fraction_of_cell = 0.25



    # ------------------------------ Run -----------------------------------------

    all_results_by_distribution = {}

    overall_start_time = time.time()



    for distribution_settings in distributions_to_test:

        distribution_name = distribution_settings["name"]

        per_run_results = []

        for repeat_index in range(number_of_repeats_per_distribution):

            seed = 50_000 + 1_000 * repeat_index + hash(distribution_name) % 10_000

            random_number_generator = np.random.default_rng(seed)

            particle_radii, distribution_metadata = generate_particle_radii_for_distribution(
                distribution_name=distribution_name,
                number_of_particles=fixed_system_size_N,
                base_radius=base_radius,
                random_number_generator=random_number_generator,
                bidisperse_large_radius_factor=distribution_settings.get("bidisperse_large_radius_factor", 1.40),
                bidisperse_large_fraction=distribution_settings.get("bidisperse_large_fraction", 0.50),
                polydisperse_min_radius_factor=distribution_settings.get("polydisperse_min_radius_factor", 0.70),
                polydisperse_max_radius_factor=distribution_settings.get("polydisperse_max_radius_factor", 1.30),
            )

            compute_heavy_outputs = (repeat_index == 0)

            print(f"\nBatch 2: {distribution_name}   (repeat {repeat_index + 1}/{number_of_repeats_per_distribution})")
            print(f"  radii summary: min={np.min(particle_radii):.4f}  mean={np.mean(particle_radii):.4f}  max={np.max(particle_radii):.4f}")

            run_start = time.time()

            results = run_one_jamming_simulation_polydisperse(
                number_of_particles=fixed_system_size_N,
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
                overlap_tolerance_fraction_of_min_radius=overlap_tolerance_fraction_of_min_radius,
                pressure_threshold=pressure_threshold,
                minimum_mean_degree_for_jam=minimum_mean_degree_for_jam,
                cell_list_rebuild_interval=cell_list_rebuild_interval,
                max_displacement_fraction_of_cell=max_displacement_fraction_of_cell,
                compute_heavy_outputs=compute_heavy_outputs,
                random_seed=seed,
                verbose=verbose_progress
            )

            run_time = time.time() - run_start

            results["distribution_name"] = distribution_name
            results["distribution_metadata"] = distribution_metadata

            per_run_results.append(results)

            print("  final:")
            print(f"    phi={results['packing_fraction']:.4f}")
            print(f"    p~{results['pressure_proxy']:.3e}")
            print(f"    <k>={results['mean_degree']:.2f}")
            print(f"    C(local)~{results['mean_local_clustering']:.3f}")
            print(f"    max overlap={results['final_max_overlap_absolute']:.3e}")
            print(f"    time={run_time:.1f}s\n")


        all_results_by_distribution[distribution_name] = per_run_results



    overall_wall_time = time.time() - overall_start_time



    # ------------------------------ Summary -------------------------------------

    print("\n\n=== Batch 2 summary (PSD sweep, fixed N) ===\n")

    for distribution_settings in distributions_to_test:

        distribution_name = distribution_settings["name"]

        packing_fractions = np.array([r["packing_fraction"] for r in all_results_by_distribution[distribution_name]], dtype=float)
        pressures = np.array([r["pressure_proxy"] for r in all_results_by_distribution[distribution_name]], dtype=float)
        mean_degrees = np.array([r["mean_degree"] for r in all_results_by_distribution[distribution_name]], dtype=float)

        print(distribution_name)
        print(f"  packing fraction: mean={np.mean(packing_fractions):.4f}   std={np.std(packing_fractions):.4f}")
        print(f"  pressure proxy:   mean={np.mean(pressures):.4e}   std={np.std(pressures):.4e}")
        print(f"  average degree:   mean={np.mean(mean_degrees):.3f}")
        print()

    print(f"Total wall time: {overall_wall_time:.1f}s\n")



    ###############################################################################
    # Plotting (varied, “human” style)
    ###############################################################################


    def human_axes_cleanup():
        current_axes = plt.gca()
        current_axes.spines["top"].set_visible(False)
        current_axes.spines["right"].set_visible(False)
        current_axes.grid(True, which="major", alpha=0.25, linewidth=0.7)
        current_axes.minorticks_on()
        current_axes.grid(True, which="minor", alpha=0.12, linewidth=0.4)



    # 1) Radii distributions (sanity check)
    plt.figure(figsize=(9.5, 3.5))

    for distribution_index, distribution_settings in enumerate(distributions_to_test):

        distribution_name = distribution_settings["name"]

        radii = all_results_by_distribution[distribution_name][0]["particle_radii"]

        plt.subplot(1, len(distributions_to_test), distribution_index + 1)

        plt.hist(radii, bins=18, alpha=0.85)

        plt.title(distribution_name)
        plt.xlabel("radius")
        plt.ylabel("count")

        human_axes_cleanup()

    plt.suptitle(f"Batch 2 — grain-size distributions (N={fixed_system_size_N})", y=1.03)
    plt.tight_layout()
    plt.show()



    # 2) Packing fraction comparison (means + std)
    distribution_names = [d["name"] for d in distributions_to_test]

    phi_means = []
    phi_stds = []

    for name in distribution_names:
        values = np.array([r["packing_fraction"] for r in all_results_by_distribution[name]], dtype=float)
        phi_means.append(float(np.mean(values)))
        phi_stds.append(float(np.std(values)))

    x_positions = np.arange(len(distribution_names))

    plt.figure(figsize=(7.4, 4.2))

    plt.errorbar(x_positions,
                 phi_means,
                 yerr=phi_stds,
                 fmt="o",
                 capsize=5,
                 linewidth=1.2,
                 markersize=7,
                 alpha=0.95)

    for i, name in enumerate(distribution_names):
        scatter_y = np.array([r["packing_fraction"] for r in all_results_by_distribution[name]], dtype=float)
        jitter = 0.06 * np.random.default_rng(123 + i).normal(0.0, 1.0, size=scatter_y.size)
        plt.plot(np.full(scatter_y.size, x_positions[i]) + jitter,
                 scatter_y,
                 linestyle="None",
                 marker=".",
                 markersize=9,
                 alpha=0.55)

    plt.xticks(x_positions, distribution_names, rotation=15)
    plt.ylabel("packing fraction, ϕ")
    plt.title("Batch 2 — packing fraction at jamming vs grain-size distribution")

    human_axes_cleanup()

    plt.tight_layout()
    plt.show()



    # 3) Pressure proxy comparison (means + std)
    p_means = []
    p_stds = []

    for name in distribution_names:
        values = np.array([r["pressure_proxy"] for r in all_results_by_distribution[name]], dtype=float)
        p_means.append(float(np.mean(values)))
        p_stds.append(float(np.std(values)))

    plt.figure(figsize=(7.4, 4.2))

    plt.errorbar(x_positions,
                 p_means,
                 yerr=p_stds,
                 fmt="s",
                 capsize=5,
                 linewidth=1.2,
                 markersize=6,
                 alpha=0.95)

    for i, name in enumerate(distribution_names):
        scatter_y = np.array([r["pressure_proxy"] for r in all_results_by_distribution[name]], dtype=float)
        jitter = 0.06 * np.random.default_rng(999 + i).normal(0.0, 1.0, size=scatter_y.size)
        plt.plot(np.full(scatter_y.size, x_positions[i]) + jitter,
                 scatter_y,
                 linestyle="None",
                 marker="x",
                 markersize=7,
                 alpha=0.60)

    plt.xticks(x_positions, distribution_names, rotation=15)
    plt.ylabel("pressure proxy (a.u.)")
    plt.title("Batch 2 — pressure proxy at jamming vs grain-size distribution")

    human_axes_cleanup()

    plt.tight_layout()
    plt.show()



    # 4) Degree distributions P(k) for first repeat of each PSD
    plt.figure(figsize=(8.8, 4.6))

    for name in distribution_names:

        degrees = all_results_by_distribution[name][0]["degrees"]

        max_degree = int(np.max(degrees)) if degrees.size else 0

        bins = np.arange(max_degree + 2) - 0.5

        hist, edges = np.histogram(degrees, bins=bins, density=True)

        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.plot(centers,
                 hist,
                 marker="o",
                 markersize=4,
                 linewidth=1.1,
                 alpha=0.9,
                 label=name)

    plt.xlabel("degree k (contacts per particle)")
    plt.ylabel("P(k)")
    plt.title("Batch 2 — contact-network degree distributions (first repeat)")

    plt.legend(frameon=False)

    human_axes_cleanup()
    
    plt.tight_layout()
    plt.show()



    # 5) Normalized σ_yy stress maps (first repeat for each PSD, if available)
    available_maps = []
    for name in distribution_names:
        if "sigma_yy_map_normalized" in all_results_by_distribution[name][0]:
            available_maps.append(name)

    if len(available_maps) > 0:

        global_vmax = 0.0
        for name in available_maps:
            global_vmax = max(global_vmax, float(np.max(all_results_by_distribution[name][0]["sigma_yy_map_normalized"])))

        for name in available_maps:

            sigma_yy_map_normalized = all_results_by_distribution[name][0]["sigma_yy_map_normalized"]

            plt.figure(figsize=(6.2, 5.2))

            plt.imshow(sigma_yy_map_normalized,
                       origin="lower",
                       aspect="auto",
                       vmin=0.0,
                       vmax=global_vmax)

            plt.colorbar(label=r"$\sigma_{yy} / \langle p \rangle$ (coarse-grained)")
            plt.title(f"Batch 2 — normalized σ_yy map ({name})")
            plt.xlabel("x-bin")
            plt.ylabel("y-bin")
            plt.tight_layout()
            plt.show()

