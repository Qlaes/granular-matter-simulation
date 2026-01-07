import numpy as np
import matplotlib.pyplot as plt



###############################################################################
# Batch 0 goal:
#
# - Build a small 2D granular system
# - Run a short simulation to test numerical stability
# - Check that kinetic energy decays (because we include damping)
# - Check that overlaps remain small (time step is not too big)
#
# This version uses:
# - Leapfrog integrator (Lecture 1 style)
# - Neighbour list (Lecture 1 style)
# - Soft-sphere particle contacts
# - Soft wall contacts (replaces reflecting boundaries)
# - Simple network degree histogram (Lecture 12 style)
###############################################################################



###############################################################################
# 1) Initialization helpers
###############################################################################


def create_grid_initial_positions(number_of_particles,
                                  domain_width,
                                  domain_height,
                                  particle_radius,
                                  extra_spacing_factor=1.05):
    """
    Create initial particle positions on a grid, then cut to the requested
    number_of_particles. We use spacing slightly larger than 2R so that there
    is no initial overlap.
    """

    minimum_center_to_center_spacing = extra_spacing_factor * (2.0 * particle_radius)

    number_of_grid_points_x = int(np.floor(domain_width / minimum_center_to_center_spacing))
    number_of_grid_points_y = int(np.floor(domain_height / minimum_center_to_center_spacing))

    if number_of_grid_points_x * number_of_grid_points_y < number_of_particles:
        raise ValueError(
            "Grid is too small for requested number_of_particles. "
            "Increase domain size or reduce number_of_particles."
        )

    x_coordinates = np.linspace(particle_radius,
                                domain_width - particle_radius,
                                number_of_grid_points_x)

    y_coordinates = np.linspace(particle_radius,
                                domain_height - particle_radius,
                                number_of_grid_points_y)

    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)

    flattened_x = grid_x.flatten()
    flattened_y = grid_y.flatten()

    particle_positions = np.zeros((number_of_particles, 2), dtype=float)

    particle_positions[:, 0] = flattened_x[:number_of_particles]
    particle_positions[:, 1] = flattened_y[:number_of_particles]

    return particle_positions



def create_small_random_velocities(number_of_particles,
                                   random_speed_scale,
                                   random_seed):
    """
    Small random initial velocities (so we can see if energy decays smoothly).
    """

    rng = np.random.default_rng(random_seed)

    particle_velocities = random_speed_scale * rng.normal(0.0, 1.0, size=(number_of_particles, 2))

    return particle_velocities



###############################################################################
# 2) Neighbour list
###############################################################################


def build_neighbour_list(particle_positions,
                         cutoff_radius):
    """
    Neighbour list: for each i, store indices j within cutoff_radius.
    """

    number_of_particles = particle_positions.shape[0]

    neighbour_indices_list = []

    for i in range(number_of_particles):

        displacement_vectors = particle_positions - particle_positions[i]

        distances = np.sqrt(np.sum(displacement_vectors ** 2, axis=1))

        neighbours_for_i = np.where(distances <= cutoff_radius)[0]

        neighbour_indices_list.append(neighbours_for_i)

    return neighbour_indices_list



###############################################################################
# 3) Forces: particle contacts + wall contacts
###############################################################################


def compute_soft_sphere_contact_forces(particle_positions,
                                       particle_velocities,
                                       neighbour_indices_list,
                                       particle_radius,
                                       contact_spring_constant,
                                       contact_damping_coefficient):
    """
    Soft-sphere DEM-like contact:

    overlap = 2R - distance

    elastic normal force:   k_n * overlap
    damping normal force:   - gamma_n * (v_rel · n_hat)

    Only active if overlap > 0.
    """

    number_of_particles = particle_positions.shape[0]

    total_contact_forces = np.zeros((number_of_particles, 2), dtype=float)

    for i in range(number_of_particles):

        for j in neighbour_indices_list[i]:

            if j <= i:
                continue

            displacement_vector = particle_positions[j] - particle_positions[i]

            center_distance = np.sqrt(np.sum(displacement_vector ** 2))

            if center_distance == 0.0:
                continue

            unit_normal_vector = displacement_vector / center_distance

            overlap_distance = 2.0 * particle_radius - center_distance

            if overlap_distance > 0.0:

                elastic_force_magnitude = contact_spring_constant * overlap_distance

                relative_velocity = particle_velocities[j] - particle_velocities[i]

                normal_relative_velocity = np.dot(relative_velocity, unit_normal_vector)

                damping_force_magnitude = contact_damping_coefficient * normal_relative_velocity

                normal_force_vector = (elastic_force_magnitude - damping_force_magnitude) * unit_normal_vector

                total_contact_forces[i] -= normal_force_vector
                total_contact_forces[j] += normal_force_vector

    return total_contact_forces



def compute_soft_wall_contact_forces(particle_positions,
                                     particle_velocities,
                                     particle_radius,
                                     domain_width,
                                     domain_height,
                                     wall_spring_constant,
                                     wall_damping_coefficient):
    """
    Soft repulsive contact forces between particles and the box walls.

    Walls exert:
    - elastic repulsion proportional to overlap
    - viscous damping proportional to normal velocity
    """

    number_of_particles = particle_positions.shape[0]

    wall_contact_forces = np.zeros((number_of_particles, 2), dtype=float)

    for particle_index in range(number_of_particles):

        x_position = particle_positions[particle_index, 0]
        y_position = particle_positions[particle_index, 1]

        x_velocity = particle_velocities[particle_index, 0]
        y_velocity = particle_velocities[particle_index, 1]

        # ------------------------------------------------------------------
        # Left wall (x = 0)
        # ------------------------------------------------------------------
        overlap_left_wall = particle_radius - x_position

        if overlap_left_wall > 0.0:

            wall_normal_vector = np.array([1.0, 0.0])

            normal_velocity = x_velocity

            elastic_force = wall_spring_constant * overlap_left_wall
            damping_force = wall_damping_coefficient * normal_velocity

            wall_contact_forces[particle_index] += (elastic_force - damping_force) * wall_normal_vector

        # ------------------------------------------------------------------
        # Right wall (x = domain_width)
        # ------------------------------------------------------------------
        overlap_right_wall = x_position + particle_radius - domain_width

        if overlap_right_wall > 0.0:

            wall_normal_vector = np.array([-1.0, 0.0])

            normal_velocity = -x_velocity

            elastic_force = wall_spring_constant * overlap_right_wall
            damping_force = wall_damping_coefficient * normal_velocity

            wall_contact_forces[particle_index] += (elastic_force - damping_force) * wall_normal_vector

        # ------------------------------------------------------------------
        # Bottom wall (y = 0)
        # ------------------------------------------------------------------
        overlap_bottom_wall = particle_radius - y_position

        if overlap_bottom_wall > 0.0:

            wall_normal_vector = np.array([0.0, 1.0])

            normal_velocity = y_velocity

            elastic_force = wall_spring_constant * overlap_bottom_wall
            damping_force = wall_damping_coefficient * normal_velocity

            wall_contact_forces[particle_index] += (elastic_force - damping_force) * wall_normal_vector

        # ------------------------------------------------------------------
        # Top wall (y = domain_height)
        # ------------------------------------------------------------------
        overlap_top_wall = y_position + particle_radius - domain_height

        if overlap_top_wall > 0.0:

            wall_normal_vector = np.array([0.0, -1.0])

            normal_velocity = -y_velocity

            elastic_force = wall_spring_constant * overlap_top_wall
            damping_force = wall_damping_coefficient * normal_velocity

            wall_contact_forces[particle_index] += (elastic_force - damping_force) * wall_normal_vector

    return wall_contact_forces



###############################################################################
# 4) Integrators
###############################################################################


def leapfrog_time_step(particle_positions,
                       particle_velocities,
                       particle_mass,
                       time_step_size,
                       compute_total_forces_function):
    """
    Leapfrog / half-step scheme:

    r_half = r + 0.5 v dt
    v_new  = v + (F(r_half)/m) dt
    r_new  = r_half + 0.5 v_new dt
    """

    half_step_positions = particle_positions + 0.5 * particle_velocities * time_step_size

    forces_at_half_step = compute_total_forces_function(half_step_positions, particle_velocities)

    new_velocities = particle_velocities + (forces_at_half_step / particle_mass) * time_step_size

    new_positions = half_step_positions + 0.5 * new_velocities * time_step_size

    return new_positions, new_velocities



def overdamped_relaxation_step(particle_positions,
                               particle_velocities,
                               drag_coefficient_gamma,
                               time_step_size,
                               compute_total_forces_function):
    """
    Overdamped relaxation:

    x_{n+1} = x_n + (F/gamma) dt
    """

    total_forces = compute_total_forces_function(particle_positions, particle_velocities)

    drift_velocities = total_forces / drag_coefficient_gamma

    new_positions = particle_positions + drift_velocities * time_step_size

    new_velocities = drift_velocities

    return new_positions, new_velocities



###############################################################################
# 5) Diagnostics
###############################################################################


def compute_total_kinetic_energy(particle_velocities,
                                 particle_mass):
    kinetic_energy_per_particle = 0.5 * particle_mass * np.sum(particle_velocities ** 2, axis=1)

    return float(np.sum(kinetic_energy_per_particle))



def compute_maximum_overlap(particle_positions,
                            particle_radius):
    """
    Slow O(N^2) overlap check (fine for Batch 0 sizes).
    """

    number_of_particles = particle_positions.shape[0]

    maximum_overlap_distance = 0.0

    for i in range(number_of_particles):
        for j in range(i + 1, number_of_particles):

            displacement_vector = particle_positions[j] - particle_positions[i]

            center_distance = np.sqrt(np.sum(displacement_vector ** 2))

            overlap_distance = 2.0 * particle_radius - center_distance

            if overlap_distance > maximum_overlap_distance:
                maximum_overlap_distance = overlap_distance

    return float(maximum_overlap_distance)



def build_contact_adjacency_matrix(particle_positions,
                                  particle_radius,
                                  contact_tolerance=0.0):
    """
    Contact network adjacency matrix.
    """

    number_of_particles = particle_positions.shape[0]

    adjacency_matrix = np.zeros((number_of_particles, number_of_particles), dtype=int)

    contact_distance_threshold = 2.0 * particle_radius + contact_tolerance

    for i in range(number_of_particles):
        for j in range(i + 1, number_of_particles):

            distance_ij = np.sqrt(np.sum((particle_positions[i] - particle_positions[j]) ** 2))

            if distance_ij <= contact_distance_threshold:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix



def compute_node_degrees(adjacency_matrix):
    """
    Degree K_i = sum_j A_ij
    """

    node_degrees = np.sum(adjacency_matrix, axis=0)

    return node_degrees



###############################################################################
# 6) Batch 0 runner
###############################################################################


def run_batch0_shakedown(number_of_particles,
                         domain_width,
                         domain_height,
                         particle_radius,
                         particle_mass,
                         time_step_size,
                         number_of_time_steps,
                         neighbour_list_cutoff_radius,
                         neighbour_list_update_interval,
                         contact_spring_constant,
                         contact_damping_coefficient,
                         global_drag_coefficient,
                         gravitational_acceleration,
                         integrator_mode,
                         random_seed):
    """
    Run a Batch 0 simulation.

    integrator_mode:
        "inertial_leapfrog"
        "overdamped"
    """

    particle_positions = create_grid_initial_positions(
        number_of_particles=number_of_particles,
        domain_width=domain_width,
        domain_height=domain_height,
        particle_radius=particle_radius
    )

    particle_velocities = create_small_random_velocities(
        number_of_particles=number_of_particles,
        random_speed_scale=0.01,
        random_seed=random_seed
    )

    neighbour_indices_list = build_neighbour_list(
        particle_positions=particle_positions,
        cutoff_radius=neighbour_list_cutoff_radius
    )

    gravitational_acceleration_vector = np.array([0.0, gravitational_acceleration], dtype=float)

    kinetic_energy_time_series = np.zeros(number_of_time_steps, dtype=float)

    maximum_overlap_time_series = np.zeros(number_of_time_steps, dtype=float)



    def compute_total_forces(current_positions, current_velocities):

        contact_forces = compute_soft_sphere_contact_forces(
            particle_positions=current_positions,
            particle_velocities=current_velocities,
            neighbour_indices_list=neighbour_indices_list,
            particle_radius=particle_radius,
            contact_spring_constant=contact_spring_constant,
            contact_damping_coefficient=contact_damping_coefficient
        )



        wall_forces = compute_soft_wall_contact_forces(
            particle_positions=current_positions,
            particle_velocities=current_velocities,
            particle_radius=particle_radius,
            domain_width=domain_width,
            domain_height=domain_height,
            wall_spring_constant=contact_spring_constant,
            wall_damping_coefficient=contact_damping_coefficient
        )



        gravity_forces = particle_mass * gravitational_acceleration_vector



        global_drag_forces = - global_drag_coefficient * current_velocities



        total_forces = contact_forces + wall_forces + gravity_forces + global_drag_forces



        return total_forces



    for step_index in range(number_of_time_steps):

        if step_index % neighbour_list_update_interval == 0:
            neighbour_indices_list = build_neighbour_list(
                particle_positions=particle_positions,
                cutoff_radius=neighbour_list_cutoff_radius
            )

        if integrator_mode == "inertial_leapfrog":

            particle_positions, particle_velocities = leapfrog_time_step(
                particle_positions=particle_positions,
                particle_velocities=particle_velocities,
                particle_mass=particle_mass,
                time_step_size=time_step_size,
                compute_total_forces_function=compute_total_forces
            )

        elif integrator_mode == "overdamped":

            particle_positions, particle_velocities = overdamped_relaxation_step(
                particle_positions=particle_positions,
                particle_velocities=particle_velocities,
                drag_coefficient_gamma=max(global_drag_coefficient, 1e-12),
                time_step_size=time_step_size,
                compute_total_forces_function=compute_total_forces
            )

        else:
            raise ValueError("integrator_mode must be 'inertial_leapfrog' or 'overdamped'")

        kinetic_energy_time_series[step_index] = compute_total_kinetic_energy(
            particle_velocities=particle_velocities,
            particle_mass=particle_mass
        )

        maximum_overlap_time_series[step_index] = compute_maximum_overlap(
            particle_positions=particle_positions,
            particle_radius=particle_radius
        )



    adjacency_matrix = build_contact_adjacency_matrix(
        particle_positions=particle_positions,
        particle_radius=particle_radius,
        contact_tolerance=1e-6
    )

    node_degrees = compute_node_degrees(adjacency_matrix)



    results = {
        "final_positions": particle_positions,
        "final_velocities": particle_velocities,
        "kinetic_energy_time_series": kinetic_energy_time_series,
        "maximum_overlap_time_series": maximum_overlap_time_series,
        "contact_adjacency_matrix": adjacency_matrix,
        "contact_node_degrees": node_degrees,
    }

    return results



###############################################################################
# 7) Main: Batch 0 parameter scan (dt sanity check)
###############################################################################


if __name__ == "__main__":

    number_of_particles = 100

    domain_width = 1.0
    domain_height = 1.0

    particle_radius = 0.02
    particle_mass = 1.0



    # Contact parameters
    contact_spring_constant = 20_000.0
    contact_damping_coefficient = 20.0



    # Drag (helps kinetic energy decay)
    global_drag_coefficient = 10.0



    gravitational_acceleration = -9.81



    neighbour_list_cutoff_radius = 3.0 * particle_radius
    neighbour_list_update_interval = 10



    integrator_mode = "inertial_leapfrog"     # or "overdamped"
    random_seed = 0



    number_of_time_steps = 5_000



    candidate_time_step_sizes = [5e-4, 2e-4, 1e-4, 5e-5]



    print("\nBatch 0: scanning time_step_size for stability\n")

    for time_step_size in candidate_time_step_sizes:

        results = run_batch0_shakedown(
            number_of_particles=number_of_particles,
            domain_width=domain_width,
            domain_height=domain_height,
            particle_radius=particle_radius,
            particle_mass=particle_mass,
            time_step_size=time_step_size,
            number_of_time_steps=number_of_time_steps,
            neighbour_list_cutoff_radius=neighbour_list_cutoff_radius,
            neighbour_list_update_interval=neighbour_list_update_interval,
            contact_spring_constant=contact_spring_constant,
            contact_damping_coefficient=contact_damping_coefficient,
            global_drag_coefficient=global_drag_coefficient,
            gravitational_acceleration=gravitational_acceleration,
            integrator_mode=integrator_mode,
            random_seed=random_seed
        )

        kinetic_energy_series = results["kinetic_energy_time_series"]
        maximum_overlap_series = results["maximum_overlap_time_series"]

        print(f"time_step_size = {time_step_size: .1e}    "
              f"KE_start = {kinetic_energy_series[0]: .3e}    "
              f"KE_end = {kinetic_energy_series[-1]: .3e}    "
              f"max_overlap_end = {maximum_overlap_series[-1]: .3e}")



    chosen_time_step_size = candidate_time_step_sizes[2]



    results = run_batch0_shakedown(
        number_of_particles=number_of_particles,
        domain_width=domain_width,
        domain_height=domain_height,
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        time_step_size=chosen_time_step_size,
        number_of_time_steps=number_of_time_steps,
        neighbour_list_cutoff_radius=neighbour_list_cutoff_radius,
        neighbour_list_update_interval=neighbour_list_update_interval,
        contact_spring_constant=contact_spring_constant,
        contact_damping_coefficient=contact_damping_coefficient,
        global_drag_coefficient=global_drag_coefficient,
        gravitational_acceleration=gravitational_acceleration,
        integrator_mode=integrator_mode,
        random_seed=random_seed
    )

    kinetic_energy_series = results["kinetic_energy_time_series"]
    maximum_overlap_series = results["maximum_overlap_time_series"]

    time_axis = chosen_time_step_size * np.arange(number_of_time_steps)



    ###############################################################################
    # Plotting (ONLY PART CHANGED)
    ###############################################################################

    rng_for_plotting = np.random.default_rng(42)



    def _human_axes_formatting():

        current_axes = plt.gca()

        current_axes.spines["top"].set_visible(False)
        current_axes.spines["right"].set_visible(False)

        current_axes.grid(True, which="major", alpha=0.25, linewidth=0.6)
        current_axes.grid(True, which="minor", alpha=0.12, linewidth=0.4)

        current_axes.minorticks_on()



    # -------------------------
    # 1) Kinetic energy
    # -------------------------

    plt.figure(figsize=(7.2, 4.6))

    plt.plot(time_axis, kinetic_energy_series, linewidth=1.2)

    downsample_step = max(1, len(time_axis) // 70)

    plt.plot(time_axis[::downsample_step],
             kinetic_energy_series[::downsample_step],
             linestyle="None",
             marker="o",
             markersize=3,
             alpha=0.7)

    plt.xlabel("time (s)")
    plt.ylabel("total kinetic energy")

    plt.title("Batch 0 — kinetic energy decay as a function of time")

    if np.max(kinetic_energy_series) / max(np.min(kinetic_energy_series[kinetic_energy_series > 0]), 1e-30) > 1e3:
        plt.yscale("log")
        plt.text(0.02, 0.98, "log scale",
                 transform=plt.gca().transAxes,
                 va="top",
                 fontsize=9,
                 alpha=0.85)

    _human_axes_formatting()

    plt.tight_layout()
    plt.show()



    # -------------------------
    # 2) Maximum overlap
    # -------------------------

    plt.figure(figsize=(7.2, 4.6))

    plt.plot(time_axis, maximum_overlap_series, linewidth=1.2)

    plt.xlabel("time (s)")
    plt.ylabel("maximum overlap distance")

    plt.title("Batch 0 — Particle overlap as a function of time")

    y_max = float(np.max(maximum_overlap_series))

    plt.ylim(bottom=min(0.0, -0.05 * y_max),
             top=1.10 * y_max if y_max > 0 else 1e-6)

    _human_axes_formatting()

    plt.annotate(f" ≈ {maximum_overlap_series[-1]:.2e}",
                 xy=(time_axis[-1], maximum_overlap_series[-1]),
                 xytext=(time_axis[int(0.70 * len(time_axis))],
                         (0.85 * y_max) if y_max > 0 else 1e-6),
                 arrowprops=dict(arrowstyle="->", lw=0.8),
                 fontsize=9,
                 alpha=0.9)

    plt.tight_layout()
    plt.show()



    # -------------------------
    # 3) Final particle positions
    # -------------------------

    final_positions = results["final_positions"]

    plt.figure(figsize=(6.0, 6.0))

    jitter_strength = 0.02 * particle_radius
    jitter = rng_for_plotting.normal(0.0, jitter_strength, size=final_positions.shape)

    plt.scatter(final_positions[:, 0] + jitter[:, 0],
                final_positions[:, 1] + jitter[:, 1],
                s=18,
                alpha=0.75,
                linewidths=0.0)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Batch 0 — final particle positions")

    margin = 3.0 * particle_radius

    plt.xlim(-margin, domain_width + margin)
    plt.ylim(-margin, domain_height + margin)

    plt.gca().set_aspect("equal", adjustable="box")

    plt.text(0.02, 0.98, f"N={number_of_particles}, dt={chosen_time_step_size:.1e}",
             transform=plt.gca().transAxes,
             va="top",
             fontsize=9,
             alpha=0.85)

    _human_axes_formatting()

    plt.tight_layout()
    plt.show()



    # -------------------------
    # 4) Degree histogram
    # -------------------------

    contact_node_degrees = results["contact_node_degrees"]

    plt.figure(figsize=(7.2, 4.6))

    max_degree = int(np.max(contact_node_degrees)) if len(contact_node_degrees) else 0

    bin_edges = np.arange(max_degree + 2) - 0.5

    counts, _, _ = plt.hist(contact_node_degrees,
                            bins=bin_edges,
                            alpha=0.85,
                            rwidth=0.88)

    mean_degree = float(np.mean(contact_node_degrees))

    plt.axvline(mean_degree, linewidth=1.2, alpha=0.9)

    plt.text(mean_degree + 0.1,
             (0.95 * float(np.max(counts))) if len(counts) else 0.0,
             f"mean ≈ {mean_degree:.2f}",
             fontsize=9,
             va="top",
             alpha=0.9)

    plt.xlabel("node degree (number of contacts)")
    plt.ylabel("count")
    plt.title("Batch 0 — contact network degree distribution")
    _human_axes_formatting()
    plt.tight_layout()
    plt.show()
