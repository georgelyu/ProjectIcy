"""D2Q9 primitives used by the IceFlow2D Taichi kernels."""

import taichi as ti

D2Q9_DIRECTION_COUNT = 9


@ti.func
def lattice_direction(direction_index):
    direction = ti.Vector([0, 0], dt=ti.i32)
    if direction_index == 1:
        direction = ti.Vector([1, 0])
    elif direction_index == 2:
        direction = ti.Vector([0, 1])
    elif direction_index == 3:
        direction = ti.Vector([-1, 0])
    elif direction_index == 4:
        direction = ti.Vector([0, -1])
    elif direction_index == 5:
        direction = ti.Vector([1, 1])
    elif direction_index == 6:
        direction = ti.Vector([-1, 1])
    elif direction_index == 7:
        direction = ti.Vector([-1, -1])
    elif direction_index == 8:
        direction = ti.Vector([1, -1])
    return direction


@ti.func
def lattice_weight(direction_index):
    weight = ti.cast(4.0 / 9.0, ti.f32)
    if 1 <= direction_index <= 4:
        weight = ti.cast(1.0 / 9.0, ti.f32)
    elif direction_index >= 5:
        weight = ti.cast(1.0 / 36.0, ti.f32)
    return weight


@ti.func
def opposite_direction_index(direction_index):
    opposite = 0
    if direction_index == 1:
        opposite = 3
    elif direction_index == 2:
        opposite = 4
    elif direction_index == 3:
        opposite = 1
    elif direction_index == 4:
        opposite = 2
    elif direction_index == 5:
        opposite = 7
    elif direction_index == 6:
        opposite = 8
    elif direction_index == 7:
        opposite = 5
    elif direction_index == 8:
        opposite = 6
    return opposite


@ti.func
def momentum_equilibrium(direction_index, pressure, density_lattice, velocity_lattice):
    """Liang et al. pressure--momentum equilibrium for large density ratios.

    Its zeroth moment is zero, its first moment is ``rho * u``, and its
    second moment is ``rho * u u + pressure * I``.  Consequently material
    density remains an independent phase-field quantity rather than an
    isothermal equation-of-state pressure.
    """

    direction = ti.cast(lattice_direction(direction_index), ti.f32)
    projected_velocity = direction.dot(velocity_lattice)
    speed_squared = velocity_lattice.dot(velocity_lattice)
    velocity_part = (
        lattice_weight(direction_index)
        * density_lattice
        * (
            3.0 * projected_velocity
            + 4.5 * projected_velocity * projected_velocity
            - 1.5 * speed_squared
        )
    )
    pressure_part = 3.0 * lattice_weight(direction_index) * pressure
    if direction_index == 0:
        pressure_part -= 3.0 * pressure
    return pressure_part + velocity_part


@ti.func
def phase_equilibrium(direction_index, water_phase, velocity_lattice):
    direction = ti.cast(lattice_direction(direction_index), ti.f32)
    projected_velocity = direction.dot(velocity_lattice)
    speed_squared = velocity_lattice.dot(velocity_lattice)
    return (
        lattice_weight(direction_index)
        * water_phase
        * (
            1.0
            + 3.0 * projected_velocity
            + 4.5 * projected_velocity * projected_velocity
            - 1.5 * speed_squared
        )
    )


@ti.func
def inside_grid(i, j, nx, ny):
    return 0 <= i < nx and 0 <= j < ny


@ti.func
def mixture_density(water_phase, water_density, air_density):
    water_fraction = ti.min(1.0, ti.max(0.0, water_phase))
    return water_density * water_fraction + air_density * (1.0 - water_fraction)


@ti.func
def inverse_central_moment_basis(direction_component, velocity_component, moment_order):
    basis_weight = 0.0
    if direction_component == 0:
        if moment_order == 0:
            basis_weight = 1.0 - velocity_component * velocity_component
        elif moment_order == 1:
            basis_weight = -2.0 * velocity_component
        else:
            basis_weight = -1.0
    elif direction_component == 1:
        if moment_order == 0:
            basis_weight = 0.5 * velocity_component * (velocity_component + 1.0)
        elif moment_order == 1:
            basis_weight = velocity_component + 0.5
        else:
            basis_weight = 0.5
    else:
        if moment_order == 0:
            basis_weight = 0.5 * velocity_component * (velocity_component - 1.0)
        elif moment_order == 1:
            basis_weight = velocity_component - 0.5
        else:
            basis_weight = 0.5
    return basis_weight


@ti.func
def reconstruct_central_moment_population(
    direction_x,
    direction_y,
    velocity_x,
    velocity_y,
    moment_00,
    moment_10,
    moment_01,
    moment_20,
    moment_02,
    moment_11,
    moment_21,
    moment_12,
    moment_22,
):
    basis_x_0 = inverse_central_moment_basis(direction_x, velocity_x, 0)
    basis_x_1 = inverse_central_moment_basis(direction_x, velocity_x, 1)
    basis_x_2 = inverse_central_moment_basis(direction_x, velocity_x, 2)
    basis_y_0 = inverse_central_moment_basis(direction_y, velocity_y, 0)
    basis_y_1 = inverse_central_moment_basis(direction_y, velocity_y, 1)
    basis_y_2 = inverse_central_moment_basis(direction_y, velocity_y, 2)
    return (
        basis_x_0 * basis_y_0 * moment_00
        + basis_x_1 * basis_y_0 * moment_10
        + basis_x_0 * basis_y_1 * moment_01
        + basis_x_2 * basis_y_0 * moment_20
        + basis_x_0 * basis_y_2 * moment_02
        + basis_x_1 * basis_y_1 * moment_11
        + basis_x_2 * basis_y_1 * moment_21
        + basis_x_1 * basis_y_2 * moment_12
        + basis_x_2 * basis_y_2 * moment_22
    )


@ti.func
def cross_product_2d(a, b):
    return a.x * b.y - a.y * b.x
