# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from libc.math cimport exp, fabs, log
from libc.stdint cimport uint64_t
from libc.stdlib cimport free, malloc, qsort

import numpy as np


ctypedef struct Bond:
    double weight
    Py_ssize_t id1
    Py_ssize_t id2
    Py_ssize_t order


cdef inline uint64_t _rotl(uint64_t x, int k) noexcept:
    return (x << k) | (x >> (64 - k))


cdef inline double _random_number(uint64_t[::1] s):
    cdef int R = 23
    cdef int A = 17
    cdef int B = 45
    cdef uint64_t result_int = _rotl(s[0] + s[3], R) + s[0]
    cdef uint64_t t = s[1] << A

    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]

    s[2] ^= t
    s[3] = _rotl(s[3], B)

    return (result_int >> 11) * (1.0 / 9007199254740992.0)


cdef inline int _sign(double value) noexcept:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


cdef int _compare_bonds(const void *left, const void *right) noexcept nogil:
    cdef Bond *a = <Bond *>left
    cdef Bond *b = <Bond *>right

    if a.weight > b.weight:
        return -1
    if a.weight < b.weight:
        return 1
    if a.order < b.order:
        return -1
    if a.order > b.order:
        return 1
    return 0


cdef Py_ssize_t _uf_find(
    Py_ssize_t i,
    Py_ssize_t *parent,
    Py_ssize_t *dx,
    Py_ssize_t *dy,
):
    cdef Py_ssize_t p
    cdef Py_ssize_t root

    if parent[i] == i:
        return i

    p = parent[i]
    root = _uf_find(p, parent, dx, dy)

    dx[i] += dx[p]
    dy[i] += dy[p]
    parent[i] = root

    return root


cdef bint _uf_union(
    Py_ssize_t N,
    Py_ssize_t i,
    Py_ssize_t j,
    Py_ssize_t *parent,
    Py_ssize_t *rank,
    Py_ssize_t *dx,
    Py_ssize_t *dy,
):
    cdef Py_ssize_t root_i = _uf_find(i, parent, dx, dy)
    cdef Py_ssize_t root_j = _uf_find(j, parent, dx, dy)
    cdef Py_ssize_t xi = i % N
    cdef Py_ssize_t yi = i // N
    cdef Py_ssize_t xj = j % N
    cdef Py_ssize_t yj = j // N
    cdef Py_ssize_t bdx = xj - xi
    cdef Py_ssize_t bdy = yj - yi
    cdef Py_ssize_t temp

    if bdx == N - 1:
        bdx = -1
    elif bdx == -(N - 1):
        bdx = 1

    if bdy == N - 1:
        bdy = -1
    elif bdy == -(N - 1):
        bdy = 1

    if root_i == root_j:
        if -dx[i] + bdx + dx[j] != 0 or -dy[i] + bdy + dy[j] != 0:
            return True
        return False

    if rank[root_i] < rank[root_j]:
        temp = root_i
        root_i = root_j
        root_j = temp

        temp = i
        i = j
        j = temp

        bdx = -bdx
        bdy = -bdy

    parent[root_j] = root_i

    if rank[root_i] == rank[root_j]:
        rank[root_i] += 1

    dx[root_j] = -dx[j] - bdx + dx[i]
    dy[root_j] = -dy[j] - bdy + dy[i]

    return False


cpdef tuple swedson_wang_phi4(double[:, ::1] lattice, Py_ssize_t N, uint64_t[::1] s):
    cdef Py_ssize_t total_spins = N * N
    cdef Py_ssize_t max_bonds = 2 * total_spins
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t spin_id
    cdef Py_ssize_t row_offset
    cdef Py_ssize_t down_i
    cdef Py_ssize_t down_offset
    cdef Py_ssize_t right_j
    cdef Py_ssize_t bond_count = 0
    cdef Py_ssize_t bond_index
    cdef Py_ssize_t root
    cdef double spin_here
    cdef double spin_right
    cdef double spin_down
    cdef double u
    cdef double J_ij
    cdef double T_req
    cdef double T_eff = float("inf")
    cdef double value
    cdef int sign_here
    cdef Py_ssize_t *parent = NULL
    cdef Py_ssize_t *rank = NULL
    cdef Py_ssize_t *dx = NULL
    cdef Py_ssize_t *dy = NULL
    cdef int *root_seen = NULL
    cdef double *root_spins = NULL
    cdef Bond *bonds = NULL

    parent = <Py_ssize_t *>malloc(total_spins * sizeof(Py_ssize_t))
    rank = <Py_ssize_t *>malloc(total_spins * sizeof(Py_ssize_t))
    dx = <Py_ssize_t *>malloc(total_spins * sizeof(Py_ssize_t))
    dy = <Py_ssize_t *>malloc(total_spins * sizeof(Py_ssize_t))
    root_seen = <int *>malloc(total_spins * sizeof(int))
    root_spins = <double *>malloc(total_spins * sizeof(double))
    bonds = <Bond *>malloc(max_bonds * sizeof(Bond))

    if (
        parent == NULL or rank == NULL or dx == NULL or dy == NULL or
        root_seen == NULL or root_spins == NULL or bonds == NULL
    ):
        free(parent)
        free(rank)
        free(dx)
        free(dy)
        free(root_seen)
        free(root_spins)
        free(bonds)
        raise MemoryError("could not allocate invaded-cluster work arrays")

    try:
        for spin_id in range(total_spins):
            parent[spin_id] = spin_id
            rank[spin_id] = 0
            dx[spin_id] = 0
            dy[spin_id] = 0
            root_seen[spin_id] = 0
            root_spins[spin_id] = 0.0

        for i in range(N):
            row_offset = i * N
            down_i = i + 1
            if down_i == N:
                down_i = 0
            down_offset = down_i * N

            for j in range(N):
                spin_here = lattice[i, j]
                sign_here = _sign(spin_here)
                spin_id = row_offset + j

                right_j = j + 1
                if right_j == N:
                    right_j = 0
                spin_right = lattice[i, right_j]

                if sign_here == _sign(spin_right):
                    u = _random_number(s)
                    if u == 0.0:
                        u = 1e-10
                    if u == 1.0:
                        u = 0.9999999999

                    J_ij = fabs(spin_here * spin_right)
                    T_req = (2.0 * J_ij) / -log(1.0 - u)

                    bonds[bond_count].weight = T_req
                    bonds[bond_count].id1 = spin_id
                    bonds[bond_count].id2 = row_offset + right_j
                    bonds[bond_count].order = bond_count
                    bond_count += 1

                spin_down = lattice[down_i, j]

                if sign_here == _sign(spin_down):
                    u = _random_number(s)
                    if u == 0.0:
                        u = 1e-10
                    if u == 1.0:
                        u = 0.9999999999

                    J_ij = fabs(spin_here * spin_down)
                    T_req = (2.0 * J_ij) / -log(1.0 - u)

                    bonds[bond_count].weight = T_req
                    bonds[bond_count].id1 = spin_id
                    bonds[bond_count].id2 = down_offset + j
                    bonds[bond_count].order = bond_count
                    bond_count += 1

        qsort(bonds, <size_t>bond_count, sizeof(Bond), _compare_bonds)

        for bond_index in range(bond_count):
            if _uf_union(
                N,
                bonds[bond_index].id1,
                bonds[bond_index].id2,
                parent,
                rank,
                dx,
                dy,
            ):
                T_eff = bonds[bond_index].weight
                break

        for spin_id in range(total_spins):
            root = _uf_find(spin_id, parent, dx, dy)

            if root_seen[root] == 0:
                root_seen[root] = 1
                if _random_number(s) < 0.5:
                    root_spins[root] = 1.0
                else:
                    root_spins[root] = -1.0

            i = spin_id // N
            j = spin_id - i * N
            value = lattice[i, j]
            if value < 0.0:
                value = -value
            lattice[i, j] = value * root_spins[root]

        return np.asarray(lattice), T_eff
    finally:
        free(parent)
        free(rank)
        free(dx)
        free(dy)
        free(root_seen)
        free(root_spins)
        free(bonds)


cpdef object metropolis_phi4(
    double[:, ::1] lattice,
    Py_ssize_t N,
    uint64_t[::1] s,
    int sweeps,
    double lambda_L,
    double mu_sq,
):
    cdef int oliver
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t next_i
    cdef Py_ssize_t prev_i
    cdef Py_ssize_t next_j
    cdef Py_ssize_t prev_j
    cdef double phi_old
    cdef double phi_new
    cdef double sum_neighbors
    cdef double term1
    cdef double term2
    cdef double term3
    cdef double delta_S
    cdef double mass_coeff = 2.0 + 0.5 * mu_sq
    cdef double quartic_coeff = lambda_L / 4.0
    cdef double old2
    cdef double new2

    for oliver in range(sweeps):
        for i in range(N):
            next_i = i + 1
            if next_i == N:
                next_i = 0
            prev_i = i - 1
            if prev_i < 0:
                prev_i = N - 1

            for j in range(N):
                phi_old = lattice[i, j]
                phi_new = phi_old + (_random_number(s) * 3.0 - 1.5)

                next_j = j + 1
                if next_j == N:
                    next_j = 0
                prev_j = j - 1
                if prev_j < 0:
                    prev_j = N - 1

                sum_neighbors = (
                    lattice[next_i, j] + lattice[prev_i, j] +
                    lattice[i, next_j] + lattice[i, prev_j]
                )
                old2 = phi_old * phi_old
                new2 = phi_new * phi_new
                term1 = (phi_new - phi_old) * sum_neighbors
                term2 = mass_coeff * (old2 - new2)
                term3 = quartic_coeff * ((old2 * old2) - (new2 * new2))
                delta_S = term1 + term2 + term3

                if delta_S >= 0.0:
                    lattice[i, j] = phi_new
                else:
                    if _random_number(s) < exp(delta_S):
                        lattice[i, j] = phi_new

    return np.asarray(lattice)


cpdef tuple invaded_cluster_phi4(
    double[:, ::1] lattice,
    Py_ssize_t N,
    uint64_t[::1] s,
    int total_steps,
    double lambda_L,
    double mu_sq_init,
    double gamma,
):
    cdef double mu_sq = mu_sq_init
    cdef double T_eff
    cdef double deviation
    cdef int step
    cdef int progress_interval = total_steps // 10
    cdef list mu_history = []
    cdef list teff_history = []

    if progress_interval < 1:
        progress_interval = 1

    for step in range(total_steps):
        _, T_eff = swedson_wang_phi4(lattice, N, s)

        deviation = T_eff - 1.0
        if deviation > 2.0:
            deviation = 2.0
        elif deviation < -2.0:
            deviation = -2.0

        mu_sq = mu_sq + gamma * deviation

        metropolis_phi4(lattice, N, s, sweeps=5, lambda_L=lambda_L, mu_sq=mu_sq)

        mu_history.append(mu_sq)
        teff_history.append(T_eff)

        if step > 0 and step % progress_interval == 0:
            print(f"Step {step}: T_eff = {T_eff:.4f}, mu_sq = {mu_sq:.4f}")

    return np.asarray(lattice), mu_history, teff_history
