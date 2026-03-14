from math import exp
import numpy as np
import math

# Import the Cython-compiled random number generator
from rng import random_number

class UnionFindIsingPBC:
    def __init__(self, N):
        # grid dimension and number of spins
        self.N = N
        total_spins = N * N

        # every spin starts as its own root
        # this makes an array of all the spins
        self.parent = np.arange(total_spins)
        # rank keeps the trees completely flat for speed
        # array of zeros
        self.rank = np.zeros(total_spins, dtype=int)

        # track distance to the root
        # this moves the 1d back to the 2d lattice vectors
        # note right now each spin is a cluster of one (distance is 0)
        self.dx = np.zeros(total_spins, dtype=int)
        self.dy = np.zeros(total_spins, dtype=int)

    def find(self, i):
        # this is the easy it is its own parent check
        if self.parent[i] == i:
            return i

        # recursive search to find the ultimate parent
        p = self.parent[i]
        root = self.find(p)

        # self.parent will keep updating using the find function
        # points every node directly to the root AND updates its physical distance
        self.dx[i] += self.dx[p]
        self.dy[i] += self.dy[p]
        self.parent[i] = root

        return root

    def union(self, i, j):
        # finds the parent root for i and j
        root_i = self.find(i)
        root_j = self.find(j)

        # Calculate physical bond distance, accounting for wrap-around
        xi, yi = i % self.N, i // self.N
        xj, yj = j % self.N, j // self.N

        bdx = xj - xi
        if bdx == self.N - 1: bdx = -1
        elif bdx == -(self.N - 1): bdx = 1

        bdy = yj - yi
        if bdy == self.N - 1: bdy = -1
        elif bdy == -(self.N - 1): bdy = 1

        # if they are already in the same cluster do nothing
        # EXCEPT check if this union caused percolation
        if root_i == root_j:
            cycle_x = -self.dx[i] + bdx + self.dx[j]
            cycle_y = -self.dy[i] + bdy + self.dy[j]
            if cycle_x != 0 or cycle_y != 0:
                return True # percolation
            return False

        # attach the smaller tree to the root of the larger tree
        # this is making it so that every root_i is always the larger tree
        if self.rank[root_i] < self.rank[root_j]:
            root_i, root_j = root_j, root_i
            i, j = j, i
            bdx, bdy = -bdx, -bdy # reverse the bond direction because we swapped

        # make the parent of the tree connect back to the larger tree now
        self.parent[root_j] = root_i

        # when they are the same height just add one to the rank
        if self.rank[root_i] == self.rank[root_j]:
            self.rank[root_i] += 1

        # update the distance vectors of the new master root
        # this accounts for the new cluster sizes (replacing the bounding box)
        self.dx[root_j] = -self.dx[j] - bdx + self.dx[i]
        self.dy[root_j] = -self.dy[j] - bdy + self.dy[i]

        return False

def swedson_wang_phi4(lattice, N, s):
    # call on the new cluster maker
    uf = UnionFindIsingPBC(N)
    bonds = []

    # collect all parallel neighbors
    # only needs to check right and down logically to avoid double-counting bonds
    for i in range(N):
        for j in range(N):
            # Pre-extract the current spin and its sign for speed
            spin_here = lattice[i, j]
            sign_here = np.sign(spin_here)

            # check right neighbor
            right_j = (j + 1) % N # the coordinate
            spin_right = lattice[i, right_j]

            if sign_here == np.sign(spin_right): # bc continuous space now its sign related
                u = random_number(s)
                if u == 0: u = 1e-10 # protect against log(0)
                if u == 1.0: u = 0.9999999999 # protect against log(0) from 1.0 - u

                # calculate J_ij and T_req
                # uses the relationship for normal bond creation and solves for T
                J_ij = abs(spin_here * spin_right)
                T_req = (2.0 * J_ij) / -math.log(1.0 - u)

                bonds.append((T_req, i, j, i, right_j)) # tuple added with T_req instead of random number

            # check down neighbor
            down_i = (i + 1) % N
            spin_down = lattice[down_i, j]

            if sign_here == np.sign(spin_down):
                u = random_number(s)
                if u == 0: u = 1e-10
                if u == 1.0: u = 0.9999999999

                J_ij = abs(spin_here * spin_down)
                T_req = (2.0 * J_ij) / -math.log(1.0 - u)

                bonds.append((T_req, i, j, down_i, j))

    # sort bonds from highest to lowest weight (temperature)
    bonds.sort(key=lambda x: x[0], reverse=True) # want highest T_req bonds made first

    # the new metric
    T_eff = float('inf') # Replaced p_crit to represent the effective temperature

    # bond until percolation
    for weight, i1, j1, i2, j2 in bonds:
        # flattens the 2D coordinates into 1D IDs
        id1 = i1 * N + j1
        id2 = i2 * N + j2

        # checks the percolation by using union
        percolated = uf.union(id1, id2)
        if percolated:
            # setting it equal to the T_req of the bond that caused wrap-around
            T_eff = weight
            break

    # flip the clusters
    root_spins = {} # to save the parent node sign
    for i in range(N):
        for j in range(N):
            # finds the parent of the new indice
            spin_id = i * N + j
            root = uf.find(spin_id)

            if root not in root_spins:
                root_spins[root] = 1 if random_number(s) < 0.5 else -1

            # does some flips to match the parent
            lattice[i, j] = abs(lattice[i, j]) * root_spins[root]

    return lattice, T_eff

def metropolis_phi4(lattice, N, s, sweeps, lambda_L, mu_sq):
    # sweeps the lattice
    # barred to the number of sweeps
    for oliver in range(sweeps):
      for i in range(N):
          for j in range(N):
            # extract current spin
            phi_old = lattice[i, j]

            # Propose a new value within the localized [-1.5, 1.5] window
            phi_new = phi_old + (random_number(s) * 3.0 - 1.5)

            # sum neighbors
            sum_neighbors = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] +
                               lattice[i, (j+1)%N] + lattice[i, (j-1)%N])
            term1 = (phi_new - phi_old) * sum_neighbors
            term2 = (2.0 + 0.5 * mu_sq) * ((phi_old ** 2) - (phi_new ** 2))
            term3 = (lambda_L / 4.0) * ((phi_old ** 4) - (phi_new ** 4))
            delta_S = term1 + term2 + term3

            if delta_S >= 0.0:
                lattice[i, j] = phi_new
            else:
              if random_number(s) < math.exp(delta_S):
                     lattice[i, j] = phi_new

    return lattice

def invaded_cluster_phi4(lattice, N, s, total_steps, lambda_L, mu_sq_init, gamma):
    # initialize the tuning parameter
    mu_sq = mu_sq_init

    # tracking for analysis
    mu_history = []
    teff_history = []

    # local binding for speed
    mu_append = mu_history.append
    teff_append = teff_history.append

    for step in range(total_steps):
        # cluster update and teff measurement
        lattice_after_sw, T_eff = swedson_wang_phi4(lattice, N, s)
        lattice = lattice_after_sw

        # feedback loop
        deviation = T_eff - 1.0
        deviation = max(min(deviation, 2.0), -2.0)

        # traditional gradient loss
        mu_sq = mu_sq + gamma * deviation

        # metropolis
        lattice = metropolis_phi4(lattice, N, s, sweeps=5, lambda_L=lambda_L, mu_sq=mu_sq)

        # record the data
        mu_append(mu_sq)
        teff_append(T_eff)

        # progress check
        if step > 0 and step % (total_steps // 10) == 0:
            print(f"Step {step}: T_eff = {T_eff:.4f}, mu_sq = {mu_sq:.4f}")

    return lattice, mu_history, teff_history
