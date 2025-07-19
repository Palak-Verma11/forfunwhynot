# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:34:41 2025

@author: v
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy.stats import unitary_group
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

start = time.time()

# Numba-optimized functions
@jit(nopython=True, cache=True)
def reshape_state(psi, NA, NB):
    """Reshape state vector for bipartition"""
    return psi.reshape(NA, NB)

@jit(nopython=True, cache=True)
def compute_reduced_density_matrix_A(psi_reshaped):
    """Compute reduced density matrix for subsystem A"""
    return psi_reshaped @ psi_reshaped.conj().T

@jit(nopython=True, cache=True)
def von_neumann_entropy_fast(rho):
    """Fast computation of von Neumann entropy"""
    eigenvals = np.linalg.eigvals(rho)
    eigenvals_real = np.real(eigenvals)
    
    # Filter out numerical zeros
    valid_eigenvals = eigenvals_real[eigenvals_real > 1e-12]
    
    if len(valid_eigenvals) == 0:
        return 0.0
    
    entropy = 0.0
    for val in valid_eigenvals:
        entropy -= val * np.log(val)
    
    return entropy

@jit(nopython=True, cache=True, parallel=True)
def compute_all_entropies(eigenvectors, NA, NB):
    """Compute entanglement entropies for all eigenvectors in parallel"""
    n_states = eigenvectors.shape[1]
    entropies = np.zeros(n_states, dtype=np.float64)
    
    for i in prange(n_states):
        psi = eigenvectors[:, i]
        psi_reshaped = reshape_state(psi, NA, NB)
        rho_a = compute_reduced_density_matrix_A(psi_reshaped)
        entropies[i] = von_neumann_entropy_fast(rho_a)
    
    return entropies

def build_spin_operators(s):
    """Build generalized spin operators for spin s"""
    # Dimension of the spin space
    d = int(2*s + 1)
    
    # Create raising and lowering operators
    m_values = np.arange(-s, s+1)
    
    # S_z operator (diagonal)
    sz = np.diag(m_values).astype(np.complex128)
    
    # S_+ operator (raising operator)
    sp = np.zeros((d, d), dtype=np.complex128)
    for i in range(d-1):
        m = m_values[i]
        sp[i+1, i] = np.sqrt(s*(s+1) - m*(m+1))
    
    # S_- operator (lowering operator)
    sm = sp.conj().T
    
    # S_x operator
    sx = 0.5 * (sp + sm)
    
    # S_y operator
    sy = -0.5j * (sp - sm)
    
    return sx, sy, sz

def build_identity_chain(spin_chain):
    """Build identity operator for the entire chain"""
    operators = []
    for s in spin_chain:
        d = int(2*s + 1)
        operators.append(np.eye(d, dtype=np.complex128))
    
    result = operators[0]
    for i in range(1, len(operators)):
        result = np.kron(result, operators[i])
    
    return result

def build_single_site_operator(spin_chain, site_idx, local_op):
    """Build operator that acts on a single site with identity on all others"""
    operators = []
    for i, s in enumerate(spin_chain):
        d = int(2*s + 1)
        if i == site_idx:
            operators.append(local_op)
        else:
            operators.append(np.eye(d, dtype=np.complex128))
    
    result = operators[0]
    for i in range(1, len(operators)):
        result = np.kron(result, operators[i])
    
    return result

def build_two_site_operator(spin_chain, site1_idx, site2_idx, op1, op2):
    """Build operator that acts on two sites with identity on all others"""
    operators = []
    for i, s in enumerate(spin_chain):
        d = int(2*s + 1)
        if i == site1_idx:
            operators.append(op1)
        elif i == site2_idx:
            operators.append(op2)
        else:
            operators.append(np.eye(d, dtype=np.complex128))
    
    result = operators[0]
    for i in range(1, len(operators)):
        result = np.kron(result, operators[i])
    
    return result

def calculate_bipartition_dimensions(spin_chain, LA):
    """Calculate dimensions of subsystems A and B for bipartition"""
    L = len(spin_chain)
    LB = L - LA
    
    # Dimension of subsystem A
    NA = 1
    for i in range(LA):
        d = int(2*spin_chain[i] + 1)
        NA *= d
    
    # Dimension of subsystem B
    NB = 1
    for i in range(LA, L):
        d = int(2*spin_chain[i] + 1)
        NB *= d
    
    return NA, NB

def build_hamiltonian_kfim(spin_chain, J, b, h):
    """Build KFIM Hamiltonian for arbitrary spin chain"""
    L = len(spin_chain)
    
    # Calculate total dimension
    total_dim = 1
    for s in spin_chain:
        total_dim *= int(2*s + 1)
    
    # Build Hz (ZZ interactions + longitudinal fields)
    Hz = np.zeros((total_dim, total_dim), dtype=np.complex128)
    
    # ZZ nearest neighbor interactions (periodic boundary conditions)
    for j in range(L):
        j_next = (j + 1) % L
        
        # Get spin operators for sites j and j+1
        sx_j, sy_j, sz_j = build_spin_operators(spin_chain[j])
        sx_next, sy_next, sz_next = build_spin_operators(spin_chain[j_next])
        
        # Build ZZ interaction term
        zz_term = build_two_site_operator(spin_chain, j, j_next, sz_j, sz_next)
        Hz += J * zz_term
    
    # Longitudinal magnetic fields
    for j in range(L):
        sx_j, sy_j, sz_j = build_spin_operators(spin_chain[j])
        z_term = build_single_site_operator(spin_chain, j, sz_j)
        Hz += h[j] * z_term
    
    # Build Hx (transverse field)
    Hx = np.zeros((total_dim, total_dim), dtype=np.complex128)
    
    # Transverse magnetic field
    for j in range(L):
        sx_j, sy_j, sz_j = build_spin_operators(spin_chain[j])
        x_term = build_single_site_operator(spin_chain, j, sx_j)
        Hx += b * x_term
    
    return Hz, Hx

def COE(N):
    """Generate a random matrix from the Circular Orthogonal Ensemble (COE)"""
    # Generate Haar-random unitary matrix
    U = unitary_group.rvs(N)
    # COE matrix: U * U^T is symmetric and unitary
    M = U @ U.T
    return M

def process_system(spin_chain, J, b, h, use_kfim=True):
    """Process a single system configuration"""
    # Calculate total dimension
    total_dim = 1
    for s in spin_chain:
        total_dim *= int(2*s + 1)
    
    if use_kfim:
        # Build KFIM Hamiltonian
        Hz, Hx = build_hamiltonian_kfim(spin_chain, J, b, h)
        
        # Floquet operator U = exp(-iHz) * exp(-iHx)
        U = expm(-1j * Hz) @ expm(-1j * Hx)
    else:
        # COE case
        U = COE(total_dim)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(U)
    
    # Sort by quasi-energy (phase of eigenvalues)
    quasienergies = np.angle(eigenvalues)
    sorted_indices = np.argsort(quasienergies)
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Compute entanglement entropies for bipartition
    L = len(spin_chain)
    LA = L // 2
    NA, NB = calculate_bipartition_dimensions(spin_chain, LA)
    
    entropies = compute_all_entropies(eigenvectors, NA, NB)
    
    return np.mean(entropies), np.std(entropies)

def create_spin_chain(L, spin_values):
    """Create a spin chain with specified spin values"""
    if isinstance(spin_values, (int, float)):
        # Uniform chain
        return [spin_values] * L
    elif isinstance(spin_values, list):
        if len(spin_values) == L:
            # Exact specification
            return spin_values
        else:
            # Repeat pattern to fill chain
            return [spin_values[i % len(spin_values)] for i in range(L)]
    else:
        raise ValueError("spin_values must be a number or list of numbers")

def print_chain_info(spin_chain):
    """Print information about the spin chain"""
    L = len(spin_chain)
    total_dim = 1
    for s in spin_chain:
        total_dim *= int(2*s + 1)
    
    print(f"Chain length: L = {L}")
    print(f"Spin values: {spin_chain}")
    print(f"Local dimensions: {[int(2*s + 1) for s in spin_chain]}")
    print(f"Total Hilbert space dimension: {total_dim}")

"""
===============================================================================
===============================================================================
===============================================================================
===============================================================================
===============================================================================
===============================================================================
===============================================================================
"""

# Main computation parameters
L_max = 7  # Maximum chain length
n_realizations = 60  # Number of disorder realizations per system size

# Define spin chain - Examples:
# For uniform spin-1/2 chain: spin_pattern = 0.5
# For uniform spin-1 chain: spin_pattern = 1
# For alternating spin-1/2 and spin-1: spin_pattern = [0.5, 1]
# For specific pattern: spin_pattern = [0.5, 1, 1.5, 1, 0.5]

spin_pattern = [1/2, 1.5, 1/2, 1.5, 1/2, 1.5, 1/2, 1.5]  # Example: alternating pattern

# Parameters from paper
J = np.pi/4  # Self-dual point
b = np.pi/4  # Self-dual point  
h_mean = 0.6  # Mean longitudinal field
h_std = np.pi/4  # Standard deviation of longitudinal field

print("Starting computation...")
print("="*60)

# Storage for results
L_values = list(range(2, L_max + 1))
coe_means = []
coe_stds = []
kfim_means = []
kfim_stds = []

# Process each system size
for L in L_values:
    spin_chain = create_spin_chain(L, spin_pattern)
    
    print(f"\nProcessing L = {L}:")
    print_chain_info(spin_chain)
    
    # Check if system is too large
    total_dim = 1
    for s in spin_chain:
        total_dim *= int(2*s + 1)
    
    
    # Storage for this system size
    coe_entropy_means = []
    coe_entropy_stds = []
    kfim_entropy_means = []
    kfim_entropy_stds = []
    
    # Multiple disorder realizations
    for realization in range(n_realizations):
        # Generate random longitudinal fields
        np.random.seed(realization + 1000)
        h = np.random.normal(h_mean, h_std, size=L)
        
        # COE system
        np.random.seed(realization + 2000)
        mean_entropy_coe, std_entropy_coe = process_system(spin_chain, J, b, h, use_kfim=False)
        coe_entropy_means.append(mean_entropy_coe)
        coe_entropy_stds.append(std_entropy_coe)
        
        # KFIM system
        mean_entropy_kfim, std_entropy_kfim = process_system(spin_chain, J, b, h, use_kfim=True)
        kfim_entropy_means.append(mean_entropy_kfim)
        kfim_entropy_stds.append(std_entropy_kfim)
    
    # Average over disorder realizations
    coe_means.append(np.mean(coe_entropy_means))
    coe_stds.append(np.mean(coe_entropy_stds))
    kfim_means.append(np.mean(kfim_entropy_means))
    kfim_stds.append(np.mean(kfim_entropy_stds))
    
    print(f"  COE:  Mean = {coe_means[-1]:.4f}, Std = {coe_stds[-1]:.4f}")
    print(f"  KFIM: Mean = {kfim_means[-1]:.4f}, Std = {kfim_stds[-1]:.4f}")

end = time.time()
print(f"\nExecution time: {end - start:.2f} seconds")

# Only analyze if we have results
if coe_means:
    # Analysis following paper's methodology
    mean_differences = [abs(c - k) for c, k in zip(coe_means, kfim_means)]
    std_differences = [abs(c - k) for c, k in zip(coe_stds, kfim_stds)]
    
    # Ratio R = |mean_diff| / std_deviation
    ratios = []
    for i in range(len(mean_differences)):
        if kfim_stds[i] > 0:
            ratios.append(mean_differences[i] / kfim_stds[i])
        else:
            ratios.append(0)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Spin pattern: {spin_pattern}")
    print(f"{'L':<4} {'COE_mean':<10} {'KFIM_mean':<10} {'Mean_diff':<10} {'Std_diff':<10} {'Ratio':<10}")
    print("-"*60)
    processed_L = L_values[:len(coe_means)]
    for i, L in enumerate(processed_L):
        print(f"{L:<4} {coe_means[i]:<10.4f} {kfim_means[i]:<10.4f} {mean_differences[i]:<10.4f} {std_differences[i]:<10.4f} {ratios[i]:<10.4f}")
    
    # Plotting function
    def create_plot(x_values, y_values, title, ylabel, color='blue'):
        """Create a scatter plot"""
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values, c=color, s=50, alpha=0.7)
        plt.plot(x_values, y_values, '--', color=color, alpha=0.5)
        plt.xlabel('Chain Length (L)')
        plt.ylabel(ylabel)
        plt.title(f"{title}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Create plots
    create_plot(processed_L, mean_differences, 
               "Mean Entanglement Entropy Differences |COE - KFIM|", 
               "Mean Difference", 'red')
    
    create_plot(processed_L, std_differences, 
               "Standard Deviation Differences |COE - KFIM|", 
               "Std Difference", 'green')
    
    create_plot(processed_L, ratios, 
               "Ratio R = Mean Difference / Standard Deviation", 
               "Ratio R", 'blue')

print("\nNote: The paper's main finding is that while average entanglement")
print("approaches COE predictions, the distributions remain distinct.")
print("This is captured by the ratio R, which should approach a constant > 1")
print("rather than decreasing to zero if distributions were converging.")

print("\n" + "="*60)
print("SPIN CHAIN EXAMPLES:")
print("="*60)
print("To change the spin pattern, modify the 'spin_pattern' variable:")
print("  - Uniform spin-1/2: spin_pattern = 0.5")
print("  - Uniform spin-1: spin_pattern = 1")
print("  - Alternating 1/2-1: spin_pattern = [0.5, 1]")
print("  - Complex pattern: spin_pattern = [0.5, 1, 1.5, 1, 0.5]")
print("  - Current pattern:", spin_pattern)