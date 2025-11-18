Below is a suggested `PROJECT.md` that describes the testing scheme for parts (B) and (C): mixed quantum–classical equilibrium and Langevin–Ehrenfest dynamics for your two‑oscillator model. 

````markdown
# Project: Test of Thermalization in Mixed Quantum–Classical Dynamics

This project implements numerical tests of thermalization for a quantum harmonic oscillator \(x\) coupled to a classical harmonic oscillator \(y\).  

We implement two schemes:

- **(B)** A *static* mixed quantum–classical equilibrium construction.
- **(C)** A *dynamical* Langevin–Ehrenfest scheme that evolves \(y\) with Langevin dynamics and \(x\) quantum‑mechanically.

Each scheme is implemented in a separate Python file and produces an estimated equilibrium population distribution over the \(x\)‑oscillator levels.

---

## 1. Physical model

We work in units with \(\hbar = k_B = 1\) and, unless stated otherwise, \(m_x = m_y = 1\).

- Quantum degree of freedom: harmonic oscillator coordinate \(x\), represented in a truncated harmonic‑oscillator basis \(\{|n_x\rangle\}_{n_x=0}^{N_x-1}\).
- Classical degree of freedom: harmonic oscillator coordinate \(y\) with conjugate momentum \(p_y\).

Parameters:

- `omega_x` – frequency of the quantum oscillator \(x\) (fast mode).
- `omega_y` – frequency of the classical oscillator \(y\) (slow mode).
- `alpha` – strength of the bilinear coupling.
- `beta` – inverse temperature \(1/T\).

Hamiltonian:

\[
H = H_x + H_y + V_{\text{coup}},
\]

where

- Quantum part (operator on the \(x\) Hilbert space)
  \[
  H_x = \omega_x \left(a^\dagger a + \tfrac12 I\right),
  \]
- Classical part (scalar function of \(y,p_y\))
  \[
  H_y(y,p_y) = \frac{p_y^2}{2} + \frac12 \omega_y^2 y^2,
  \]
- Coupling
  \[
  V_{\text{coup}} = \alpha\,\hat x\, y,
  \]
  with position operator
  \[
  \hat x = \sqrt{\frac{1}{2\omega_x}}\,(a + a^\dagger).
  \]

Ladder operators in the truncated basis:

- \(a_{n-1,n} = \sqrt{n}\) for \(n=1,\dots,N_x-1\),
- \(a^\dagger = a^\dagger_{\text{matrix}} = a^\top\).

The observable of interest in both schemes is the equilibrium population distribution

\[
p_{n_x} = \langle n_x | \rho_x | n_x \rangle,
\]

where \(\rho_x\) is the reduced density operator of the \(x\) oscillator.

Both codes should also compute the *reference* Boltzmann distribution for the **uncoupled** \(x\) oscillator,

\[
p^{\text{bare}}_{n_x} \propto e^{-\beta \varepsilon_{n_x}},
\quad
\varepsilon_{n_x} = \omega_x\left(n_x + \tfrac12\right),
\]

for later comparison.

---

## 2. Part (B): Mixed quantum–classical equilibrium

**File:** `equilibrium_B_mqc.py`  

This script computes the mixed quantum–classical equilibrium distribution \(p^{(B)}_{n_x}\) for the \(x\) levels by integrating over the classical equilibrium distribution of \(y\).

### 2.1. Mathematical formulation

In the mixed quantum–classical picture:

\[
\rho_x \propto \iint dy\,dp_y\ e^{-\beta H_y(y,p_y)}\, e^{-\beta\left(H_x + \alpha \hat x\, y\right)}.
\]

The momentum integral factorizes and contributes only an overall constant, so we effectively need

\[
\rho_x \propto \int dy\ \exp\!\left[-\beta \tfrac12 \omega_y^2 y^2\right]\,
      e^{-\beta\left(H_x + \alpha \hat x\, y\right)}.
\]

The integral over \(y\) is evaluated numerically by Monte Carlo sampling from the Gaussian weight
\[
P(y) \propto \exp\!\left[-\beta \tfrac12 \omega_y^2 y^2\right].
\]

For each sampled \(y\), we form the effective Hamiltonian
\[
H_{\text{eff}}(y) = H_x + \alpha \hat x\, y,
\]
compute the corresponding Boltzmann operator
\[
\rho_x(y) = e^{-\beta H_{\text{eff}}(y)},
\]
and average over samples:

\[
\rho_x \approx \frac{1}{\mathcal N}
\sum_{k=1}^{N_{\text{samples}}} \rho_x(y_k),
\qquad
\mathcal N = \mathrm{Tr}\Big[\sum_k \rho_x(y_k)\Big].
\]

Then \(p^{(B)}_{n_x} = \langle n_x|\rho_x|n_x\rangle\).

### 2.2. Required functionality

The script should:

1. **Import dependencies**

   - `numpy` as the main numerical library.
   - Use `numpy.linalg.eigh` for Hermitian diagonalization.

2. **Define parameters (at top of file)**

   Include reasonable defaults, with possibility for later modification:

   ```python
   N_x = 16          # basis size for x
   omega_x = 1.0
   omega_y = 0.2
   alpha = 0.3
   beta = 1.0        # inverse temperature
   n_samples_y = 5000
   random_seed = 1234
````

3. **Construct harmonic-oscillator operators**

   * Build matrices `a`, `adag`, `I`, `H_x`, and `x_op` as specified in Section 1.
   * Functions to build these operators are recommended, e.g. `build_ho_operators(N_x, omega_x)`.

4. **Sample classical y values**

   * Set random seed for reproducibility.
   * Classical equilibrium for (y) is Gaussian with variance
     (\sigma_y^2 = 1 / (\beta \omega_y^2)).
   * Draw `n_samples_y` samples using `np.random.normal(0.0, sigma_y, size=n_samples_y)`.

5. **Accumulate the reduced density matrix**

   * Initialize `rho_total = np.zeros((N_x, N_x), dtype=complex)`.
   * For each sampled `y`:

     * Form `H_eff = H_x + alpha * y * x_op`.
     * Diagonalize: `evals, evecs = np.linalg.eigh(H_eff)`.
     * Construct `rho_y = evecs @ np.diag(np.exp(-beta * evals)) @ evecs.conj().T`.
     * Add to accumulator: `rho_total += rho_y`.
   * Normalize:

     ```python
     trace_total = np.trace(rho_total).real
     rho_x = rho_total / trace_total
     ```

6. **Compute populations and reference Boltzmann distribution**

   * `p_B = np.real(np.diag(rho_x))`.
   * Energies for bare (x) oscillator:

     ```python
     n = np.arange(N_x)
     eps_x = omega_x * (n + 0.5)
     boltz = np.exp(-beta * eps_x)
     boltz /= boltz.sum()
     ```

7. **Output**

   * Print a table to stdout with columns:
     `n_x`, `p_B_mqc`, `p_bare_boltz`.
   * Also save to a text file, e.g.:

     ```python
     data = np.column_stack([n, p_B, boltz])
     header = "n_x p_B_mqc p_bare_boltz"
     np.savetxt("distribution_B.txt", data, header=header)
     ```

---

## 3. Part (C): Langevin–Ehrenfest dynamics

**File:** `dynamics_C_langevin_ehrenfest.py`

This script simulates coupled dynamics where:

* The classical coordinate (y(t)) obeys a Langevin equation.
* The quantum state (|\psi(t)\rangle) of the (x) oscillator evolves under the time‑dependent Hamiltonian (H_{\text{eff}}(y(t)) = H_x + \alpha \hat x, y(t)) (Ehrenfest coupling).

Time‑averaging over long trajectories (and, optionally, multiple stochastic realizations) yields an empirical equilibrium distribution (p^{(C)}_{n_x}).

### 3.1. Equations of motion

Langevin equation for (y(t)) and velocity (v(t) = \dot y) (with (m_y = 1)):

[
\dot y = v,
]
[
\dot v = -\omega_y^2 y - \alpha \langle \hat x \rangle_t - \gamma v + \eta(t),
]

where:

* (\gamma) is a friction coefficient.
* (\eta(t)) is Gaussian white noise with
  (\langle \eta(t)\rangle = 0),
  (\langle \eta(t)\eta(t')\rangle = 2 \gamma \beta^{-1}\delta(t-t')).

Quantum evolution for (|\psi(t)\rangle):

[
i,\frac{d}{dt}|\psi(t)\rangle = H_{\text{eff}}(y(t)),|\psi(t)\rangle.
]

The expectation value entering the force is

[
\langle \hat x \rangle_t = \langle \psi(t) | \hat x | \psi(t) \rangle.
]

### 3.2. Numerical scheme

We use a discrete time step `dt` and Euler–Maruyama integration for the Langevin equation, combined with a unitary propagator for the quantum state.

For each time step:

1. Compute `H_eff(y)` and construct the unitary propagator
   [
   U(y) = e^{-i H_{\text{eff}}(y) \Delta t}.
   ]
   Implement via spectral decomposition using `numpy.linalg.eigh`:

   * `evals, evecs = np.linalg.eigh(H_eff)`
   * `U = evecs @ np.diag(np.exp(-1j * evals * dt)) @ evecs.conj().T`

2. Update the wavefunction:

   * `psi = U @ psi`
   * Normalize: `psi /= np.linalg.norm(psi)`

3. Compute expectation value of (x):

   * `x_expect = np.real(np.vdot(psi, x_op @ psi))`

4. Update the classical variables (Euler–Maruyama):

   ```python
   noise = np.random.normal(0.0, 1.0)
   sigma_v = np.sqrt(2.0 * gamma / beta * dt)

   v += dt * (-omega_y**2 * y - alpha * x_expect - gamma * v) + sigma_v * noise
   y += dt * v
   ```

5. After an initial burn‑in time, accumulate populations
   (p_{n_x}(t) = |\langle n_x | \psi(t)\rangle|^2).

### 3.3. Required functionality

The script should:

1. **Import dependencies**

   * `numpy` (and `numpy.linalg.eigh`).

2. **Define parameters**

   Example defaults:

   ```python
   N_x = 16
   omega_x = 1.0
   omega_y = 0.2
   alpha = 0.3
   beta = 1.0

   gamma = 0.5          # friction coefficient
   dt = 0.01            # time step
   n_steps = 50000      # total number of time steps per trajectory
   burn_in_steps = 5000 # steps to discard as transient
   n_trajectories = 20  # number of independent Langevin runs
   random_seed = 4321
   ```

3. **Construct harmonic-oscillator operators**

   As in Part (B): build `H_x` and `x_op` in the truncated basis.

4. **Main simulation loop**

   * Set global random seed.
   * Initialize an array `pop_accum = np.zeros(N_x)`.

   For each trajectory:

   * Initialize `y = 0.0`, `v = 0.0`.
   * Initialize `psi` in the ground state of the bare (x) oscillator, e.g.

     ```python
     psi = np.zeros(N_x, dtype=complex)
     psi[0] = 1.0
     ```
   * Loop over `step` from `0` to `n_steps - 1`:

     * Perform the five update substeps outlined in Section 3.2.
     * For `step >= burn_in_steps`, accumulate the level populations:

       ```python
       pop_accum += np.abs(psi)**2
       ```

5. **Normalize and compute final populations**

   * Total number of samples contributing:

     ```python
     n_effective = (n_steps - burn_in_steps) * n_trajectories
     populations_C = pop_accum / n_effective
     # Enforce exact normalization:
     populations_C /= populations_C.sum()
     ```
   * Compute bare Boltzmann distribution `boltz` as in Part (B).

6. **Output**

   * Print a table with columns: `n_x`, `p_C_dyn`, `p_bare_boltz`.
   * Save to file, e.g.

     ```python
     n = np.arange(N_x)
     data = np.column_stack([n, populations_C, boltz])
     header = "n_x p_C_dyn p_bare_boltz"
     np.savetxt("distribution_C.txt", data, header=header)
     ```

---

## 4. Comparison and diagnostics

The two scripts are independent and can be run separately:

```bash
python equilibrium_B_mqc.py      # produces distribution_B.txt
python dynamics_C_langevin_ehrenfest.py  # produces distribution_C.txt
```

For analysis, one can compare:

* The static mixed equilibrium populations `p_B_mqc` (from Part B),
* The dynamical populations `p_C_dyn` (from Part C),
* The bare Boltzmann distribution of the uncoupled (x) oscillator.

Agreement between `p_B_mqc` and `p_C_dyn` tests whether the Langevin–Ehrenfest dynamics correctly samples the mixed quantum–classical equilibrium. Deviations from the bare Boltzmann distribution reveal the effect of coupling to the classical degree of freedom.

```
```
