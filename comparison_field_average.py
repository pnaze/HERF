import numpy as np
import matplotlib.pyplot as plt

from scipy.special import (
    ellipj, ellipk, beta as beta_function,
    gamma, roots_legendre
)

# ==========================================================
# Parameters
# ==========================================================

beta = 1.0
lambda0 = 1.0

t = np.linspace(0, 15, 601)

C = 7 / (48 * beta * lambda0**2)

# ==========================================================
# Hamiltonian field prediction
# ==========================================================

def hamiltonian_fields(t, M=12, Nphi=8192, NE=1000):

    # Universal Fourier coefficients
    phi = 2 * np.pi * np.arange(Nphi) / Nphi

    K = ellipk(0.5)

    _, y, _, _ = ellipj(
        2 * K * phi / np.pi, 0.5
    )

    a = np.fft.rfft(y**4).real / Nphi

    # Dimensionless frequency and time
    Omega = (
        2 * np.sqrt(2) * np.pi
        / beta_function(0.25, 0.5)
    )

    z = (4 * lambda0 / beta)**0.25 * t

    # Gauss-Legendre integration after x = u^4
    nodes, weights = roots_legendre(NE)

    u = 1.5 * (nodes + 1)

    weights = (
        1.5 * weights
        * 4 * u**10 * np.exp(-u**4)
    )

    psi = np.zeros_like(t)

    for m in range(1, M + 1):

        integral = (
            np.cos(
                m * Omega * z[:, None] * u[None, :]
            ) @ weights
        )

        psi += a[2*m]**2 * integral

    psi *= 2 / (
        beta * lambda0**2 * gamma(0.75)
    )

    return psi


# ==========================================================
# Direct canonical average
# ==========================================================

def canonical_average(t, N=240, substeps=10):

    # Gauss-Legendre quadrature in microscopic phase space
    nodes, weights = roots_legendre(N)

    qmax = 4 * (beta * lambda0)**(-0.25)
    pmax = 9 / np.sqrt(beta)

    # Symmetry allows restriction to q >= 0
    q0 = qmax * (nodes + 1) / 2
    p0 = pmax * nodes

    wq = qmax * weights / 2
    wp = pmax * weights

    q, p = np.meshgrid(q0, p0, indexing="ij")

    # Initial canonical distribution
    H0 = p**2 / 2 + lambda0 * q**4 / 4

    W = (
        wq[:, None]
        * wp[None, :]
        * np.exp(-beta * H0)
    )

    W /= np.sum(W)

    # Initial generalized force
    A0 = q**4 / 4

    weighted_A0 = W * A0

    psi = np.zeros_like(t)

    dt = (t[1] - t[0]) / substeps

    for k in range(len(t)):

        # Direct canonical correlation
        A = q**4 / 4

        psi[k] = (
            beta * np.sum(weighted_A0 * A) - C
        )

        if k == len(t) - 1:
            break

        # Velocity-Verlet integration of
        # dq/dt = p
        # dp/dt = -lambda0*q^3

        for _ in range(substeps):

            p -= 0.5 * dt * lambda0 * q*q*q

            q += dt * p

            p -= 0.5 * dt * lambda0 * q*q*q

    return psi


# ==========================================================
# Numerical calculations
# ==========================================================

psi_fields = hamiltonian_fields(t)

psi_canonical = canonical_average(t)

error = np.max(
    np.abs(psi_fields - psi_canonical)
)

print("Hamiltonian fields at t=0:",
      psi_fields[0])

print("Canonical average at t=0:",
      psi_canonical[0])

print("Exact initial value:",
      1 / (6 * beta * lambda0**2))

print("Maximum absolute discrepancy:", error)


# ==========================================================
# PRE-style figure
# ==========================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42
})

fig, ax = plt.subplots(figsize=(3.4, 2.55))

ax.plot(
    t,
    psi_fields,
    color="black",
    linewidth=1.25,
    label="Hamiltonian fields"
)

ax.plot(
    t[::12],
    psi_canonical[::12],
    linestyle="none",
    marker="o",
    markersize=3.1,
    markeredgewidth=0.85,
    markerfacecolor="none",
    markeredgecolor="#bd351b",
    label="Canonical average"
)

ax.axhline(
    0,
    color="0.65",
    linewidth=0.65,
    linestyle="--",
    zorder=0
)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\Psi_0(t)$")
ax.set_xlim(0, 15)

ax.legend(
    frameon=False,
    fontsize=8,
    loc="upper right"
)

ax.tick_params(
    direction="in",
    top=True,
    right=True
)

fig.tight_layout(pad=0.65)

fig.savefig(
    "quartic_relaxation_comparison.pdf",
    bbox_inches="tight"
)

plt.show()