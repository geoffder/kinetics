import numpy as np
import matplotlib.pyplot as plt


def disc2D(molecules, D, h, r):
    """Return a vectorized closure used to map time to concentration [mM].
        molecules = number of molecules released
        D = diffusion coefficient (for given neurotransmitter) [m^2 / s]
        h = height of the disc [m]
        r = distance from instantaneous line source of the transmitter [m]
    Equation replicates diffusion from release site as described in review by
    Barbour & Hausser, 1997.
    original values:
        molecules = 4700
        D = 7.6e-10 [literature value for glutamine]
        h = 20e-9
        r = 0 (directly across, distance of h), 1.1e-6 (peri-synaptic)
    """
    def closure(t):
        M = molecules / 6.02e23  # conversion to moles
        d = t * D
        a = M / (4 * h * np.pi * d)
        b = np.exp(-r ** 2 / (4 * d))
        return a * b
    return np.vectorize(closure)


def space3D(molecules, D, r, alpha=.21, lam=1.55):
    """Same as disc2D(), but modelling a restricted 3D space. alpha and lam(da)
    are dimensionless constants.
        alpha = volume fraction (% of total volume diffusion is restricted to)
            This models how there are many obstacles filling up space in the
            neural medium. Increases concentraion via (M / alpha).
        lambda = tortuosity (avg diffusional path is longer (*) by this factor)
            This models the extra distance that molecules must diffuse around
            the obstacles. Lowers diffusion via (D / lam ** 2).
    Returns concentration in mM."""
    def closure(t):
        M = molecules / 6.02e23  # conversion to moles
        d = t * (D / lam ** 2)
        a = M / (8 * alpha * (np.pi * d) ** 1.5)
        b = np.exp(-r ** 2 / (4 * d))
        return a * b
    return np.vectorize(closure)


def ach_2D(radius):
    return disc2D(10000, 4e-10, 20e-9, radius)


def ach_3D(radius):
    return space3D(10000, 4e-10, radius, alpha=.12)


def glut_2D(radius):
    return disc2D(4700, 7.6e-10, 20e-9, radius)


def glut_3D(radius):
    return space3D(4700, 7.6e-10, radius, alpha=.12)


if __name__ == "__main__":
    time_ax = np.arange(1, 25001) * .001  # [ms]
    time = time_ax / 1000  # [s]

    centre = disc2D(4700, 7.6e-10, 20e-9, 0.)(time) / 1000
    dist = disc2D(4700, 7.6e-10, 20e-9, 1.1e-6)(time) / 1000

    centre3D = space3D(4700, 7.6e-10, 0.)(time) / 1000
    dist3D = space3D(4700, 7.6e-10, 1.1e-6)(time) / 1000

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.plot(time_ax, centre, label="r = 0")
    ax.plot(time_ax, dist, label="r = 1.1μm")
    ax.plot(time_ax, centre3D, label="r = 0 (3D)")
    ax.plot(time_ax, dist3D, label="r = 1.1μm (3D)")
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 1e-3)
    ax.set_xlim(0, 5)
    ax.set_ylabel("Concentration (M)", fontsize=12)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.legend(frameon=False, fontsize=11)

    for ticks in (ax.get_xticklabels() + ax.get_yticklabels()):
        ticks.set_fontsize(11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()
