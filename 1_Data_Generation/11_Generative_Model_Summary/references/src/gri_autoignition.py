#!/usr/bin/env python3
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001
for Los Alamos National Laboratory (LANL), which is operated by Triad
National Security, LLC for the U.S. Department of Energy/National Nuclear
Security Administration. All rights in the program are reserved by Triad
National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others
acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
in this material to reproduce, prepare derivative works, distribute copies
to the public, perform publicly and display publicly, and to permit others
to do so.

==============================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

gri_autoignition.py - Cantera GRI-Mech 3.0 autoignition chemistry solver.

This module is the DATA-GENERATING MODEL for the chemical-kinetics example of
the LAROMance workflow. It is the kinetics analogue of the crystal-plasticity
(VPSC / EVP-FFT) codes used in the published creep models: a high-fidelity
physics model whose expensive-at-scale evaluation the LAROMance surrogate is
meant to replace.

Physics
-------
A homogeneous (0-D) reacting gas parcel of a methane/air mixture is ignited and
its chemistry integrated in time with a stiff ODE solver (handled internally by
Cantera's ``ReactorNet``). We use a constant-pressure, adiabatic reactor
(``IdealGasConstPressureReactor``), i.e. a classic *autoignition* problem:

    * The mixture starts at an initial temperature ``T0``, pressure ``P`` and
      equivalence ratio ``phi`` (fuel/air ratio relative to stoichiometric).
    * Nothing happens for an induction (ignition-delay) period, then the
      radical pool runs away and the temperature jumps by ~1000-2000 K over a
      very short time -> a stiff, strongly path-dependent trajectory.

Mechanism
---------
GRI-Mech 3.0 (``gri30.yaml``, shipped with Cantera): 53 species, 325
reactions, validated for natural-gas (methane-dominated) combustion.

Why this is a good LAROMance test case
--------------------------------------
    * Few scalar control inputs (T0, P, phi)        -> small input grid.
    * A monotonic progress axis (time)              -> the ``evm`` analogue.
    * Internal STATE fed back in (T + species)      -> the ``rhoc`` analogue
      (path dependence captured via state, not recurrence).
    * Outputs (production rates) span many ORDERS of magnitude and are near
      zero during induction then spike at ignition  -> exercises the
      log10/symlog output scalings exactly as the creep rates do.
    * Cheap to generate (ms-s per run) so the whole dataset builds locally,
      with no HPC/SLURM batch machinery.

The companion generator in ``13_Batch_Template_Setup`` imports
:func:`run_single_case` to turn each row of the design-of-experiment
``samples.csv`` into one simulation trajectory, and assembles the LAROMance
``.pickle`` consumed by ``2_Data_Processing``.
"""

from __future__ import annotations

import numpy as np

try:
    import cantera as ct
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "This model requires the 'cantera' package. Install it with\n"
        "    conda install -c conda-forge cantera\n"
        "or\n"
        "    pip install cantera\n"
    ) from exc


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

#: Reaction mechanism shipped with Cantera (GRI-Mech 3.0).
MECHANISM = "gri30.yaml"

#: Fuel and oxidizer definitions for a methane/air mixture. ``set_equivalence_ratio``
#: uses these to build the initial composition for a given ``phi``.
FUEL = "CH4:1"
OXIDIZER = "O2:1, N2:3.76"

#: Tracked species subset. GRI-Mech has 53 species, but a surrogate does not
#: need all of them: we keep the major reactants/products plus the key radicals
#: that drive the ignition dynamics. These become the per-trajectory STATE
#: variables (both surrogate inputs and outputs, like ``rhoc`` in creep).
#:
#: The number of surrogate OUTPUTS is always ``1 (dT_dt) + len(TRACKED_SPECIES)``
#: and the coupled feedback-STATE inputs are ``temperature, pressure, phi, t, T``
#: plus one entry per tracked species. To change the problem size, edit this
#: list only -- everything downstream derives from it automatically.
#:
#: FIRST PASS (minimal 2-output model): track a single, well-behaved major
#: component (CH4 fuel consumption) alongside temperature. This gives exactly
#: two outputs ``[dT_dt, dCH4_dt]`` and coupled inputs
#: ``temperature, pressure, phi, t, T, CH4``.
TRACKED_SPECIES = [
    "CH4",  # fuel
    "O2",   # oxidizer
    "CO2",  # product
    "H2O",  # product
    "CO",   # intermediate
    "OH",   # radical (ignition driver)
    "H",    # radical
    "O",    # radical
]

#: Full 8-species set used for the richer model (restore by assigning this to
#: TRACKED_SPECIES). Kept here so expanding the state later is a one-line change.
TRACKED_SPECIES_FULL = [
    "CH4",  # fuel
    "O2",   # oxidizer
    "CO2",  # product
    "H2O",  # product
    "CO",   # intermediate
    "OH",   # radical (ignition driver)
    "H",    # radical
    "O",    # radical
]


#: Default validated operating envelope for the autoignition regime. These are
#: the bounds suggested for the design-of-experiment (step 1.2).
DEFAULT_RANGES = {
    "TEMP": (1000.0, 1800.0),   # initial temperature [K]
    "PRESSURE": (1.0e5, 4.0e6),  # pressure [Pa]
    "PHI": (0.5, 1.5),           # equivalence ratio [-]
}



# ----------------------------------------------------------------------------
# Core single-case simulation
# ----------------------------------------------------------------------------

def run_single_case(
    T0: float,
    P: float,
    phi: float,
    *,
    t_end: float = 1.0,
    n_steps: int = 200,
    tracked_species=None,
    mechanism: str = MECHANISM,
    log_time: bool = True,
    t_start: float = 1.0e-8,
    peak_refine: bool = False,
    peak_species: str = "CH4",
    peak_density: int = 200,
    peak_pad_decades: float = 1.0,
    decay_frac: float = 0.01,
):
    """Integrate one constant-pressure autoignition trajectory.

    Parameters
    ----------
    T0 : float
        Initial temperature [K].
    P : float
        Pressure [Pa] (held constant for the const-pressure reactor).
    phi : float
        Equivalence ratio [-] (fuel/air relative to stoichiometric).
    t_end : float, optional
        Physical end time of the integration [s]. Default 1.0 s is long enough
        to capture ignition across the whole envelope, including the coldest
        (slowest-igniting) corner T0~1000 K at 1 atm.
    n_steps : int, optional
        Number of output time points on the *backbone* grid (the baseline log/
        linear grid). When ``peak_refine`` is True, additional points are
        inserted around the ignition pulse, so the returned arrays are LONGER
        than ``n_steps`` (and may differ in length between cases -- this is
        allowed by the LAROMance layout, which stores a per-simulation
        ``sim_lengths``). When ``peak_refine`` is False this is exactly the
        returned array length (legacy behaviour).
    tracked_species : list[str] or None, optional
        Species to record. Defaults to :data:`TRACKED_SPECIES`.
    mechanism : str, optional
        Cantera mechanism file. Default :data:`MECHANISM` (GRI-Mech 3.0).
    log_time : bool, optional
        If True (default) the output time grid is logarithmically spaced
        between ``t_start`` and ``t_end`` so the (very short) ignition front is
        well resolved; this mirrors the log-spaced ``evm`` axis used in creep.
    t_start : float, optional
        First (non-zero) time sample [s] when ``log_time`` is True.
    peak_refine : bool, optional
        If True, run a cheap coarse pass first to locate the ignition pulse,
        then integrate on a refined grid that densely resolves the rise AND the
        drop of the rate signal. Default False (legacy single-pass behaviour).
    peak_species : str, optional
        Tracked species whose production-rate magnitude ``|d<species>/dt|`` is
        used to locate the ignition pulse. Default ``"CH4"`` (fuel consumption
        gives the cleanest single peak/drop). If the species is unavailable the
        temperature rate ``|dT/dt|`` is used as a fallback.
    peak_density : int, optional
        Number of extra points to insert across the ignition window (log-spaced
        between the padded start and end of the pulse). Default 200.
    peak_pad_decades : float, optional
        How many decades of time to pad on each side of the detected pulse when
        building the dense window, so the induction tail and the post-ignition
        decay are both captured. Default 1.0 decade.
    decay_frac : float, optional
        Fraction of the peak rate magnitude used to define the end of the drop:
        the dense window extends to the first time after the peak where the rate
        magnitude falls back below ``decay_frac * peak``. Default 0.01 (1%).

    Returns
    -------
    dict
        A single-simulation record in LAROMance layout::

            {
                # ---- scalar control inputs, repeated per step ----
                "temperature": np.ndarray,   # T0 (constant)
                "pressure":    np.ndarray,   # P  (constant)
                "phi":         np.ndarray,   # phi (constant)
                # ---- monotonic progress axis ----
                "t":           np.ndarray,   # time [s]
                # ---- feedback STATE (input AND output) ----
                "T":           np.ndarray,   # current temperature [K]
                "<species>":   np.ndarray,   # current mole fraction, per species
                ...
                # ---- outputs: rates to be integrated forward ----
                "U": {
                    "dT_dt":        np.ndarray,       # dT/dt [K/s]
                    "d<species>_dt": np.ndarray,      # dX/dt [1/s], per species
                },
            }

        All arrays within a single record share the same length. That length
        equals ``n_steps`` when ``peak_refine`` is False, and is larger (and
        case-dependent) when ``peak_refine`` is True.
    """
    if tracked_species is None:
        tracked_species = TRACKED_SPECIES

    def _integrate(times):
        """Integrate the reactor over ``times`` and build a LAROMance record.

        A fresh gas/reactor/network is created on every call so that each pass
        starts from the same initial state (Cantera reactors are stateful).
        """
        n = len(times)

        # --- Build the gas and set the initial thermodynamic state ---
        gas = ct.Solution(mechanism)
        gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
        gas.TP = T0, P

        # --- Constant-pressure, adiabatic reactor (autoignition) ---
        reactor = ct.IdealGasConstPressureReactor(gas, clone=False)
        net = ct.ReactorNet([reactor])

        # --- Pre-resolve species indices for speed ---
        sp_idx = [gas.species_index(s) for s in tracked_species]

        # --- Storage ---
        T_hist = np.empty(n)
        X_hist = np.empty((n, len(tracked_species)))
        dT_hist = np.empty(n)
        dX_hist = np.empty((n, len(tracked_species)))

        def _record(i):
            """Capture state and net production rates at the current reactor time."""
            T_hist[i] = reactor.T
            X = reactor.phase.X                      # mole fractions [-]
            X_hist[i, :] = X[sp_idx]
            # Net molar production rates -> convert to a mole-fraction rate [1/s].
            conc_total = reactor.phase.density_mole   # [kmol/m^3]
            wdot = reactor.phase.net_production_rates  # [kmol/m^3/s]
            dX_dt = wdot / conc_total                 # d(mole fraction)/dt [1/s]
            dX_hist[i, :] = dX_dt[sp_idx]
            # Temperature rate via energy release (finite reactor derivative).
            dT_hist[i] = _temperature_rate(reactor)

        # --- Integrate, recording at each output time ---
        for i, t in enumerate(times):
            if t > 0.0:
                net.advance(t)
            _record(i)

        # --- Assemble the LAROMance record ---
        ones = np.ones(n)
        rec = {
            "temperature": T0 * ones,
            "pressure": P * ones,
            "phi": phi * ones,
            "t": np.asarray(times, dtype=float).copy(),
            "T": T_hist,
        }
        U = {"dT_dt": dT_hist}
        for j, s in enumerate(tracked_species):
            rec[s] = X_hist[:, j]
            U["d{}_dt".format(s)] = dX_hist[:, j]
        rec["U"] = U
        return rec

    # --- Baseline (backbone) output time grid ---
    if log_time:
        backbone = np.concatenate(([0.0], np.logspace(np.log10(t_start),
                                                      np.log10(t_end),
                                                      n_steps - 1)))
    else:
        backbone = np.linspace(0.0, t_end, n_steps)

    if not peak_refine:
        # Legacy single-pass behaviour.
        return _integrate(backbone)

    # ------------------------------------------------------------------
    # Peak-aware two-pass refinement
    # ------------------------------------------------------------------
    # 1. Coarse pass on the backbone grid to locate the ignition pulse.
    coarse = _integrate(backbone)

    # Choose the driver signal: |d<peak_species>/dt| (cleanest drop), falling
    # back to |dT/dt| if that species is not tracked.
    driver_key = "d{}_dt".format(peak_species)
    if driver_key in coarse["U"]:
        driver = np.abs(coarse["U"][driver_key])
    else:
        driver = np.abs(coarse["U"]["dT_dt"])

    t_coarse = coarse["t"]
    i_peak = int(np.argmax(driver))
    peak_val = driver[i_peak]
    t_peak = t_coarse[i_peak]

    # If no meaningful pulse was found (e.g. a non-igniting cold case) fall back
    # to the backbone grid so we never produce a degenerate dense window.
    if peak_val <= 0.0 or t_peak <= 0.0:
        return coarse

    # 2. Find the end of the drop: first time after the peak where the driver
    #    magnitude falls back below ``decay_frac * peak_val``.
    threshold = decay_frac * peak_val
    after = np.where((np.arange(len(driver)) > i_peak) & (driver < threshold))[0]
    t_decay_end = t_coarse[after[0]] if after.size > 0 else t_coarse[-1]

    # 3. Build the dense window, padded by ``peak_pad_decades`` on each side
    #    (in log-time), clipped to the integration bounds.
    pad = 10.0 ** peak_pad_decades
    lo = max(t_start, t_peak / pad)
    hi = min(t_end, t_decay_end * pad)
    if hi <= lo:
        return coarse

    dense = np.logspace(np.log10(lo), np.log10(hi), max(2, int(peak_density)))

    # 4. Merge backbone + dense window, sort, and drop near-duplicate times.
    merged = np.unique(np.concatenate([backbone, dense]))
    merged = merged[np.concatenate(([True], np.diff(merged) > 0.0))]

    # 5. Refined pass on the merged grid (true solver evaluations, no interp).
    return _integrate(merged)



def _temperature_rate(reactor):
    """Instantaneous dT/dt [K/s] for a constant-pressure adiabatic reactor.

    dT/dt = -(1 / (rho * cp)) * sum_k ( h_k * wdot_k )

    where ``h_k`` are partial molar enthalpies [J/kmol], ``wdot_k`` the net
    molar production rates [kmol/m^3/s], ``rho`` the mass density and ``cp``
    the mass-specific heat at constant pressure.
    """
    thermo = reactor.phase
    h_molar = thermo.partial_molar_enthalpies    # [J/kmol]
    wdot = thermo.net_production_rates            # [kmol/m^3/s]
    rho = thermo.density_mass                     # [kg/m^3]
    cp = thermo.cp_mass                           # [J/kg/K]
    return float(-np.dot(h_molar, wdot) / (rho * cp))


# ----------------------------------------------------------------------------
# Standalone demo / smoke test
# ----------------------------------------------------------------------------

def _demo():
    """Run a few representative cases and print a short diagnostic summary.

    Also computes the ignition-delay time (tau_ign, defined here as the time of
    maximum dT/dt) for each case, as a sanity check that the chemistry behaves
    physically (hotter / stoichiometric mixtures ignite faster).
    """
    cases = [
        (1200.0, 101325.0, 1.0),
        (1400.0, 101325.0, 1.0),
        (1400.0, 2.0e6, 0.7),
    ]
    print("GRI-Mech 3.0 constant-pressure autoignition - demo\n")
    print("{:>7} {:>12} {:>5}  {:>12} {:>10}".format(
        "T0[K]", "P[Pa]", "phi", "tau_ign[s]", "Tfinal[K]"))
    for T0, P, phi in cases:
        rec = run_single_case(T0, P, phi)
        i_ign = int(np.argmax(rec["U"]["dT_dt"]))
        tau = rec["t"][i_ign]
        Tfinal = rec["T"][-1]
        print("{:>7.0f} {:>12.0f} {:>5.2f}  {:>12.3e} {:>10.1f}".format(
            T0, P, phi, tau, Tfinal))
    print("\nTracked species:", ", ".join(TRACKED_SPECIES))
    print("Record keys:", list(rec.keys()))
    print("U (output) keys:", list(rec["U"].keys()))
    print("Array length per trajectory:", len(rec["t"]))


if __name__ == "__main__":
    _demo()

