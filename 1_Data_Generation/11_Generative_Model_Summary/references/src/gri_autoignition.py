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
    "CH4",  # fuel -- monotonic consumption, cleanest single progress scalar
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
        Number of output time points (the length of every returned array).
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

        All arrays have length ``n_steps``.
    """
    if tracked_species is None:
        tracked_species = TRACKED_SPECIES

    # --- Build the gas and set the initial thermodynamic state ---
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
    gas.TP = T0, P

    # --- Constant-pressure, adiabatic reactor (autoignition) ---
    reactor = ct.IdealGasConstPressureReactor(gas, clone=False)
    net = ct.ReactorNet([reactor])

    # --- Output time grid ---
    if log_time:
        times = np.concatenate(([0.0], np.logspace(np.log10(t_start),
                                                    np.log10(t_end),
                                                    n_steps - 1)))
    else:
        times = np.linspace(0.0, t_end, n_steps)

    # --- Pre-resolve species indices for speed ---
    sp_idx = [gas.species_index(s) for s in tracked_species]

    # --- Storage ---
    T_hist = np.empty(n_steps)
    X_hist = np.empty((n_steps, len(tracked_species)))
    dT_hist = np.empty(n_steps)
    dX_hist = np.empty((n_steps, len(tracked_species)))

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
    ones = np.ones(n_steps)
    record = {
        "temperature": T0 * ones,
        "pressure": P * ones,
        "phi": phi * ones,
        "t": times.copy(),
        "T": T_hist,
    }
    U = {"dT_dt": dT_hist}
    for j, s in enumerate(tracked_species):
        record[s] = X_hist[:, j]
        U["d{}_dt".format(s)] = dX_hist[:, j]
    record["U"] = U
    return record


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

