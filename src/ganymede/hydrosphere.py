"""Stage 6a: evolution of Ganymede's salty hydrosphere -- does a freezing ocean make Ganymede grow?

Column (spherically symmetric) model of the H2O layer on a fixed rock interior:
    ice Ih shell  |  ocean: well-mixed upper layer + dense brine layer at the bottom  |  high-pressure ice
The total H2O + salt mass and the rock are fixed; the outer radius R follows from the mass balance.
Freezing water into ice Ih expands the body; freezing it into high-pressure ice (III, V, VI, denser
than water) contracts it. The question is which wins as Ganymede cools.

Salt (NaCl-like, mass fraction S of the liquid):
  * freezing-point depression of every ice phase (ideal solution):  dT = i_vh K_f m,  K_f = R T_m^2 / L
  * liquid density  rho = rho_water(P, T) (1 + beta S)
  * the shell base is a mush: new ice keeps a fraction k_eff of the salt; the rest is rejected as brine,
    of which a fraction f_sink sinks to the bottom brine layer (f_sink = 0: well-mixed ocean) with
    concentration S_brine = min(c_brine S_top, S_eutectic);  HP ice rejects all its salt the same way.
Thermal state (as Stage 0b): ice Ih shell conductive in temperature profile (k = a/T), ocean and HP ice
adiabatic; the shell base and the ocean floor sit at the salt-depressed liquidus. The shell's heat loss
uses the Stage 2d convective heat-flow curve for the chosen grain size.
Time: energy balance  Q_surface - Q_rock = latent heat of freezing (Ih and HP) - change of sensible heat,
stepping the shell thickness. Rock heat flux decays exponentially (radiogenic).

Simplifications: spherical symmetry; brine layer adiabatic (no double-diffusive interface); no salt in
the HP ices; constant latent heats; entrained salt does not change ice density.
"""
from dataclasses import dataclass, field, replace
import json
import logging
import os

import numpy as np

G = 6.674e-11
R_GAS = 8.314
YEAR = 3.15576e7
L_IH, L_HP = 334e3, 330e3          # latent heats [J/kg]
M_SALT = 0.05844                   # NaCl [kg/mol]
S_EUT = 0.233                      # NaCl eutectic mass fraction


# --- equation of state: fast bilinear tables built once from SeaFreeze ---------------------------

class EOS:
    """SeaFreeze density, expansivity and heat capacity for each phase on a (P, T) grid, plus the
    liquidus T_liq(P) and the solid phase just below it, resolved to 0.01 K."""

    def __init__(self, P_max=2000.0, nP=401, T_min=90.0, T_max=360.0, nT=271, cache=True):
        path = os.path.expanduser("~/.cache/ganymede_eos.npz")
        if cache and os.path.exists(path):
            d = np.load(path)
            self.__dict__.update({k: d[k] for k in d.files})
            return
        from seafreeze import seafreeze as sf
        logging.getLogger("lbftd").disabled = True
        self.P = np.linspace(0.1, P_max, nP)
        self.T = np.linspace(T_min, T_max, nT)
        PT = np.array([self.P, self.T], dtype=object)
        props = {}
        for code, name in {0: "water1", 1: "Ih", 3: "III", 5: "V", 6: "VI"}.items():
            out = sf.getProp(PT, name)
            for k in ("rho", "alpha", "Cp"):
                a = np.asarray(getattr(out, k), float)
                props[f"{k}{code}"] = _fill_nan(a)
        self.__dict__.update(props)
        # liquidus: coarse scan then bisection to 0.01 K
        Pl = np.linspace(0.1, P_max, 801)
        Tg = np.arange(230.0, 370.0, 0.05)
        ph = sf.whichphase(np.array([Pl, Tg], dtype=object))
        first = np.argmax(ph == 0, axis=1)
        self.Pl = Pl
        self.Tliq = Tg[first] - 0.025
        self.solid = ph[np.arange(len(Pl)), np.maximum(first - 1, 0)].astype(int)
        if cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, P=self.P, T=self.T, Pl=self.Pl, Tliq=self.Tliq, solid=self.solid, **props)

    def props(self, code, P, T):
        """(rho, alpha, cp) by bilinear interpolation; P [MPa], T [K]."""
        fi = (P - self.P[0]) / (self.P[1] - self.P[0])
        fj = (T - self.T[0]) / (self.T[1] - self.T[0])
        i = int(min(max(fi, 0), len(self.P) - 2)); j = int(min(max(fj, 0), len(self.T) - 2))
        a, b = fi - i, fj - j
        out = []
        for k in ("rho", "alpha", "Cp"):
            A = getattr(self, f"{k}{code}")
            out.append((1 - a) * (1 - b) * A[i, j] + a * (1 - b) * A[i + 1, j] + (1 - a) * b * A[i, j + 1]
                       + a * b * A[i + 1, j + 1])
        return out

    def liquidus(self, P):
        """Pure-water liquidus T [K] and the ice phase that freezes there, at P [MPa]."""
        T = np.interp(P, self.Pl, self.Tliq)
        k = int(min(np.searchsorted(self.Pl, P), len(self.Pl) - 1))
        return T, int(self.solid[k])


def _fill_nan(a):
    from scipy.ndimage import distance_transform_edt
    nan = np.isnan(a)
    if not nan.any():
        return a
    i = distance_transform_edt(nan, return_distances=False, return_indices=True)
    return a[tuple(i)]


def depression(S, T_m, L, i_vh=1.8):
    """Freezing-point depression [K] of an ice in contact with liquid of salt mass fraction S (ideal)."""
    S = min(max(S, 0.0), S_EUT)
    m = S / ((1 - S) * M_SALT)
    return i_vh * R_GAS * T_m**2 / L * m


# --- parameters and state -----------------------------------------------------------------------

@dataclass(frozen=True)
class HydroParams:
    r_rock: float = 1829.1e3       # radius of the rock interior [m] (Stage 0b fit)
    M_rock: float = None           # mass of rock; default: Ganymede mass minus the H2O mass
    M_h2o: float = None            # H2O + salt mass [kg]; default from the reference state
    T_s: float = 110.0
    S0: float = 0.01               # initial ocean salinity (mass fraction)
    k_eff: float = 0.1             # fraction of salt kept in the new ice at the mushy shell base
    f_sink: float = 1.0            # fraction of rejected brine that sinks to the bottom layer
    c_brine: float = 3.0           # sinking brine is c_brine times saltier than the upper ocean
    beta: float = 0.75             # haline density coefficient (rho = rho_w (1 + beta S))
    grain: float = 0.3             # grain size for the shell heat flow [mm] (0 = conductive shell)
    q_rock0: float = 30e-3         # rock heat flux at t = 0 [W/m^2]
    tau_rock: float = 2.0e9        # e-folding time of the rock heat flux [yr]
    k_water: float = 0.6           # thermal conductivity of the brine layer [W/m/K]
    Nu_brine: float = 1.0          # heat-transport enhancement in the stratified brine layer (1 = conduction)
    h_min: float = 500.0           # minimum brine-layer thickness used for its heat flux [m]

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass
class State:
    D: float                       # ice Ih shell thickness [m]
    s_top: float                   # salt mass in the upper ocean [kg]
    W_bot: float = 0.0             # water mass in the bottom brine layer [kg]
    s_bot: float = 0.0             # salt mass in the bottom brine layer [kg]
    s_ice: float = 0.0             # salt mass locked in the ice Ih shell [kg]
    t: float = 0.0                 # time [yr]
    hp: float = None               # HP-ice thickness [m]; None = floor in equilibrium with the ocean adiabat


@dataclass
class Column:
    R: float; D: float; z_hp: float; T_b: float; T_floor: float; S_top: float; S_bot: float
    M_liq: float; M_ih: float; M_hp: float; E_sens: float; h_bot: float; hp_phase: int
    z: np.ndarray = field(repr=False, default=None); T: np.ndarray = field(repr=False, default=None)
    rho: np.ndarray = field(repr=False, default=None); phase: np.ndarray = field(repr=False, default=None)

    @property
    def ocean(self):
        return self.z_hp - self.D

    @property
    def hp(self):
        return self.R - self.z_hp - self._r_rock

    def summary(self):
        return dict(R_km=self.R / 1e3, shell_km=self.D / 1e3, ocean_km=(self.z_hp - self.D) / 1e3,
                    hp_km=(self.R - self.z_hp - self._r_rock) / 1e3, T_base=self.T_b, T_floor=self.T_floor,
                    S_top=self.S_top, S_bot=self.S_bot, brine_km=self.h_bot / 1e3, hp_phase=self.hp_phase)


# --- the column -----------------------------------------------------------------------------------

def march(eos, p, R, D, S_top, S_bot, M_bot, M_rock, dz=500.0, keep=False, hp_thick=None):
    """Integrate the H2O layer from the surface (radius R) down to the rock (p.r_rock).
    hp_thick=None: the ocean floor is where the ocean adiabat meets the HP-ice liquidus (equilibrium);
    otherwise the HP-ice layer has this thickness (floor set by its own heat balance, brine mode).
    Returns (mass of H2O+salt above the rock, Column)."""
    r, z = R, 0.0
    m_above = 0.0
    P = 0.1                                            # MPa
    M_tot = M_rock                                      # rock mass; H2O mass added as we go down
    # gravity needs the mass below r: total mass minus what lies above; we start from M_total guess
    # and correct with the marched H2O mass (two passes are enough; g varies by < 20 % in the layer)
    mass_total = M_rock + march.M_h2o_guess
    T_b = None
    T = p.T_s
    phase, layer = 1, "ih"
    rows = []
    E_sens = M_ih = M_liq = M_hp = 0.0
    z_hp = None; T_floor = None; hp_phase = 0; T_a = None; P_floor = None
    r_floor = None if hp_thick is None else p.r_rock + hp_thick
    while r > p.r_rock + 1e-6:
        h = min(dz, r - p.r_rock)
        if layer == "ih":
            h = min(h, D - z) if D - z > 1e-6 else h
        g = G * (mass_total - m_above) / r**2
        if layer == "ih":
            if T_b is None:                            # base temperature: salty liquidus at the base pressure
                P_base = P + march.rho_ih_guess * g * D / 1e6
                Tl, _ = eos.liquidus(P_base)
                T_b = Tl - depression(S_top, Tl, L_IH)
            T = p.T_s * (T_b / p.T_s) ** (min(z + 0.5 * h, D) / D) if D > 0 else T_b
            rho, alpha, cp = eos.props(1, P, T)
        elif layer == "ocean" and r_floor is not None:
            if r <= r_floor + 1e-6:                    # prescribed floor reached
                layer, z_hp, T_floor, hp_phase, P_floor = "hp", z, T, eos.liquidus(P)[1], P
                if hp_phase not in (3, 5, 6):
                    hp_phase = 3
                continue
            h = min(h, r - r_floor)
            in_brine = M_bot > 0 and M_liq >= march.M_liq_guess - M_bot
            if in_brine and T_a is None:
                T_a = T
            S = S_bot if in_brine else S_top
            rho_w, alpha, cp = eos.props(0, P, T)
            rho = rho_w * (1 + p.beta * S)
        elif layer == "ocean":
            S = S_top if (M_bot <= 0 or M_liq < march.M_liq_guess - M_bot) else S_bot
            S_f = S_bot if M_bot > 0 else S_top

            def margin(Pq, Tq):
                Tl, sol = eos.liquidus(Pq)
                return (Tq - (Tl - depression(S_f, Tl, L_HP))) if sol != 1 else 1e9

            m0 = margin(P, T)
            rho_w, alpha, cp = eos.props(0, P, T)
            rho = rho_w * (1 + p.beta * S)
            T1 = T + alpha * g * T / cp * h
            P1 = P + rho * g * h / 1e6
            m1 = margin(P1, T1)
            if m0 <= 0:                                # the adiabat is at the HP-ice liquidus: ocean floor
                layer, z_hp, T_floor, hp_phase, P_floor = "hp", z, T, eos.liquidus(P)[1], P
                continue
            if m1 < 0:                                 # crossing inside this step: stop exactly at it
                h = h * m0 / (m0 - m1)
        else:
            Tl, sol = eos.liquidus(P)
            ph = sol if sol in (3, 5, 6) else hp_phase
            rho, alpha, cp = eos.props(ph, P, T)
        rm = r - 0.5 * h
        dm = rho * 4 * np.pi * rm**2 * h
        if layer == "ih":
            M_ih += dm
        elif layer == "ocean":
            M_liq += dm
        else:
            M_hp += dm
        E_sens += dm * cp * T
        if keep:
            rows.append((z, T, rho, {"ih": 1, "ocean": 0, "hp": hp_phase}[layer]))
        if layer != "ih":
            T += alpha * g * T / cp * h                # adiabat (ocean and HP ice)
        P += rho * g * h / 1e6
        m_above += dm
        r -= h
        z += h
        if layer == "ih" and z >= D - 1e-6:
            layer, T = "ocean", T_b
    if z_hp is None:
        z_hp, T_floor, P_floor = R - p.r_rock, T, P
    rho_b = None
    h_bot = 0.0
    if M_bot > 0:
        rp = R - z_hp
        h_bot = M_bot / (eos.props(0, P, T_floor)[0] * (1 + p.beta * S_bot) * 4 * np.pi * rp**2)
    col = Column(R=R, D=D, z_hp=z_hp, T_b=T_b, T_floor=T_floor, S_top=S_top, S_bot=S_bot, M_liq=M_liq,
                 M_ih=M_ih, M_hp=M_hp, E_sens=E_sens, h_bot=h_bot, hp_phase=hp_phase)
    col._r_rock = p.r_rock
    col.T_a = T_a if T_a is not None else T_floor     # temperature at the top of the brine layer
    col.P_floor = P_floor
    col.hp_thick = R - z_hp - p.r_rock
    if keep:
        a = np.array([(x[0], x[1], x[2], x[3]) for x in rows])
        col.z, col.T, col.rho, col.phase = a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(int)
    return m_above, col


march.M_h2o_guess = 0.0
march.rho_ih_guess = 930.0
march.M_liq_guess = 1e30


def solve_column(eos, p, st, M_h2o, M_rock, R_guess, keep=False, tol=0.5, S_guess=None):
    """Find the radius R that holds the H2O + salt mass for this state (shell thickness, salts).
    R_guess and S_guess (upper-ocean salinity) from the previous step make this much faster."""
    march.M_h2o_guess = M_h2o
    S_top = 0.0 if S_guess is None else S_guess
    M_bot = st.W_bot + st.s_bot
    S_bot = st.s_bot / M_bot if M_bot > 0 else 0.0
    R = R_guess
    col = None
    for outer in range(4):                              # salinity of the upper ocean depends on its mass
        def f(Rt):
            m, c = march(eos, p, Rt, st.D, S_top, S_bot, M_bot, M_rock, keep=False, hp_thick=st.hp)
            return m - M_h2o, c
        R0, R1 = R, R + (2e3 if outer == 0 and S_guess is None else 200.0)
        f0, _ = f(R0)
        f1, _ = f(R1)
        for _ in range(30):                             # secant on R
            R2 = R1 - f1 * (R1 - R0) / (f1 - f0)
            R0, f0 = R1, f1
            R1 = R2
            f1, col = f(R1)
            if abs(R1 - R0) < tol:
                break
        R = R1
        march.M_liq_guess = col.M_liq
        M_top = col.M_liq - M_bot
        S_new = st.s_top / M_top if M_top > 0 else S_EUT
        if abs(S_new - S_top) < 1e-7:
            break
        S_top = S_new
    _, col = march(eos, p, R, st.D, S_top, S_bot, M_bot, M_rock, keep=keep, hp_thick=st.hp)
    return col


# --- shell heat loss --------------------------------------------------------------------------------

class ShellHeatFlow:
    """q_out(D, T_b): Stage 2d Nusselt numbers (grain size, D) times the k = a/T conductive flux."""

    def __init__(self, grain_mm, results="results/shell_equilibrium.json"):
        self.D = self.Nu = None
        if grain_mm and os.path.exists(results):
            rows = [r for r in json.load(open(results)) if abs(r["d"] * 1e3 - grain_mm) < 1e-9
                    and r.get("aspect", 1.0) == 1.0 and not r.get("branch")]
            rows.sort(key=lambda r: r["D"])
            self.D = np.array([r["D"] for r in rows])
            self.Nu = np.array([max(r["Nu"], 1.0) for r in rows])

    def __call__(self, D, T_b, T_s):
        Nu = 1.0 if self.D is None else float(np.interp(D, self.D, self.Nu))
        return Nu * 651.0 * np.log(T_b / T_s) / D


# --- evolution --------------------------------------------------------------------------------------

def evolve(p, eos=None, D0=15e3, t_end=4.5e9, dD=1000.0, dhp=1000.0, dt_max=50e6, M_h2o=None, M_rock=None,
           R0=2631.2e3, verbose=True, heat_flow=None, log_every=1):
    """Thermal and salt evolution of the hydrosphere. Returns (records, info).

    No brine layer: the ocean is adiabatic from the shell base to the floor, both interfaces at the
    (salt-depressed) liquidus; a step in shell thickness fixes the whole column and the time it takes
    follows from the energy balance (latent heat of Ih and HP freezing, sensible heat).
    With a brine layer (rejected brine that sank): the layer is stably stratified and passes heat by
    conduction only (times Nu_brine). The floor sits at the brine liquidus and moves by its own
    Stefan balance  L_HP dM_hp/dt = Q_brine - Q_rock;  the shell base receives Q_brine:
    L_IH dM_ih/dt = Q_surface - Q_brine. Energy is conserved by construction (sum = Q_surface - Q_rock).
    """
    from .constants import M_GANYMEDE
    eos = eos or EOS()
    hf = heat_flow or ShellHeatFlow(p.grain)
    if M_h2o is None:
        march.M_h2o_guess = 0.0
        M_h2o, _ = march(eos, p, R0, D0, 0.0, 0.0, 0.0, M_GANYMEDE)
    M_rock = M_GANYMEDE - M_h2o if M_rock is None else M_rock
    st = State(D=D0, s_top=0.0)
    col = solve_column(eos, p, st, M_h2o, M_rock, R0)
    st.s_top = p.S0 * col.M_liq
    col = solve_column(eos, p, st, M_h2o, M_rock, col.R)
    salt_total = st.s_top
    rec = []
    n = [0]
    E0 = enthalpy(col)
    budget = dict(Q_net=0.0, Q_s=0.0)               # integral of (Q_surface - Q_rock) dt and of Q_surface dt [J]

    def Q_rock(t):
        return p.q_rock0 * np.exp(-t / p.tau_rock) * 4 * np.pi * p.r_rock**2

    def Q_surf(c):
        return hf(c.D, c.T_b, p.T_s) * 4 * np.pi * c.R**2

    def record(c, s, Qb=np.nan, mode="equilibrium"):
        d = c.summary()
        Qs, Qr = Q_surf(c), Q_rock(s.t)
        d.update(t_myr=s.t / 1e6, q_surface=Qs / (4 * np.pi * c.R**2), q_rock=Qr / (4 * np.pi * p.r_rock**2),
                 q_brine=Qb / (4 * np.pi * (p.r_rock + c.hp_thick) ** 2), mode=mode,
                 hp_km=c.hp_thick / 1e3, T_a=c.T_a,
                 salt_error=(s.s_top + s.s_bot + s.s_ice - salt_total) / salt_total,
                 energy_error=(enthalpy(c) - E0 + budget["Q_net"]) / max(budget["Q_s"], 1e-30))
        rec.append(d)
        if verbose and n[0] % log_every == 0:
            print(f"t={d['t_myr']:8.1f} Myr R={d['R_km']:9.3f} km shell={d['shell_km']:6.1f} ocean={d['ocean_km']:6.1f} "
                  f"hp={d['hp_km']:6.1f} brine={d['brine_km']:6.2f} km S_top={d['S_top']*1e3:6.1f} S_bot={d['S_bot']*1e3:6.1f} g/kg "
                  f"q_s={d['q_surface']*1e3:5.2f} q_r={d['q_rock']*1e3:5.2f} q_b={d["q_brine"]*1e3:6.2f} mW/m2 Eerr={d["energy_error"]:.1e} [{mode}]",
                  flush=True)
        n[0] += 1

    def copy(s):
        return State(**{k: getattr(s, k) for k in s.__dataclass_fields__})

    record(col, st)
    while st.t < t_end:
        if col.S_top >= S_EUT or (st.W_bot + st.s_bot > 0 and col.S_bot >= S_EUT):
            rec[-1]["stopped"] = "ocean brine reached the eutectic (beyond the model)"
            if verbose:
                print("stop: the ocean reached the eutectic salinity", flush=True)
            break
        if eos.liquidus(_base_pressure(col, p))[1] != 1:
            rec[-1]["stopped"] = "shell base reached the ice Ih - III limit"
            if verbose:
                print("stop: the shell base reached the ice Ih - III limit", flush=True)
            break
        Qs, Qr = Q_surf(col), Q_rock(st.t)
        brine_mode = (st.W_bot + st.s_bot > 0) or (p.f_sink > 0 and Qs > Qr)
        if not brine_mode:
            # --- no brine layer: equilibrium column, step the shell thickness -----------------------
            step = dD if Qs > Qr else -dD
            step = max(step, 2e3 - st.D)
            ok = False
            for attempt in range(6):
                new = copy(st)
                new.D = st.D + step
                ncol = solve_column(eos, p, new, M_h2o, M_rock, col.R, S_guess=col.S_top)
                dM_ih, dM_hp = ncol.M_ih - col.M_ih, ncol.M_hp - col.M_hp
                _apply_salt(p, new, col, dM_ih, dM_hp)
                ncol = solve_column(eos, p, new, M_h2o, M_rock, ncol.R, S_guess=ncol.S_top)
                released = L_IH * (ncol.M_ih - col.M_ih) + L_HP * (ncol.M_hp - col.M_hp) \
                    - (ncol.E_sens - col.E_sens)
                net = 0.5 * (Qs + Q_surf(ncol)) - Qr
                dt = released / net / YEAR if net != 0 else np.inf
                if dt > 0 and np.isfinite(dt):
                    if dt > dt_max:
                        step *= dt_max / dt
                        continue
                    ok = True
                    break
                step *= 0.25
            if not ok:                                  # at equilibrium: wait for the rock flux to change
                budget["Q_net"] += (Qs - Qr) * dt_max * YEAR
                budget["Q_s"] += Qs * dt_max * YEAR
                st.t += dt_max
                record(col, st)
                continue
            budget["Q_net"] += net * dt * YEAR
            budget["Q_s"] += 0.5 * (Qs + Q_surf(ncol)) * dt * YEAR
            new.t = st.t + dt
            st, col = new, ncol
            record(col, st)
            continue
        # --- brine layer present: floor and shell base each follow their own heat balance ------------
        #   floor:      L_HP dM_hp/dt = Q_brine - Q_rock
        #   shell base: L_IH dM_ih/dt - dE_sens/dt = Q_surface - Q_brine   (the ocean's sensible heat counts)
        # A step in shell thickness is taken; dt and the floor's motion are iterated to satisfy both.
        if st.hp is None:
            st.hp = col.hp_thick
        A_f = 4 * np.pi * (p.r_rock + st.hp) ** 2
        T_f = eos.liquidus(col.P_floor)[0]
        T_f -= depression(col.S_bot, T_f, L_HP)
        h = max(col.h_bot, p.h_min)
        Qb = p.Nu_brine * p.k_water * (T_f - col.T_a) / h * A_f       # upward through the brine layer
        top = Qs - Qb                                  # net heat lost by the upper ocean + shell base
        rho_hp = eos.props(col.hp_phase if col.hp_phase in (3, 5, 6) else 6, col.P_floor, T_f)[0]
        step = dD if top > 0 else -dD
        step = max(step, 2e3 - st.D)
        dt = abs(step) * 930.0 * 4 * np.pi * (col.R - col.D) ** 2 * L_IH / max(abs(top), 1e-30)
        dt = min(dt, dt_max * YEAR)
        for it in range(8):
            rate_hp = (Qb - Qr) / L_HP / (rho_hp * A_f)
            dt = min(dt, dhp / max(abs(rate_hp), 1e-30))
            new = copy(st)
            new.D = st.D + step
            new.hp = max(st.hp + rate_hp * dt, 0.0)
            ncol = solve_column(eos, p, new, M_h2o, M_rock, col.R, S_guess=col.S_top)
            dM_ih, dM_hp = ncol.M_ih - col.M_ih, ncol.M_hp - col.M_hp
            # total energy fixes dt exactly; the floor's share follows (Q_brine - Q_rock) dt
            released = L_IH * dM_ih + L_HP * dM_hp - (ncol.E_sens - col.E_sens)
            dt_new = released / (Qs - Qr) if Qs != Qr else dt_max * YEAR
            if dt_new <= 0 or not np.isfinite(dt_new):
                step = -0.5 * step if it % 2 else 0.5 * step
                continue
            if dt_new > dt_max * YEAR:
                step *= dt_max * YEAR / dt_new
                dt = dt_max * YEAR
                continue
            if abs(dt_new - dt) < 1e-3 * dt:
                dt = dt_new
                break
            dt = dt_new
        _apply_salt(p, new, col, dM_ih, dM_hp)
        new.t = st.t + dt / YEAR
        budget["Q_net"] += (Qs - Qr) * dt
        budget["Q_s"] += Qs * dt
        ncol = solve_column(eos, p, new, M_h2o, M_rock, col.R, S_guess=col.S_top)
        M_bot = new.W_bot + new.s_bot
        mode = "brine"
        if M_bot <= 0 or new.s_bot / M_bot <= ncol.S_top:
            # The bottom layer is no longer saltier (denser) than the ocean above: it overturns and mixes.
            # Mixing is fast, so total enthalpy is conserved: the floor returns to equilibrium with the
            # well-mixed ocean and the shell base melts or freezes to absorb the latent heat.
            target = enthalpy(ncol)
            new.s_top += new.s_bot
            new.s_bot = new.W_bot = 0.0
            new.hp = None
            ncol = _match_enthalpy(eos, p, new, M_h2o, M_rock, ncol.R, target)
            mode = "overturn"
        st, col = new, ncol
        record(col, st, Qb, mode=mode)
    return rec, dict(M_h2o=M_h2o, M_rock=M_rock, salt_total=salt_total)


def enthalpy(col):
    """Enthalpy of the H2O layer relative to all-liquid [J]: sensible heat minus latent heat of the ices."""
    return col.E_sens - L_IH * col.M_ih - L_HP * col.M_hp


def _match_enthalpy(eos, p, st, M_h2o, M_rock, R_guess, target, tol=1e-6):
    """Adjust the shell thickness (equilibrium floor) until the column enthalpy equals `target`."""
    def f(D):
        st.D = D
        c = solve_column(eos, p, st, M_h2o, M_rock, R_guess)
        return enthalpy(c) - target, c
    D0, D1 = st.D, st.D - 2e3
    f0, c0 = f(D0)
    f1, c1 = f(D1)
    for _ in range(25):
        if abs(f1) < tol * abs(target) or f1 == f0:
            break
        D2 = D1 - f1 * (D1 - D0) / (f1 - f0)
        D2 = min(max(D2, 2e3), 2 * max(D0, D1))
        D0, f0 = D1, f1
        D1 = D2
        f1, c1 = f(D1)
    st.D = D1
    return c1


def _base_pressure(col, p):
    """Pressure at the shell base [MPa] from the ice Ih mass above it (surface gravity is enough here)."""
    from .constants import M_GANYMEDE
    g = G * M_GANYMEDE / col.R**2
    return 0.1 + col.M_ih * g / (4 * np.pi * (col.R - 0.5 * col.D) ** 2) / 1e6


def _apply_salt(p, st, col_old, dM_ih, dM_hp):
    """Salt bookkeeping for one step (modifies st in place). Ice Ih keeps k_eff of the salt of the water it
    froze from; rejected salt (and HP-ice rejection) sinks with probability f_sink as brine."""
    M_bot = st.W_bot + st.s_bot
    S_top = col_old.S_top
    S_brine = min(p.c_brine * max(S_top, 1e-12), S_EUT)

    def sink(salt):
        """Move `salt` (already in the upper ocean) to the bottom layer as brine with water."""
        if salt <= 0:
            return
        moved = p.f_sink * salt
        st.s_top -= moved
        st.s_bot += moved
        st.W_bot += moved * (1 / S_brine - 1)

    if dM_ih > 0:                                       # freezing onto the shell
        kept = p.k_eff * S_top * dM_ih
        st.s_ice += kept
        st.s_top -= kept
        sink((1 - p.k_eff) * S_top * dM_ih)
    elif dM_ih < 0:                                     # shell melting: its salt returns to the ocean
        frac = min(-dM_ih / max(col_old.M_ih, 1e-30), 1.0)
        back = st.s_ice * frac
        st.s_ice -= back
        st.s_top += back
    if dM_hp > 0:                                       # ocean floor freezing: HP ice rejects all salt
        take_bot = min(dM_hp, st.W_bot)
        st.W_bot -= take_bot                            # salt stays in the brine layer
        rest = dM_hp - take_bot
        if rest > 0:
            sink(S_top * rest)
    elif dM_hp < 0:                                     # floor melting: fresh water into the bottom layer
        if M_bot > 0:
            st.W_bot += -dM_hp
