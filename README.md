# ganymede-interior

A readable, tested interior-structure and convection model for **Ganymede**, written from scratch in Python. The aim is a code built around ice physics, not adapted from a silicate-mantle code, and growing step by step towards two-phase (ice–water) convection in the high-pressure ice layer.

> Work in progress. Each stage is checked against an analytic or published benchmark before the next one starts.

## Roadmap

| Stage | Content | Benchmark | Status |
|---|---|---|---|
| 0 | 1D structure: density, gravity, pressure, C/MR² fit | Uniform sphere (analytic) | ✅ |
| 0b | Self-consistent H₂O layer: SeaFreeze phases, conductive ice Ih shell, adiabatic ocean / HP ice | Closed-form k = a/T shell; M and C/MR² recovered | ✅ (PlanetProfile comparison pending) |
| 1 | 1D two-phase compaction (McKenzie equations) | Solitary porosity waves: second-order convergence, c = 2A + 1, mass conservation | ✅ |
| 2 | 2D Stokes convection on a staggered grid, T-dependent viscosity | Analytic Stokes flow (order 2, div v ≈ 0); Blankenbach et al. (1989) 1a and 2a within 0.05 % | ✅ |
| 2d | Real ice: Arrhenius viscosity, dimensional ice-shell setup | Conductive vs convective shell (critical Ra) | ⬜ |
| 3 | Two-phase convection in the high-pressure ice layer | Literature setups | ⬜ |
| — | 3D (PETSc / Firedrake) | — | later |

## First results (pure-water H₂O layer, self-consistent with M and C/MR²)

| Surface heat flux | H₂O layer | Ice Ih shell | Ocean | HP ice | Phase sequence |
|---|---|---|---|---|---|
| 5 mW m⁻² | 809 km | 111 km | 224 km | 474 km | Ih → liquid → V → VI |
| 15 mW m⁻² | 802 km | 39 km | 455 km | 308 km | Ih → liquid → VI |
| 40 mW m⁻² | 799 km | 15 km | 537 km | 247 km | Ih → liquid → VI |

Gravity fixes the *total* H₂O thickness; the heat flux decides how it splits between ocean and high-pressure ice. Assumptions: conductive (non-convecting) Ih shell, adiabatic ocean and HP ice, no salts.

### Meltwater transport through high-pressure ice (Stage 1)

Porosity waves carry water upward in discrete pulses. For rough ice VI parameters (φ₀ = 1 %, Δρ ≈ 120 kg m⁻³), a wave crosses 300 km of HP ice in ~10³–10⁷ yr depending on grain size (10–0.1 mm). The crossing time depends on permeability only, **not** on ice viscosity, which sets the wave width (compaction length ~10 m – 10 km).

### Convection benchmarks (Stage 2)

| Blankenbach et al. (1989) | Nu (this code) | Nu (reference) | v_rms (this code) | v_rms (reference) |
|---|---|---|---|---|
| 1a: Ra = 10⁴, constant η | 4.8846 | 4.8844 | 42.865 | 42.865 |
| 2a: Ra = 10⁴, η contrast 10³ | 10.061 | 10.066 | 480.63 | 480.43 |

Richardson-extrapolated from 64² and 128² grids.

## Install

```bash
pip install -e ".[dev,eos]"
pytest              # fast tests (~25 s)
pytest -m slow      # 128² Blankenbach 2a run (~40 s)
```

## Layout

```
src/ganymede/   model code
tests/          benchmarks as unit tests
notebooks/      one notebook per stage, worked through cell by cell
```

## License

MIT
