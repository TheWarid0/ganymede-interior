"""Export a two-phase convection run (results/*.pkl from hp_convection.py) to ParaView.

    python scripts/to_paraview.py results/hp_mu16.pkl --out paraview/hp_mu16

Writes one VTK ImageData file per snapshot (snap_0000.vti, ...) and a time index run.pvd.
In ParaView: File > Open > run.pvd > Apply; the time slider steps through the snapshots.

Fields (cell data, distances in km):
    T            temperature [K]
    T_minus_Tm   temperature below the melting point [K] (0 = temperate ice)
    porosity     water content [%]
    (shell runs also: log10_viscosity [Pa s], tidal_heating [microW/m^3]; height is above the ocean)
    velocity     ice velocity [cm/yr] (vector; z component is vertical)
    speed        |velocity| [cm/yr]
No dependencies beyond NumPy: the files are plain VTK XML.
"""
import argparse
import base64
import os
import pickle

import numpy as np

from ganymede.twophase import YEAR, HPIce


def _b64(arr):
    """VTK XML binary encoding: base64 of (uint32 byte count + raw little-endian data)."""
    raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()       # single precision: plenty for plotting
    return base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode("ascii")


def write_vti(path, fields, nx, nz, dx, dz):
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt32">',
             f'  <ImageData WholeExtent="0 {nx} 0 {nz} 0 0" Origin="0 0 0" Spacing="{dx} {dz} 1">',
             f'    <Piece Extent="0 {nx} 0 {nz} 0 0">',
             '      <CellData Scalars="T_minus_Tm" Vectors="velocity">']
    for name, a in fields.items():
        ncomp = a.shape[-1] if a.ndim == 3 else 1
        flat = a.reshape(nz * nx, ncomp) if ncomp > 1 else a.ravel()   # x fastest, then z: matches (nz, nx)
        lines.append(f'        <DataArray type="Float32" Name="{name}" NumberOfComponents="{ncomp}" '
                     f'format="binary">{_b64(flat)}</DataArray>')
    lines += ['      </CellData>', '    </Piece>', '  </ImageData>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(a):
    r = pickle.load(open(a.results, "rb"))
    par = r.get("params", {})
    shell = "info" in r                                    # Stage 5 ice shell run (shell_run.py)
    if shell:
        vscale, H_km, aspect = r["info"]["vscale_cm_yr"], r["info"]["D_km"], r["info"]["aspect"]
    else:                                                 # Stage 4 HP ice run (hp_convection.py)
        p = HPIce(H=par.get("H", 200.0) * 1e3)
        vscale, H_km, aspect = p.kappa / p.H * YEAR * 100, p.H / 1e3, 2.0
    os.makedirs(a.out, exist_ok=True)
    snaps = []
    for s in r["snaps"]:                                  # drop repeated final states
        if not snaps or abs(s["t_myr"] - snaps[-1]["t_myr"]) > 1e-6:
            snaps.append(s)
    nz, nx = snaps[0]["theta"].shape
    dx, dz = aspect * H_km / nx, H_km / nz
    entries = []
    for k, s in enumerate(snaps):
        vxc = 0.5 * (s["vx"][:, 1:] + s["vx"][:, :-1]) * vscale
        vzc = 0.5 * (s["vz"][1:] + s["vz"][:-1]) * vscale
        vel = np.stack([vxc, vzc, np.zeros_like(vxc)], axis=-1)
        fields = {
            "T": s["T"] if shell else p.Tm_top + s["theta"] * (p.Tm_bot - p.Tm_top),
            "T_minus_Tm": s["dT"],
            "porosity": s["phi"] * 100,
            "velocity": vel,
            "speed": np.hypot(vxc, vzc),
        }
        if shell:
            fields["log10_viscosity"] = np.log10(s["eta"])
            fields["tidal_heating"] = s["H_tidal"] * 1e6          # microW/m^3
        name = f"snap_{k:04d}.vti"
        write_vti(os.path.join(a.out, name), fields, nx, nz, dx, dz)
        entries.append(f'    <DataSet timestep="{s["t_myr"]:.6g}" file="{name}"/>')
    with open(os.path.join(a.out, "run.pvd"), "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0">\n  <Collection>\n'
                + "\n".join(entries) + "\n  </Collection>\n</VTKFile>\n")
    print(f"wrote {len(entries)} snapshots ({nx} x {nz} cells, t = {snaps[0]['t_myr']:.2f}-{snaps[-1]['t_myr']:.2f} Myr) "
          f"to {a.out}/run.pvd")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None, help="output folder (default: paraview/<results name>)")
    args = ap.parse_args()
    args.out = args.out or os.path.join("paraview", os.path.splitext(os.path.basename(args.results))[0])
    main(args)
