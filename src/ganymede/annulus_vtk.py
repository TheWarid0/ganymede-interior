"""ParaView output for annulus-sector fields: VTK StructuredGrid (.vts) on the real curved geometry.

Points are the cell corners (r_f, theta_f) mapped to x = r sin(theta), y = r cos(theta): theta = 0 is the
top (north) and a half annulus (theta in [0, pi]) is the right half of a cross-section through Ganymede.
Fields are cell data; vectors are given in Cartesian components (for Glyph / Stream Tracer) and the
radial / tangential parts are written as scalars too. Plain VTK XML, NumPy only.
"""
import base64

import numpy as np


def _b64(arr):
    raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
    return base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode("ascii")


def cell_velocity(g, vr, vt):
    """Interior-face velocities -> cell-centred (v_r, v_theta), walls included as zeros."""
    vr_f = np.vstack([np.zeros((1, g.nt)), vr, np.zeros((1, g.nt))])
    vt_f = np.hstack([np.zeros((g.nr, 1)), vt, np.zeros((g.nr, 1))])
    return 0.5 * (vr_f[1:] + vr_f[:-1]), 0.5 * (vt_f[:, 1:] + vt_f[:, :-1])


def write_vts(path, g, fields, scale=1.0):
    """fields: name -> (nr, nt) array (scalar) or (nr, nt, 3) (vector). scale multiplies coordinates."""
    R, T = np.meshgrid(g.r_f * scale, g.t_f, indexing="ij")            # (nr+1, nt+1)
    pts = np.stack([R * np.sin(T), R * np.cos(T), np.zeros_like(R)], axis=-1)
    # VTK structured order: i (theta) fastest, then j (r)
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">',
             f'  <StructuredGrid WholeExtent="0 {g.nt} 0 {g.nr} 0 0">',
             f'    <Piece Extent="0 {g.nt} 0 {g.nr} 0 0">',
             '      <Points>',
             f'        <DataArray type="Float32" NumberOfComponents="3" format="binary">{_b64(pts.reshape(-1, 3))}</DataArray>',
             '      </Points>',
             '      <CellData>']
    for name, a in fields.items():
        a = np.asarray(a)
        nc = a.shape[-1] if a.ndim == 3 else 1
        flat = a.reshape(-1, nc) if nc > 1 else a.ravel()
        lines.append(f'        <DataArray type="Float32" Name="{name}" NumberOfComponents="{nc}" format="binary">{_b64(flat)}</DataArray>')
    lines += ['      </CellData>', '    </Piece>', '  </StructuredGrid>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines))


def stokes_fields(g, vr, vt, P, eta_c):
    """Standard field set for a Stokes solution."""
    ur, ut = cell_velocity(g, vr, vt)
    Tc = g.t_c[None, :]
    vx = ur * np.sin(Tc) + ut * np.cos(Tc)
    vy = ur * np.cos(Tc) - ut * np.sin(Tc)
    return {"velocity": np.stack([vx, vy, np.zeros_like(vx)], axis=-1), "v_r": ur, "v_theta": ut,
            "speed": np.hypot(ur, ut), "pressure": P, "log10_viscosity": np.log10(eta_c)}
