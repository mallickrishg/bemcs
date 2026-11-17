# %% [markdown]
# # Plane strain: a buried fault near a bimaterial interface (Dundurs check)
# Two bonded half-planes with the same Poisson's ratio: material 1 (MU1) below the interface
# y = 0 and material 2 (MU2 = 4 MU1) above. A buried fault in material 1 dips at 40 degrees, with
# prescribed slip, shear and opening, tapering smoothly to zero at both tips.
#
# - The fault is a line of slip elements (`'s'`, one-ring hats).
# - The interface is a line of force elements (`'h'`, trapezoids, the default force basis), out to
#   |x| = L with log-spaced elements.
# - Every element lists the shear modulus on its -n and +n side. The interface runs left to right,
#   so its unit normal points up: mu_minus = MU1 (below), mu_plus = MU2 (above).
#
# **Exact solution.** The field of an edge dislocation near a bonded bimaterial interface (Dundurs
# 1969) is the homogeneous dislocation in material 1 plus image terms at the mirror point, and
# transmitted terms in material 2. Their coefficients follow from continuity of displacement and
# traction along the interface (solved below; the residual confirms the ansatz). The fault's slip s(xi)
# is a dislocation density -s'(xi) along the fault, so its exact field is a quadrature over the fault.
#
# The script compares the BEM with the exact stresses, and the trapezoids with the optional "nodal"
# (tent) force basis, which is second order on straight interfaces in plane strain.

# %% imports
import numpy as np
import matplotlib.pyplot as plt
import bemcs

MU1, MU2, NU = 1.0, 4.0, 0.25
CENTRE, DIP, HALF = np.array([0.0, -0.9]), np.deg2rad(40.0), 0.5  # fault
SLIP = (1.0, 0.5)  # shear and opening slip at the fault centre (local t, n)
H_EL = 0.05  # element length on the fault and near it on the interface
L = 300.0  # interface half-length
geometry, mu = "planestrain", 1.0
T_F = np.array([np.cos(DIP), np.sin(DIP)])  # fault direction
N_F = np.array([-T_F[1], T_F[0]])  # fault normal


def slip_profile(xi):
    return np.cos(0.5 * np.pi * xi / HALF) ** 2


def mu_field(x, y):
    return np.where(y > 0, MU2, MU1)


# %% the exact solution: an edge dislocation near the interface, integrated over the fault


def _clog(w, up):
    """log w with its branch cut along +i (up) or -i from 0, away from the interface."""
    rot = 1j if up else -1j
    return np.log(np.abs(w)) + 1j * (np.angle(w * rot) - np.angle(rot))


def _terms(z, p, up):
    w = z - p
    return [(_clog(w, up), 1 / w, -1 / w**2), (1 / w, -1 / w**2, 2 / w**3),
            (1 / w**2, -2 / w**3, 6 / w**4), (1 / w**3, -3 / w**4, 12 / w**5)]  # log, w^-1, w^-2, w^-3


def _potentials(z, z0, A, c, upper):
    """phi, phi', phi'', psi, psi' in material 1 (upper=False) or 2. c: 7 complex image /
    transmitted coefficients (phi: log, ^-1, ^-2; psi: log, ^-1, ^-2, ^-3) [+ 2 constants]."""
    t = _terms(z, z0 if upper else np.conj(z0), up=not upper)
    if upper:
        phi, psi = [0 * z] * 3, [0 * z] * 2
    else:  # the homogeneous dislocation at z0, cut downward
        t0 = _terms(z, z0, up=False)
        phi = [A * t0[0][k] for k in range(3)]
        psi = [np.conj(A) * t0[0][k] - A * np.conj(z0) * t0[1][k] for k in range(2)]
    phi = [phi[k] + sum(c[m] * t[m][k] for m in range(3)) for k in range(3)]
    psi = [psi[k] + sum(c[3 + m] * t[m][k] for m in range(4)) for k in range(2)]
    if upper:
        phi[0], psi[0] = phi[0] + c[7], psi[0] + c[8]
    return phi[0], phi[1], phi[2], psi[0], psi[1]


def dislocation_constants(z0, b, mu2=MU2):
    """Image (material 1) and transmitted (material 2) coefficients for a dislocation with
    Burgers vector b (complex) at z0 (Im z0 < 0), from continuity along y = 0."""
    kappa = 3 - 4 * NU
    A = -1j * MU1 * b / (np.pi * (kappa + 1))
    d = -z0.imag
    z = z0.real + d * np.r_[-np.geomspace(1e3, 1e-2, 100), np.geomspace(1e-2, 1e3, 100)] + 0j

    def residual(q):
        v = q[0::2] + 1j * q[1::2]
        out = []
        for upper, mu_ in ((False, MU1), (True, mu2)):
            phi, dphi, _, psi, _ = _potentials(z, z0, A, v[:7] if not upper else v[7:], upper)
            out.append(((3 - 4 * NU) * phi - z * np.conj(dphi) - np.conj(psi)) / (2 * mu_))
            out.append(phi + z * np.conj(dphi) + np.conj(psi))
        r = np.r_[out[0] - out[2], out[1] - out[3]]
        return np.r_[r.real, r.imag]

    n = 2 * 16
    r0 = residual(np.zeros(n))
    M = np.stack([residual(np.eye(n)[k]) - r0 for k in range(n)], axis=1)
    q = np.linalg.lstsq(M, -r0, rcond=None)[0]
    v = q[0::2] + 1j * q[1::2]
    return A, v[:7], v[7:], np.abs(M @ q + r0).max() / np.abs(r0).max()


def exact_stress(x, y, mu2=MU2, n_quad=100):
    """True stresses (sxx, syy, sxy) of the fault, as a quadrature of dislocations."""
    z = np.asarray(x, float) + 1j * np.asarray(y, float)
    out = np.zeros((3,) + z.shape)
    g, w = np.polynomial.legendre.leggauss(n_quad)
    worst = 0.0
    for gi, wi in zip(g, w):
        xi = HALF * gi
        ds = -(np.pi / (2 * HALF)) * np.sin(np.pi * xi / HALF)
        b = -ds * (SLIP[0] * T_F + SLIP[1] * N_F)
        p = CENTRE + xi * T_F
        z0 = complex(*p)
        A, c1, c2, res = dislocation_constants(z0, complex(*b), mu2)
        worst = max(worst, res)
        for upper, c in ((False, c1), (True, c2)):
            mask = (z.imag > 0) if upper else (z.imag <= 0)
            zz = z[mask]
            _, dphi, d2phi, _, dpsi = _potentials(zz, z0, A, c, upper)
            s_dif = 2 * (np.conj(zz) * d2phi + dpsi)
            out[0][mask] += HALF * wi * 0.5 * (4 * dphi.real - s_dif.real)
            out[1][mask] += HALF * wi * 0.5 * (4 * dphi.real + s_dif.real)
            out[2][mask] += HALF * wi * 0.5 * s_dif.imag
    return out, worst


# %% the mesh


def graded(h, L, flat=2.0, ratio=1.08, h_max=5.0):
    r, step = list(np.arange(0.0, flat + 1e-12, h)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    r = np.array(r)
    return np.r_[-r[:0:-1], r]


def fault_mesh(h=H_EL, force_basis="trapezoid", mu2=MU2):
    n_f = int(round(2 * HALF / h))
    xi = np.linspace(-HALF, HALF, n_f + 1)
    P = CENTRE + xi[:, None] * T_F
    xs = graded(h, L)
    n_i = len(xs) - 1
    els = bemcs.initialize_els()
    els.x1, els.y1 = np.r_[P[:-1, 0], xs[:-1]], np.r_[P[:-1, 1], np.zeros(n_i)]
    els.x2, els.y2 = np.r_[P[1:, 0], xs[1:]], np.r_[P[1:, 1], np.zeros(n_i)]
    bemcs.standardize_els_geometry(els, reorder=False)
    prof = slip_profile(0.5 * (xi[:-1] + xi[1:]))
    return bemcs.Mesh(
        els,
        bc_type=["s"] * n_f + ["h"] * n_i,
        bc_value=np.r_[np.c_[SLIP[0] * prof, SLIP[1] * prof], np.zeros((n_i, 2))],
        mu_minus=MU1,
        mu_plus=np.r_[np.full(n_f, MU1), np.full(n_i, mu2)],
        basis={"slip": "hat", "force": force_basis},
    )


mesh = fault_mesh()
print(mesh)
_, residual = exact_stress(np.array([1.0]), np.array([0.5]))
print(f"exact solution: largest continuity residual of the image coefficients {residual:.1e}")

# %% geometry
fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
bemcs.plot_mesh(mesh, ax=ax, show_normals=True, offset=0.5)
ax.axhline(0, color="r", lw=1.5, zorder=0)
ax.plot(*np.c_[CENTRE - HALF * T_F, CENTRE + HALF * T_F], "k-", lw=2.5)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2, 2)
ax.set_ylim(-1.6, 0.8)
ax.text(-1.9, 0.45, f"material 2: $\\mu_2$ = {MU2:g}", fontsize=10)
ax.text(-1.9, -0.45, f"material 1: $\\mu_1$ = {MU1:g}", fontsize=10)
ax.set_title("geometry: fault (black, 's'), interface (red, 'h');\n"
             "dots: $\\mu$ below and above each interface element", fontsize=10)
ax = axes[1]
xi = np.linspace(-HALF, HALF, 200)
ax.plot(xi, SLIP[0] * slip_profile(xi), label="shear slip")
ax.plot(xi, SLIP[1] * slip_profile(xi), label="opening")
ax.set_xlabel("position along the fault")
ax.set_ylabel("prescribed slip")
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("prescribed slip: tapered to zero at both tips", fontsize=10)
fig.tight_layout()
plt.show()

# %% solve and compare the stress fields
x_trap = bemcs.solve(mesh, geometry=geometry, mu=mu, nu=NU)
X, Y = np.meshgrid(np.linspace(-2.0, 2.0, 161), np.linspace(-2.2, 1.2, 137))
f = bemcs.evaluate_fields(mesh, x_trap, X, Y, geometry=geometry, mu=mu, nu=NU, mu_field=mu_field)
bem = np.array([f["sxx"], f["syy"], f["sxy"]])
ex, _ = exact_stress(X, Y)
# mask points close to the fault (the exact quadrature and the BEM near field) for the maps
d_fault = np.abs((X - CENTRE[0]) * N_F[0] + (Y - CENTRE[1]) * N_F[1])
along = np.abs((X - CENTRE[0]) * T_F[0] + (Y - CENTRE[1]) * T_F[1])
near = (d_fault < 0.03) & (along < HALF + 0.03)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
names = (r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$")
for k in range(3):
    vmax = np.percentile(np.abs(ex[k][~near]), 98)
    for row, (field, label) in enumerate(((bem[k], "BEM (trapezoids)"), (ex[k], "exact (Dundurs)"))):
        ax = axes[row, k]
        im = ax.pcolormesh(X, Y, np.ma.masked_where(near, field), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto")
        ax.plot(*np.c_[CENTRE - HALF * T_F, CENTRE + HALF * T_F], "k-", lw=1.5)
        ax.axhline(0, color="0.3", lw=0.8, ls="--")
        ax.set_aspect("equal")
        ax.set_title(f"{label}: true {names[k]}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"true stresses, material 1 below ($\\mu_1$ = {MU1:g}), material 2 above ($\\mu_2$ = {MU2:g})")
fig.tight_layout()
plt.show()

# %% profiles above and below the interface, and convergence
xp = np.linspace(-2.0, 2.0, 201)
x_nod = bemcs.solve(fault_mesh(force_basis="nodal"), geometry=geometry, mu=mu, nu=NU)
mesh_nod = fault_mesh(force_basis="nodal")
profiles = {}
for yp in (0.3, -0.3):
    ex_p, _ = exact_stress(xp, np.full_like(xp, yp))
    fp_t = bemcs.evaluate_fields(mesh, x_trap, xp, np.full_like(xp, yp), geometry=geometry, mu=mu, nu=NU,
                                 mu_field=mu_field)
    fp_n = bemcs.evaluate_fields(mesh_nod, x_nod, xp, np.full_like(xp, yp), geometry=geometry, mu=mu, nu=NU,
                                 mu_field=mu_field)
    profiles[yp] = (ex_p, np.array([fp_t[k] for k in ("sxx", "syy", "sxy")]),
                    np.array([fp_n[k] for k in ("sxx", "syy", "sxy")]))

# convergence: error at points away from the fault and the interface
th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
P = np.r_[CENTRE + 0.9 * np.c_[np.cos(th), np.sin(th)], np.c_[np.linspace(-2, 2, 5), np.full(5, 0.5)]]
P = P[np.abs(P[:, 1]) > 0.2]
ex_c, _ = exact_stress(P[:, 0], P[:, 1])
h_list = [0.1, 0.05, 0.025]
conv = {"trapezoid": [], "nodal": [], "no contrast (trapezoid)": []}
ex_hom = None
for h in h_list:
    for name, fb, m2 in (("trapezoid", "trapezoid", MU2), ("nodal", "nodal", MU2),
                         ("no contrast (trapezoid)", "trapezoid", MU1)):
        m = fault_mesh(h, fb, m2)
        xh = bemcs.solve(m, geometry=geometry, mu=mu, nu=NU)
        fh = bemcs.evaluate_fields(m, xh, P[:, 0], P[:, 1], geometry=geometry, mu=mu, nu=NU,
                                   mu_field=lambda a, b, m2=m2: np.where(b > 0, m2, MU1))
        got = np.array([fh["sxx"], fh["syy"], fh["sxy"]])
        if m2 == MU1:
            if ex_hom is None:
                ex_hom, _ = exact_stress(P[:, 0], P[:, 1], mu2=MU1)  # the exact solution without contrast
            ref = ex_hom
        else:
            ref = ex_c
        conv[name].append(np.abs(got - ref).max() / np.abs(ref).max())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, yp in zip(axes[:2], (0.3, -0.3)):
    ex_p, tr, nd = profiles[yp]
    for k, (name, col) in enumerate(zip(names, ("C0", "C1", "C2"))):
        ax.plot(xp, ex_p[k], "-", color=col, lw=1.6, label=f"{name} exact")
        ax.plot(xp[::6], tr[k][::6], "o", color=col, ms=3.5, mfc="none", label=f"{name} trapezoids")
        ax.plot(xp[::6], nd[k][::6], "x", color=col, ms=4, label=f"{name} nodal (tents)")
    ax.set_xlabel("x")
    ax.set_ylabel("true stress")
    ax.grid(alpha=0.3)
    side = "above the interface (material 2)" if yp > 0 else "below the interface (material 1)"
    ax.set_title(f"profile at y = {yp:+g}, {side}", fontsize=10)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=8)
ax = axes[2]
for name, style in zip(conv, ("o-", "s-", "^--")):
    ax.loglog(h_list, conv[name], style, label=name)
h = np.array(h_list)
ax.loglog(h, conv["trapezoid"][0] * h / h[0], "k:", lw=1, label="first order")
ax.loglog(h, conv["nodal"][0] * (h / h[0]) ** 2, "k--", lw=1, label="second order")
ax.set_xlabel("element length")
ax.set_ylabel("max relative stress error")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.set_title("convergence: trapezoids first order with a contrast,\ntents (nodal) second order", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1))
plt.show()
print("convergence (h = 0.1, 0.05, 0.025):")
for name, e in conv.items():
    print(f"  {name:24s}: " + "  ".join(f"{v:.1e}" for v in e))

# %% [markdown]
# **Reading the figures.** The BEM stresses match the exact bimaterial solution on both sides of the
# interface. Without a contrast the fault alone converges very fast (about 1e-5 at h = 0.05). With a
# contrast the default trapezoids converge at first order in plane strain: they hold the force constant
# over each collocated element, and on the interface a tangential force produces a normal traction
# through a 1/x kernel, which turns that into an O(h) error (tasks.md P1b). The optional "nodal" (tent)
# force basis is second order on straight interfaces like this one.
