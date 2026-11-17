# %% [markdown]
# # Strike-slip fault in a layered half-space (antiplane, heterogeneous)
# A compliant layer (shear modulus MU1, thickness H) over a stiff half-space (MU2 = 4 MU1), with a
# vertical strike-slip fault that breaks the free surface. Two faults, with the same slip profile
# (1 at the surface, tapering smoothly to 0 at depth D):
# - **fault in the layer**, D = 0.6 H;
# - **fault through the interface**, D = 1.63 H: it cuts through the material contrast.
#
# The material interface is a line of force elements (`BC_type = 'h'`) whose force density is
# solved for so that the true traction is continuous across it (exact at collocation points). The
# kernels use one reference modulus (`mu = 1`); every element lists the modulus on its -n and +n
# side (`mu_minus`, `mu_plus`).
#
# **Check against an exact solution.** For this geometry the surface displacement has a classical
# image-series solution (Rybicki 1971; Segall 2010, ch. 5). Unit slip from the surface to depth d:
#
# - d < H: G(x, d) = B(d) + sum_m kappa^m [B(2mH + d) - B(2mH - d)];
# - d > H: G(x, d) = (1 - kappa) sum_m kappa^m B(d + 2mH),
#
# with B(x, d) = atan(d / x) / pi (homogeneous half-space) and kappa = (MU1 - MU2) / (MU1 + MU2).
# A tapered slip s(d) is a superposition of these: u(x) = integral of -s'(d) G(x, d) over d.
#
# **Input files.** Run `ConstructMesh.py` in this folder first. For each fault it writes a mesh file
# (geometry and moduli of every element) and two connectivity files that carry the boundary
# conditions: one row per slip hat (`s` on the fault, `t` on the free surface; the hats at the
# fault-surface triple junction list the third branch in `e4`) and one row per force trapezoid (`h`).

# %% imports
import numpy as np
import matplotlib.pyplot as plt
import bemcs
from ConstructMesh import BASIS, D_CROSS, D_IN, H, H_EL, L, MU1, MU2, layered_mesh, slip_profile

geometry = "antiplane"
mu = 1.0  # reference shear modulus of the kernels (any value works)
CASES = {"fault in the layer": ("fault_in_layer", D_IN),
         "fault through the interface": ("fault_through_interface", D_CROSS)}


def mu_field(x, y):
    """True shear modulus, used to turn reference-medium stresses into true stresses."""
    return np.where(y > -H, MU1, MU2)


# %% the exact solution: image series


def image_series(x, D, mu1=MU1, mu2=MU2, n_quad=400):
    """Surface displacement of slip_profile(., D) in the layered half-space (x != 0)."""
    kappa = (mu1 - mu2) / (mu1 + mu2)
    m_max = 1 if kappa == 0 else int(np.ceil(np.log(1e-17) / np.log(abs(kappa)))) + 1

    def B(d):
        return np.arctan(d / x) / np.pi

    def G(d):
        if d < H:
            return B(d) + sum(kappa**m * (B(2 * m * H + d) - B(2 * m * H - d)) for m in range(1, m_max))
        return (1 - kappa) * sum(kappa**m * B(d + 2 * m * H) for m in range(m_max))

    u = np.zeros_like(x, dtype=float)
    g, w = np.polynomial.legendre.leggauss(n_quad)
    edges = [0.0, D] if D < H else [0.0, H, D]  # G has a kink at d = H
    for lo, hi in zip(edges[:-1], edges[1:]):
        for d, wd in zip(0.5 * (lo + hi) + 0.5 * (hi - lo) * g, 0.5 * (hi - lo) * w):
            ds = -0.5 * np.pi / D * np.sin(np.pi * d / D)  # slope of slip_profile
            u += wd * (-ds) * G(d)
    return u


# %% read the meshes and solve
solutions = {}
for name, (path, D) in CASES.items():
    # mesh file + connectivity files with the boundary conditions (see ConstructMesh.py)
    mesh = bemcs.Mesh.from_csv(f"{path}.csv", basis=BASIS,
                               connectivity={"slip": f"{path}_slip_connectivity.csv",
                                             "force": f"{path}_force_connectivity.csv"})
    print(f"{name}: fault crossings on the interface: {mesh.fault_crossings.tolist()}")
    x = bemcs.solve(mesh, geometry=geometry, mu=mu)
    solutions[name] = (mesh, x, D)
    print(f"{name}: {mesh}")

mesh, x, D = solutions["fault through the interface"]
fig, ax = plt.subplots(figsize=(10, 4))
bemcs.plot_mesh(mesh, ax=ax, show_normals=False, offset=0.6)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-2.0, 0.3)
ax.set_title("fault through the interface: mesh near the fault (black: slip, red: interface);\n"
             r"dots: $\mu$ below and above each interface element, squares: $\mu$ of the free surface")
plt.show()

# %% surface displacement: BEM vs the exact solution
xs = np.r_[-np.linspace(4, 1e-3, 120), np.linspace(1e-3, 4, 120)]
surface = {}
for name, (mesh, x, D) in solutions.items():
    u = bemcs.evaluate_fields(mesh, x, xs, np.full_like(xs, -1e-9), geometry=geometry, mu=mu)["uz"]
    surface[name] = (u, image_series(xs, D), image_series(xs, D, mu2=MU1))

# convergence with element size, and the effect of where the free surface and the interface end
# (meshes built in memory, same model)
xc = np.r_[-np.geomspace(5, 0.02, 15), np.geomspace(0.02, 5, 15)]


def surface_error(D, **kw):
    m = layered_mesh(D, **kw)
    xm = bemcs.solve(m, geometry=geometry, mu=mu)
    um = bemcs.evaluate_fields(m, xm, xc, np.full_like(xc, -1e-9), geometry=geometry, mu=mu)["uz"]
    return np.abs(um - image_series(xc, D)).max()


h_list = [0.1, 0.05, 0.025]
L_list = [10.0, 30.0, 100.0, 300.0, 1000.0]
conv = {name: [surface_error(D, h=h) for h in h_list] for name, (_, _, D) in solutions.items()}
trunc = {name: [surface_error(D, L=Lh) for Lh in L_list] for name, (_, _, D) in solutions.items()}
# bemcs's default force layout: plain stride-2 trapezoids through the crossing (lopsided).
# solve warns about this (TrapezoidCrossingWarning) and suggests the fix used above.
trunc_default = [surface_error(D_CROSS, L=Lh, lopsided=True) for Lh in L_list]

# %% plot
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
ax = axes[0, 0]
for i, (name, (u, ref, homog)) in enumerate(surface.items()):
    ax.plot(xs, ref, "-", color=f"C{i}", lw=1.5, label=f"{name}: exact (image series)")
    ax.plot(xs[::6], u[::6], "o", color=f"C{i}", ms=3.5, label=f"{name}: BEM")
    ax.plot(xs, homog, "--", color=f"C{i}", lw=1, label=f"{name}: homogeneous ($\\mu_2 = \\mu_1$)")
ax.set_xlabel("x / H")
ax.set_ylabel("surface displacement / surface slip")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper right")
ax.set_title(f"(a) surface displacement; layer $\\mu_1$ = {MU1:g} over half-space $\\mu_2$ = {MU2:g}",
             fontsize=10)

ax = axes[0, 1]
for i, (name, (u, ref, _)) in enumerate(surface.items()):
    ax.semilogy(xs, np.abs(u - ref), ".-", color=f"C{i}", ms=3, lw=0.8, label=name)
ax.set_xlabel("x / H")
ax.set_ylabel("|BEM - exact|")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.set_title(f"(b) error at the surface (element length {H_EL:g} H near the fault)", fontsize=10)

ax = axes[1, 0]
for i, (name, err) in enumerate(conv.items()):
    ax.loglog(h_list, err, "o-", color=f"C{i}", label=name)
ax.set_xlabel("element length near the fault / H")
ax.set_ylabel("max |BEM - exact| for |x| < 5 H")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.set_title(f"(c) convergence with the element length near the fault\n"
             f"(lines to |x| = {L:g} H)", fontsize=10)

ax = axes[1, 1]
for i, (name, err) in enumerate(trunc.items()):
    ax.loglog(L_list, err, "o-", color=f"C{i}", label=name)
ax.loglog(L_list, trunc_default, "s--", color="C1", mfc="none",
          label="fault through the interface, default trapezoids (stride through the crossing)")
ax.set_xlabel("half-width L of the free surface and the interface / H")
ax.set_ylabel("max |BEM - exact| for |x| < 5 H")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.set_title(f"(d) where the lines end (element length {H_EL:g} H near the fault):\n"
             "beyond ~100 H truncation no longer matters", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Reading the figure.** (a) The compliant layer concentrates the deformation near the fault
# compared with a homogeneous half-space (dashed). The BEM matches the image series for both faults,
# ~2e-5 at the default element length, and converges as the elements near the fault shrink (c).
#
# (d) **Where the lines end.** The free surface and the interface are truncated, with clamped
# (tapered) ends, so the comparison with the half-space solution needs them to end far from the
# fault. Log-spaced elements make that cheap: beyond ~100 H the truncation no longer affects the error
# near the fault, and the default is 1000 H.
#
# (d) **Layout of the force trapezoids at the crossing.** The force connectivity files restart the
# trapezoid stride where the fault crosses the interface, so collocation points flank the crossing on
# both sides. The dashed curve uses bemcs's default layout instead: plain stride-2 trapezoids run
# through the crossing, with a collocation point on one side only, and `solve` warns about it. The
# problem is antisymmetric in x, so the true interface force density is odd and its net force is zero;
# the lopsided layout breaks that and leaves a net force of ~0.02, whose far field grows like log r: a
# nearly uniform offset of the surface displacement, 1-2% and growing with L.

# %% true stress and traction continuity, fault through the interface
mesh, x, D = solutions["fault through the interface"]
_, (_, _, rows) = bemcs.solve(mesh, geometry=geometry, mu=mu, return_system=True)
x_colloc = mesh.els.x_centers[rows["element"][rows["type"] == "h"]]  # interface collocation points
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 300), np.linspace(-3, -1e-3, 200))
f = bemcs.evaluate_fields(mesh, x, X, Y, geometry=geometry, mu=mu, mu_field=mu_field)


def interface_traction(xq, dy=1e-6):
    """True normal traction syz just above and just below the interface."""
    kw = dict(geometry=geometry, mu=mu, mu_field=mu_field)
    return (bemcs.evaluate_fields(mesh, x, xq, np.full_like(xq, -H + dy), **kw)["syz"],
            bemcs.evaluate_fields(mesh, x, xq, np.full_like(xq, -H - dy), **kw)["syz"])


xi = np.linspace(-0.6, 0.6, 1201)
xi = xi[np.abs(xi) > 1e-4]
above, below = interface_traction(xi)
xk = x_colloc[np.abs(x_colloc) < 0.6]
above_k, below_k = interface_traction(xk)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
tau = np.hypot(f["sxz"], f["syz"])
im = ax.pcolormesh(X, Y, tau, cmap="magma", vmin=0, vmax=np.percentile(tau, 99), shading="auto")
ax.plot([0, 0], [0, -D], "c-", lw=2, label="fault")
ax.axhline(-H, color="w", lw=1, ls="--", label="interface")
fig.colorbar(im, ax=ax, label=r"true shear stress $\sqrt{\sigma_{xz}^2 + \sigma_{yz}^2}$")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="lower right")
ax.set_title(r"(e) fault through the interface: true stress ($\mu_1$ above, $\mu_2$ below)",
             fontsize=10)

ax = axes[1]
ax.plot(xi, above, "-", lw=1.5, label=r"$\sigma_{yz}$ just above the interface ($\mu_1$)")
ax.plot(xi, below, "--", lw=1.5, label=r"$\sigma_{yz}$ just below ($\mu_2$)")
ax.plot(xk, above_k, "ko", ms=4, mfc="none", label="collocation points (centres of the\n"
        "trapezoids' middle elements)")
ax.axvline(0, color="0.5", lw=0.8)
ax.text(0.01, ax.get_ylim()[0] * 0.9, "fault", color="0.4", fontsize=8)
ax.set_xlabel("x / H")
ax.set_ylabel("true normal traction on the interface")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("(f) traction across the interface: continuous at collocation points; not on the\n"
             "two elements at the crossing, where the true force density steps (x = 0)",
             fontsize=10)
fig.tight_layout()
plt.show()
print("traction mismatch at the interface collocation points: %.1e of the peak"
      % (np.abs(above_k - below_k).max() / np.abs(above).max()))
far = np.abs(xi) > 0.1
print("max traction mismatch for 0.1 < |x| < 0.6: %.1e of the peak"
      % (np.abs(above - below)[far].max() / np.abs(above).max()))

# %% [markdown]
# **The crossing point.** Across the fault the normal traction on the interface steps in proportion
# to the slip gradient, and the step differs above and below the interface (mu_1 vs mu_2), so the
# equivalent force density steps at the crossing. Continuous trapezoids cannot represent the step:
# with the split layout both elements next to the crossing are interpolated (the force is zero at
# the crossing node, the average of the two sides), and the traction is not continuous there (f).
# The error stays local because the layout is symmetric; in a problem without that symmetry a local
# error with a nonzero net force would again spread as log r. Allowing a step at fault crossings is an
# open task (tasks.md).
