# %% [markdown]
# # Strike-slip fault in a layered medium with a lateral contrast across the fault (antiplane)
# The realistic setting for a mature strike-slip fault: the crust is layered (shear modulus increasing
# with depth), and the fault juxtaposes different rocks, so the layering differs on its two sides
# (e.g. a sedimentary basin on one side, crystalline rock on the other). This combines the two
# Rybicki (1971) examples in this folder:
#
# - `example_Rybicki_layered.py`: a layer over a half-space, the same on both sides of the fault;
# - `example_Rybicki_bimaterial.py`: two quarter-spaces in contact, the fault on (or near) the contact.
#
# **Model.** A vertical fault at x = 0 breaks the free surface, with slip 1 at the surface tapering
# smoothly to 0 at depth D. Horizontal layer interfaces at depths `DEPTHS`; the moduli of the layers
# (top to bottom, the last one the half-space) are `MU_LEFT` for x < 0 and `MU_RIGHT` for x > 0.
#
# **Mesh.** Lines of elements, each listing the modulus on its -n and +n side:
# - the fault, slip elements (`'s'`, prescribed slip);
# - the free surface, slip elements (`'t'`, traction-free); it meets the fault at a triple junction;
# - the vertical contact x = 0 between the two stacks, force elements (`'h'`) from the surface to depth
#   L. Along the fault the slip and force elements coincide (same nodes);
# - the layer interfaces on each side, force elements (`'h'`), from the contact out to |x| = L.
#
# Each layer interface meets the contact at a **junction** of four force elements. The force density of
# each branch takes its own value there, so every branch ends with a free (open) end, like the contact's
# end at the free surface (`mesh.force_junctions`, `mesh.force_open_ends`; bemcs's default trapezoid
# layout does this). Tapering the branches to zero at the junction instead gave 3-4x larger errors (part 1:
# 2.6e-2 and 1.1e-2 instead of 8.6e-3 and 3.0e-3).
#
# **Odd segments.** Stride-2 trapezoids with a free end at both ends tile only an odd number of
# elements. A segment between two break points (here: the contact between the free surface and the
# first junction, and between junctions) with an even number of elements leaves the node next to one
# end pinned to zero (`PinnedForceWarning`), which cost a factor 5 in accuracy in part 2. So the layer
# depths are odd multiples of the element length, and the convergence studies refine by a factor 3,
# which keeps them odd.
#
# **Exact check.** If the stack on the right is the one on the left scaled by a constant c
# (mu_right(z) = c mu_left(z)), the solution is u = a S for x < 0 and u = b S for x > 0, where S is the
# solution without the lateral contrast (the layered medium mu_left on both sides), a = 2c / (1 + c) and
# b = 2 / (1 + c): u jumps by the slip across the fault (a + b = 2) and the traction across the contact
# is continuous (mu_left a = c mu_left b), while the layer interfaces and the free surface hold on each
# side separately. For one layer over a half-space S is Rybicki's image series, so this case is exact.
# Part 1 checks it; part 2 is a realistic stack with no such shortcut.

# %% imports
import warnings

import numpy as np
import matplotlib.pyplot as plt
import bemcs

geometry, mu = "antiplane", 1.0  # mu: reference modulus of the kernels (any value works)
D = 1.0  # fault depth (lengths are in units of D)


def slip_profile(depth):
    """Slip cos^2(pi depth / 2D): 1 at the surface, zero slip and slope at depth D."""
    depth = np.asarray(depth, float)
    return np.where(depth < D, np.cos(0.5 * np.pi * depth / D) ** 2, 0.0)


# %% the mesh


def graded(h, L, h_max, flat=2.0, ratio=1.08):
    """0, h, 2h, ... up to `flat`, then log-spaced (each element 8% longer) up to h_max, out to L."""
    r, step = list(np.round(np.arange(0.0, flat + 1e-12, h), 12)), h
    while r[-1] < L:
        r.append(r[-1] + step)
        step = min(step * ratio, h_max)
    return np.array(r)


def layered_medium_mesh(mu_left, mu_right, depths, h=0.05, L=300.0, h_max=None, flat=2.0):
    """Fault, free surface, vertical contact and layer interfaces (see the top of this file).

    mu_left, mu_right : moduli of the layers top to bottom, the last one the half-space
        (len(depths) + 1 values each).
    depths : depths of the layer interfaces. Those above D must be multiples of h, so that the
        fault and the contact have a node on each junction.
    h : element length on the fault and near it (|x|, depth < flat).
    L : extent of the free surface and the layer interfaces (|x| < L) and of the contact (depth < L).
    h_max : longest element (log-spaced elements stop growing there); default 5x the thinnest
        layer, which keeps collocation points of neighbouring lines more than 0.1 element lengths
        from each other (`CollocationProximityWarning`).
    """
    depths = np.asarray(depths, float)
    gaps = np.diff(np.r_[0.0, depths])
    h_max = 5 * gaps.min() if h_max is None else h_max
    r = graded(h, L, h_max, flat=flat)
    n_f = int(round(D / h))
    zf = -np.round(np.linspace(0.0, D, n_f + 1), 12)
    for d in depths[depths < D]:
        if not np.isclose(zf, -d).any():
            raise ValueError(f"layer interface at depth {d} is not a fault node (h = {h})")

    def mu_at(stack, z):  # modulus at depth -z
        return np.asarray(stack, float)[np.searchsorted(depths, -np.asarray(z), side="right")]

    rows = []  # (x1, y1, x2, y2, bc, mu_minus, mu_plus, value) per line

    def line(x, y, bc, mu_minus, mu_plus, value=0.0):
        n = len(x) - 1
        rows.append((x[:-1], y[:-1], x[1:], y[1:], [bc] * n, np.broadcast_to(mu_minus, n),
                     np.broadcast_to(mu_plus, n), np.broadcast_to(value, n)))

    zm = 0.5 * (zf[:-1] + zf[1:])
    # the fault and the contact run downward: unit normal +x, so -n is the left side
    line(np.zeros(n_f + 1), zf, "s", mu_at(mu_left, zm), mu_at(mu_right, zm), slip_profile(-zm))
    xs = np.r_[-r[:0:-1], r]
    xm = 0.5 * (xs[:-1] + xs[1:])
    top = np.where(xm < 0, mu_left[0], mu_right[0])
    line(xs, np.zeros_like(xs), "t", top, top)
    zc = -r
    zcm = 0.5 * (zc[:-1] + zc[1:])
    line(np.zeros_like(zc), zc, "h", mu_at(mu_left, zcm), mu_at(mu_right, zcm))
    for k, d in enumerate(depths):
        # left to right: unit normal up, so -n is the layer below
        line(-r[::-1], np.full(len(r), -d), "h", mu_left[k + 1], mu_left[k])
        line(r, np.full(len(r), -d), "h", mu_right[k + 1], mu_right[k])

    cols = [np.concatenate([np.asarray(rw[i]) for rw in rows]) for i in range(8)]
    els = bemcs.initialize_els()
    els.x1, els.y1, els.x2, els.y2 = (c.astype(float) for c in cols[:4])
    bemcs.standardize_els_geometry(els, reorder=False)
    return bemcs.Mesh(els, bc_type=cols[4], bc_value=cols[7].astype(float), mu_minus=cols[5].astype(float),
                      mu_plus=cols[6].astype(float), basis={"slip": "hat", "force": "trapezoid"})


def surface_displacement(mesh, x_obs):
    x = bemcs.solve(mesh, geometry=geometry, mu=mu)
    return bemcs.evaluate_fields(mesh, x, x_obs, np.full_like(x_obs, -1e-9), geometry=geometry, mu=mu)["uz"]


# %% [markdown]
# ## Part 1: proportional lateral contrast, against the exact solution
# One layer (thickness H = 0.5 D, so the fault cuts through it) with modulus 1 over a half-space of
# modulus 4 on the left; the same stack times c = 2 on the right.


def image_series(x, D, H, mu1, mu2, n_quad=400):
    """Rybicki's image series: surface displacement of slip_profile in a layer (mu1, thickness H)
    over a half-space (mu2), the same on both sides of the fault (see example_Rybicki_layered.py)."""
    k = (mu1 - mu2) / (mu1 + mu2)
    m_max = 1 if k == 0 else int(np.ceil(np.log(1e-17) / np.log(abs(k)))) + 1

    def B(d):
        return np.arctan(d / x) / np.pi

    def G(d):
        if d < H:
            return B(d) + sum(k**m * (B(2 * m * H + d) - B(2 * m * H - d)) for m in range(1, m_max))
        return (1 - k) * sum(k**m * B(d + 2 * m * H) for m in range(m_max))

    u = np.zeros_like(x, dtype=float)
    g, w = np.polynomial.legendre.leggauss(n_quad)
    edges = [0.0, D] if D < H else [0.0, H, D]
    for lo, hi in zip(edges[:-1], edges[1:]):
        for d, wd in zip(0.5 * (lo + hi) + 0.5 * (hi - lo) * g, 0.5 * (hi - lo) * w):
            u += wd * 0.5 * np.pi / D * np.sin(np.pi * d / D) * G(d)
    return u


H1, MU_L1, MU_L2, C = 0.5, 1.0, 4.0, 2.0


def exact_proportional(x, c=C):
    return np.where(x < 0, 2 * c / (1 + c), 2 / (1 + c)) * image_series(x, D, H1, MU_L1, MU_L2)


xs = np.r_[-np.linspace(4, 1e-3, 120), np.linspace(1e-3, 4, 120)]
xc = np.r_[-np.geomspace(5, 0.02, 15), np.geomspace(0.02, 5, 15)]
mesh1 = layered_medium_mesh([MU_L1, MU_L2], [C * MU_L1, C * MU_L2], [H1], h=0.1 / 3)
print(f"part 1: {mesh1}")
print(f"  junctions: {mesh1.force_junctions.tolist()}, open force ends: {mesh1.force_open_ends.tolist()}")
u1 = surface_displacement(mesh1, xs)
h_list = [0.1, 0.1 / 3, 0.1 / 9]  # contact segment surface-junction: 5, 15, 45 elements
err1 = [np.abs(surface_displacement(layered_medium_mesh([MU_L1, MU_L2], [C * MU_L1, C * MU_L2], [H1], h=h), xc)
               - exact_proportional(xc)).max() for h in h_list]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ax = axes[0]
bemcs.plot_mesh(mesh1, ax=ax, show_normals=False, offset=0.15)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.6, 0.2)
ax.set_aspect("equal", adjustable="box")
ax.set_title("(a) mesh near the fault (black: slip, red: force)\n"
             r"dots: $\mu$ on each side of the force elements", fontsize=10)
ax = axes[1]
ax.plot(xs, exact_proportional(xs), "k-", lw=1.5, label="exact")
ax.plot(xs[::5], u1[::5], "o", color="C0", ms=3.5, label="BEM")
ax.plot(xs, image_series(xs, D, H1, MU_L1, MU_L2), "--", color="0.5", lw=1,
        label="no lateral contrast (c = 1)")
ax.set_xlabel("x / D")
ax.set_ylabel("surface displacement / surface slip")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title(f"(b) layer {MU_L1:g} over {MU_L2:g} (x < 0), times c = {C:g} (x > 0)", fontsize=10)
ax = axes[2]
ax.loglog(h_list, err1, "o-", label="BEM vs exact")
ax.loglog(h_list, err1[0] * np.array(h_list) / h_list[0], "k:", lw=1, label="first order")
ax.set_xlabel("element length near the fault / D")
ax.set_ylabel("max |BEM - exact| for |x| < 5 D")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.set_title("(c) convergence", fontsize=10)
fig.tight_layout()
plt.show()
print("part 1 errors (h = %s): %s" % (h_list, ", ".join(f"{e:.1e}" for e in err1)))

# %% [markdown]
# **Reading part 1.** The BEM matches the exact solution, converging at first order (3e-3 at
# h = D / 90). As in `example_Rybicki_bimaterial.py`, the error comes from the corner where the contact
# (a force layer with non-zero density at its end) meets the free surface; grading the elements towards
# that corner reduces it. The displacement jumps by the slip across the fault, but the stiffer side
# moves less: b / a = 1 / c.
#
# ## Part 2: a realistic stack
# Three layers over a half-space. On the left, crystalline rock stiffening with depth; on the right, a
# compliant sedimentary basin over the same basement. Below the deepest interface both sides are the
# same half-space, so the contact carries no force there.

DEPTHS = [0.15, 0.4, 0.75]  # layer interfaces: segments 0.15, 0.25, 0.35 = 3, 5, 7 elements of 0.05
MU_LEFT = [1.0, 1.5, 2.2, 3.0]  # crystalline side, top to bottom (last: half-space)
MU_RIGHT = [0.3, 0.7, 1.6, 3.0]  # basin side


def stack_text(stack):
    return ", ".join(f"{m:g}" for m in stack)


mesh2 = layered_medium_mesh(MU_LEFT, MU_RIGHT, DEPTHS)
print(f"part 2: {mesh2}")
with warnings.catch_warnings():
    warnings.simplefilter("error", bemcs.CollocationProximityWarning)
    x2 = bemcs.solve(mesh2, geometry=geometry, mu=mu)
u2 = bemcs.evaluate_fields(mesh2, x2, xs, np.full_like(xs, -1e-9), geometry=geometry, mu=mu)["uz"]
u_left_only = surface_displacement(layered_medium_mesh(MU_LEFT, MU_LEFT, DEPTHS), xs)
u_right_only = surface_displacement(layered_medium_mesh(MU_RIGHT, MU_RIGHT, DEPTHS), xs)
u_hom = surface_displacement(layered_medium_mesh([1.0] * 4, [1.0] * 4, DEPTHS), xs)

# self-convergence: element length near the fault, refined by 3 (segments stay odd); a shorter
# mesh (L = 100, fine elements for |x| < 1) keeps the finest one affordable
h_ref = 0.05 / 9
conv_kw = dict(L=100.0, flat=1.0)
u_ref = surface_displacement(layered_medium_mesh(MU_LEFT, MU_RIGHT, DEPTHS, h=h_ref, **conv_kw), xc)
h2 = [0.05, 0.05 / 3]
err2 = [np.abs(surface_displacement(layered_medium_mesh(MU_LEFT, MU_RIGHT, DEPTHS, h=h, **conv_kw), xc)
               - u_ref).max() for h in h2]


def mu_field(x, y):
    """True shear modulus, to turn reference-medium stresses into true stresses."""
    k = np.searchsorted(np.asarray(DEPTHS), -np.asarray(y), side="right")
    return np.where(np.asarray(x) < 0, np.asarray(MU_LEFT)[k], np.asarray(MU_RIGHT)[k])


X, Y = np.meshgrid(np.linspace(-2.0, 2.0, 320), np.linspace(-1.6, -1e-3, 160))  # no point on the fault
f2 = bemcs.evaluate_fields(mesh2, x2, X, Y, geometry=geometry, mu=mu, mu_field=mu_field)

# %% plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
im = ax.pcolormesh(X, Y, mu_field(X, Y), cmap="viridis", shading="auto")
ax.plot([0, 0], [0, -D], "w-", lw=2.5, label="fault")
for d in DEPTHS:
    ax.axhline(-d, color="w", lw=0.6, ls="--")
fig.colorbar(im, ax=ax, label=r"shear modulus $\mu$")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="lower right")
ax.set_title(f"(d) the model: left {stack_text(MU_LEFT)};\nright (basin) {stack_text(MU_RIGHT)}",
             fontsize=10)

ax = axes[0, 1]
ax.plot(xs, u2, "-", color="C3", lw=2, label="layered, with the lateral contrast")
ax.plot(xs, u_left_only, "--", color="C0", lw=1, label="crystalline stack on both sides")
ax.plot(xs, u_right_only, "--", color="C2", lw=1, label="basin stack on both sides")
ax.plot(xs, u_hom, ":", color="k", lw=1, label="homogeneous half-space")
ax.set_xlabel("x / D")
ax.set_ylabel("surface displacement / surface slip")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("(e) surface displacement: the basin side takes most of the offset", fontsize=10)

ax = axes[1, 0]
tau = np.hypot(f2["sxz"], f2["syz"])
im = ax.pcolormesh(X, Y, tau, cmap="magma", vmin=0, vmax=np.nanpercentile(tau, 99), shading="auto")
ax.plot([0, 0], [0, -D], "c-", lw=2)
for d in DEPTHS:
    ax.axhline(-d, color="w", lw=0.6, ls="--")
fig.colorbar(im, ax=ax, label=r"true shear stress $\sqrt{\sigma_{xz}^2 + \sigma_{yz}^2}$")
ax.set_aspect("equal")
ax.set_title("(f) true shear stress around the fault", fontsize=10)

ax = axes[1, 1]
ax.semilogy(xs, np.abs(u2 - u_left_only), "-", color="C0", lw=1, label="vs crystalline stack on both sides")
ax.semilogy(xs, np.abs(u2 - u_hom), "-", color="k", lw=1, label="vs homogeneous half-space")
ax.set_xlabel("x / D")
ax.set_ylabel("|difference in surface displacement|")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
ax.set_title("(g) differences of up to ~0.15 of the slip near the fault;\n"
             f"self-convergence vs h = D/180: {err2[0]:.1e} (h = D/20), {err2[1]:.1e} (h = D/60)",
             fontsize=10)
fig.tight_layout()
plt.show()
print("part 2 self-convergence vs h = %g: %s" % (h_ref, ", ".join(f"{e:.1e}" for e in err2)))

# %% [markdown]
# **Reading part 2.** The surface displacement is strongly asymmetric: the compliant basin side moves
# more than the crystalline side (0.64 vs 0.36 of the slip next to the fault), and neither laterally
# uniform model (dashed) captures it: they differ by up to ~0.15 of the slip near the fault, and a
# homogeneous half-space overestimates the far-field displacement on both sides. The stress (f) is concentrated around the fault and
# jumps across the layer interfaces and the contact, following the moduli. The discretization error (self-convergence, g) is small compared with these
# differences.
#
# **Meshing notes.** Interface depths above D are fault nodes, so each junction is a node of the fault,
# the contact and both layer interfaces. Elements away from the fault are log-spaced, capped at five
# times the thinnest layer so that the free surface and the interfaces stay resolved relative to each
# other. The default force layout restarts the trapezoids at every junction and gives each branch a
# free end there, and an open end where the contact meets the free surface. Segments of the contact
# between break points have an odd number of elements (see the top of this file).
