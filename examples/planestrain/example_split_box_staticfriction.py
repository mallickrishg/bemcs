# %% [markdown]
# # Wavy fault in a box, sliding at static friction
# A toy version of a laboratory rock-friction experiment: a square elastic box (plane strain) cut by a
# gently sinusoidal fault. The top and bottom are sheared past each other, and also pushed together so
# that the fault is under compression. The fault slides at its frictional strength.
#
# **Boundary conditions**
# - top: u_x = +1, u_y = -delta; bottom: u_x = -1, u_y = +delta (shear plus confinement)
# - left and right sides: traction free
# - fault: static friction (below) and no opening (s_n = 0)
#
# **Sign conventions** (bemcs, for every element)
# - s is the unit tangent (from node 1 to node 2) and n the unit normal, s rotated by +90 degrees. On this
#   fault, which runs from x = -L to x = +L, s points along +x and n along +y (up).
# - slip = u(+n side) - u(-n side), in local (s, n) components: s_s > 0 means the upper block moved
#   along +s relative to the lower block.
# - traction t = sigma . n, the force per area that the +n (upper) block exerts on the -n (lower)
#   block. The stress is tension-positive, so t_n < 0 is compression, and the effective normal stress
#   is sigma_n = -t_n > 0.
#
# **Friction.** Friction drags the lower block along with the sliding upper block, so the shear
# traction on the lower block points along the slip and has magnitude f sigma_n:
#
#     t_s = sign(s_s) f sigma_n + tau_0,   i.e.   t_s + sign(s_s) f t_n = tau_0
#
# with tau_0 an optional residual stress. The slip direction is not known in advance: the example
# guesses it from the loading and repeats the solve until the guess and the slip agree. The law only
# makes sense where the fault is in compression (t_n < 0) and slides (s_s != 0) along the assumed
# direction; the checks at the end of the solve verify this. (With much stronger confinement parts of
# the fault would stick, which this linear, everywhere-sliding model cannot represent.)
#
# **Assembly.** `bemcs.solve` has no friction boundary condition, so the friction rows are built from
# two `bemcs.assemble` systems: the fault as ("t_local", "s") gives the shear-traction and no-opening
# rows, and the fault as ("t_local", "t_local") gives the normal-traction rows that are added to them.

# %%
import numpy as np
import matplotlib.pyplot as plt

import bemcs

plt.rcParams.update({"font.size": 12})

# %% Parameters
mu, nu = 1.0, 0.25
geometry = "planestrain"
friction_coeff = 0.2
delta = 0.5  # confinement: top and bottom pushed together by delta each
tau_0 = 0.0  # residual (pre-) stress on the fault: t_s + sign(s_s) f t_n = tau_0
L, H = 4.0, 4.0  # box half-width and half-height
n_per_unit = 1  # box elements per unit length
n_fault = 19  # fault elements

# %% Geometry: the box (clockwise from the top-left corner) and the fault across the middle
corners = np.array([[-L, H], [L, H], [L, -H], [-L, -H], [-L, H]])
pts = [
    np.linspace(a, b, int(round(np.hypot(*(b - a)) * n_per_unit)), endpoint=False)
    for a, b in zip(corners[:-1], corners[1:])
]
box = np.vstack(pts + [corners[:1]])

xf = np.linspace(-L, L, n_fault + 1)
yf = 0.1 * np.sin(np.pi * xf / 2)
yf[[0, -1]] = 0.0  # the fault ends on box nodes (triple junctions)

els = bemcs.initialize_els()
els.x1 = np.r_[box[:-1, 0], xf[:-1]]
els.y1 = np.r_[box[:-1, 1], yf[:-1]]
els.x2 = np.r_[box[1:, 0], xf[1:]]
els.y2 = np.r_[box[1:, 1], yf[1:]]
bemcs.standardize_els_geometry(els, reorder=False)
n_els = len(els.x1)

top = np.isclose(els.y_centers, H)
bot = np.isclose(els.y_centers, -H)
sides = np.isclose(np.abs(els.x_centers), L)
fault = ~(top | bot | sides)
fe = np.flatnonzero(fault)

# %% Boundary conditions
bc_type = np.empty((n_els, 2), dtype=object)
bc_type[top | bot] = "u"
bc_type[sides] = "t"
bc_value = np.zeros((n_els, 2))
bc_value[top] = (1.0, -delta)
bc_value[bot] = (-1.0, delta)
bc_value[fault, 0] = tau_0

bc_friction = bc_type.copy()
bc_friction[fault] = ("t_local", "s")  # rows: shear traction, normal slip (= 0)
bc_normal = bc_type.copy()
bc_normal[fault] = ("t_local", "t_local")  # only its normal-traction rows are used

common = dict(labels="slip", bc_value=bc_value, mu_minus=mu, mu_plus=mu)
mesh = bemcs.Mesh(els, bc_type=bc_friction.astype(str), **common)
mesh_tn = bemcs.Mesh(els, bc_type=bc_normal.astype(str), **common)

fig, ax = plt.subplots(figsize=(6, 6))
bemcs.plot_mesh(mesh, ax=ax, show_moduli=False)
ax.set_title("mesh and unit normals")
plt.show()

# %% Friction system
kw = dict(geometry=geometry, mu=mu, nu=nu)
A0, b, rows = bemcs.assemble(mesh, **kw)
A_tn, _, rows_tn = bemcs.assemble(mesh_tn, **kw)


def row_of(r, e, c):
    """Index of the BC row of element e, component c (0: shear, 1: normal)."""
    return np.flatnonzero((r["element"] == e) & (r["component"] == c))[0]


i_shear = np.array([row_of(rows, e, 0) for e in fe])  # t_s rows of the friction system
T_s = A_tn[[row_of(rows_tn, e, 0) for e in fe]]  # x -> t_s at the fault collocation points
T_n = A_tn[[row_of(rows_tn, e, 1) for e in fe]]  # x -> t_n at the fault collocation points
assert np.allclose(A0[i_shear], T_s)


def solve_friction(slip_sign):
    """Solve with t_s + slip_sign f t_n = tau_0 on the fault."""
    A = A0.copy()
    A[i_shear] += friction_coeff * slip_sign[:, None] * T_n
    return np.linalg.solve(A, b), np.linalg.cond(A)


slip_sign = np.ones(len(fe))  # guess: the top moves along +x, which is +s on this fault
for iteration in range(10):
    x, cond = solve_friction(slip_sign)
    slip_centre = mesh.node_values(x, "slip", geometry=geometry, frame="local")[fe, 1]  # (s_s, s_n)
    if (np.sign(slip_centre[:, 0]) == slip_sign).all():
        break
    slip_sign = np.sign(slip_centre[:, 0])
else:
    raise RuntimeError("the slip direction did not settle: parts of the fault probably stick")

t_s, t_n = T_s @ x, T_n @ x
sigma_n = -t_n
print(f"solved in {iteration + 1} iteration(s); condition number {cond:.3g}")
print(f"  friction residual  max |t_s - sign(s_s) f sigma_n - tau_0| = "
      f"{np.abs(t_s - slip_sign * friction_coeff * sigma_n - tau_0).max():.1e}")
print(f"  no opening         max |s_n| = {np.abs(slip_centre[:, 1]).max():.1e}")
print(f"  fault compressed   sigma_n in [{sigma_n.min():.3f}, {sigma_n.max():.3f}]: {(sigma_n > 0).all()}")
print(f"  friction resists   t_s s_s >= 0 everywhere: {(t_s * slip_centre[:, 0] >= 0).all()}")
if not (sigma_n > 0).all():
    print("  WARNING: parts of the fault are in tension; friction is not meaningful there (raise delta)")

# %% Slip and tractions along the fault
xs_, ys_, slip = mesh.values_on_elements(x, "slip", geometry=geometry, n_eval=10, frame="local")
on_fault = np.repeat(fault, 10)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(xs_[on_fault], slip[on_fault, 0], "b-", label="$s_s$ (slip)")
ax.plot(xs_[on_fault], slip[on_fault, 1], "g-", label="$s_n$ (opening)")
ax.plot(els.x_centers[fe], t_s, "bo", label="$t_s$")
ax.plot(els.x_centers[fe], sigma_n, "gs", label=r"$\sigma_n = -t_n$")
ax.plot(els.x_centers[fe], friction_coeff * sigma_n + tau_0, "k--", lw=1,
        label=r"$f\,\sigma_n + \tau_0$ (strength)")
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xlabel("x")
ax.set_ylabel("slip or traction")
ax.set_xlim(-L, L)
ax.legend()
ax.grid(alpha=0.3)
ax.set_title("fault: slip, and tractions at the collocation points")
plt.show()

# %% Shear stress and pressure in the box, with the displacement field
width = 2.001
fig, axes, fields = bemcs.plot_fields(
    mesh, x, **kw, xlim=(-width, width), ylim=(-1, 1), n=(80, 40),
    # mean stress, tension positive like every stress here: negative under confinement
    derived={"pressure": lambda f: 0.5 * (f["sxx"] + f["syy"])},
    components=["sxy", "pressure"], cmap="PiYG",
)
skip = (slice(None, None, 4), slice(None, None, 4))
axes[0, 0].quiver(fields["x"][skip], fields["y"][skip], fields["ux"][skip], fields["uy"][skip],
                  width=0.002, scale=30)
fig.set_size_inches(12, 9)
plt.show()
