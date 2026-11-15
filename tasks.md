# bemcs – capability review and heterogeneous-elasticity tasks

Branch: `het-planestrain`. Status is based on reading the code. Run everything in the `bemcs` conda env
(`conda run -n bemcs python ...`); the base env's numpy is broken (missing `libgfortran`) and must not be used.

---

## 1. What the code base can already do

### 1.1 Slip elements (displacement-discontinuity sources)

| Capability | Plane strain | Antiplane | Where |
| --- | --- | --- | --- |
| Constant slip | yes | – | `bemcs.constant_kernel`, `displacements_stresses_constant_no_rotation`, `get_displacement_stress_kernel_constant` |
| Quadratic slip, C1 (slip-gradient) continuous across elements; 6 DOF/element (3 nodes x shear/normal) | yes | yes (3 DOF/element) | `bemcs.quadratic_kernel_farfield/_coincident`, `displacements_stresses_quadratic_*`, `get_displacement_stress_kernel*`, `get_matrices_slip_slip_gradient*`, `construct_smoothoperator*` |
| Open ends, overlapping nodes, triple junctions (`label_nodes`) | yes | yes | `bemcs.label_nodes`, `construct_smoothoperator*` |
| Boundary conditions on slip elements | slip (`'s'`), traction (`'t'`) | slip, traction, displacement (`'u'`) | `get_traction_kernels*`, `bemAssembly.construct_linearoperator_slip_antiplane` |
| Analysis / plotting | yes | yes | `plot_displacements_stresses_els`, `get_slipvector_on_fault*`, `get_strainenergy_from_stress`, `bemAssembly.plot_BEM_field` |
| Numba-accelerated kernels | – | yes | `bemAssembly.get_displacement_stress_kernel_slip_antiplane` (`@njit`) |
| **New:** "quadratic hat" slip basis, 3-element patches, one hat per patch, 2 DOF/patch (shear, normal) | yes | – | `bemAssembly.compute_coefs_quadratichatslip_planestrain`, `get_kernels_quadratichatslip_planestrain` |

### 1.2 Force elements (body-force / Kelvin sources)

| Capability | Plane strain | Antiplane | Where |
| --- | --- | --- | --- |
| Linear-force line elements | yes | yes | `bemcs.displacements_stresses_linear_force_no_rotation_{planestrain,antiplane}`, `get_displacement_stress_kernel_force_{planestrain,antiplane}` |
| Constant-force triangles (volume sources) | yes | yes | `bemcs.displacements_stresses_triangle_force_{planestrain,antiplane}` (+ `_nearfield` / `_farfield`) |
| Trapezoidal force basis, 3-element patches built from linear line elements | yes (new, partly validated) | yes | `bemAssembly.compute_coefs_trapezoidalforce_planestrain`, `get_kernels_trapezoidalforce_planestrain`, `get_kernels_trapezoidalforce_antiplane` |

### 1.3 Documentation status

- The README only links to two wiki tutorials (homogeneous plane-strain, and antiplane heterogeneous).
  **The heterogeneous-elasticity theory is not in the README.** The formulation below should be added there
  (or to the wiki) once the plane-strain version is validated.
- `point_source/` holds the Mathematica/MATLAB derivations behind the kernels (reference only).

---

## 2. Heterogeneous elasticity: formulation

Idea: for piecewise-uniform `mu` the effect of `grad(mu)` on the material interfaces is replaced by
an equivalent body force `f` on those interfaces, so a homogeneous-medium Green's function
can be used everywhere. `f` is itself an unknown that must be consistent with the strain it produces.

### 2.1 Antiplane (implemented)

Normalized body force `phi = f/mu = -(mu_,n/mu) * psi` with `psi = eps_{zj} n_j` (the strain
equivalent of traction). Unknowns: slip coefficients `s` (3/element) and force coefficients `f`
(one per force patch). Rows, in order:

1. `'s'` rows: slip at collocation node = prescribed value.
2. `'t'` rows: `Psi^(s) s + Psi^(f) f = psi_given`. Slip elements may also feel the force elements.
3. `'h'` rows: `alpha Psi^(s) s + (I + alpha Psi^(f)) f = 0`, with `alpha = mu_,n/mu`.
4. Smoothness rows (open nodes, overlaps, triple junctions) for slip elements, RHS = 0.

Implemented in `bemAssembly.solveAntiplaneBEM(fileinput, connectivityfile)`:
mesh CSV columns `x1,z1,x2,z2,BC_type,value` (for `'h'` rows `value` holds `alpha`), connectivity CSV
lists 3-element force patches. Demo: `examples/antiplane/HeterogeneousElasticity_forces.py`.

### 2.2 Plane strain (to implement / validate)

Assumes uniform Poisson's ratio `nu`, so heterogeneity is a single scalar field `mu(x)`.

Hooke: `sigma_ij = 2 mu eps_ij + (2 mu nu/(1-2nu)) delta_ij eps_kk`.

Equivalent body force: `phi_i = f_i/mu = -(mu_,n/mu) * ( psi_i + n_i * nu/(1-2nu) * eps_kk )`,
with `psi_i = eps_ij n_j`. Unlike antiplane, the volumetric strain enters.

Kernel hierarchy (per collocation element `i`, source DOF `l`, source element `j`):

| Symbol | Meaning | Built from |
| --- | --- | --- |
| `Sigma` | stress (xx, yy, xy) | existing stress kernels |
| `E` | strain | `C^-1 Sigma` |
| `Psi` | projected strain `eps_km n_m` (x, y) | `P_i E`, `P_i = [[nx,0,ny],[0,ny,nx]]` |
| `Lambda` | volumetric strain `eps_kk` | `[1 1 0] E` |
| `Psi~` | augmented projected strain | `Psi + nu/(1-2nu) n_k Lambda` |

Row types (all `2x2` blocks; slip DOFs local `{t,n}`, force DOFs and `'h'`/`'t'` rows global `{x,y}`):

- `'s'`: `I s = s_given`
- `'t'`: `Psi^(s) s + Psi^(f) f = psi_given`
- `'h'`: `alpha Psi~^(s) s + (I + alpha Psi~^(f)) f = 0`

Square system of `2(N_s + N_f)` equations.

---

## 3. Current state of the plane-strain implementation

- `examples/utilfunctions.py::solve_planestrain_heterogeneous` (uncommitted) assembles and solves
  the system above with hat-slip + trapezoidal-force patches. It lives in `examples/`, not in the package.
- `examples/testing_HeterogeneousMaterialsPlanestrain.py` (uncommitted): 3 slip + 3 force
  elements, one patch each, uniform shear slip, `alpha = 0.5`.
- Not yet done: any validation against an independent solution, a `'t'`-type test, multi-patch
  meshes, non-planar interfaces, a user-facing mesh/CSV interface like the antiplane one.

---

## 4. Issues spotted during review (verify, then fix)

1. ~~**Compliance uses the wrong Poisson ratio**~~ **SOLVED and tested (2026-11-16):** `bemAssembly.plane_strain_compliance`
   uses `nu` and inverts plane-strain Hooke's law exactly (`test_compliance_inverts_plane_strain_hooke`). For the
   example geometry (horizontal fault and interface, shear slip) the solution is unchanged, because there the projected
   strain does not depend on nu'. The notes still carry the plane-stress form; fix them. Original: bug in the notes and the code, confirmed.
   Notes and `compute_strains` use `nu' = nu/(1+nu)` with `C^-1 = 1/(2mu) [[1-nu', -nu'],[-nu', 1-nu']]`.
   Inverting the plane-strain Hooke's law above gives `eps_kk = (1-2nu) sigma_kk/(2mu)` and
   `eps_xx = (1/(2mu)) [(1-nu) sigma_xx - nu sigma_yy]`, i.e. `nu' = nu`. The `nu/(1+nu)` form is the plane-*stress*
   compliance. At `nu = 0.25` this is 0.2 vs 0.25. The `'t'` and `'h'` rows (and `Lambda`) are all affected.
   Verified numerically in the `bemcs` env: inverting the Hooke matrix at `nu = 0.25` gives
   `[[0.75, -0.25], [-0.25, 0.75]] / (2mu)`, not `[[0.8, -0.2], ...]`.
   Location: `compute_strains` in `examples/utilfunctions.py`.
2. **Observation-point side for `'h'` rows.** Plane strain uses `x_c + dr*n` with `dr = -1e-9`
   for slip and force elements alike. Antiplane uses `x_c - dr*n` for force elements. Stress from a
   line force is discontinuous across the interface, so the side matters. Decide
   what `psi` should be sampled on (probably each side's value, or an average, related to how `alpha` is defined) and
   document it.
3. **Definition of `alpha` at a sharp interface.** `alpha = mu_,n/mu` is a delta function at a jump.
   The code takes `alpha_force` as an input. Write down how it relates to `(mu1, mu2)` (and which
   side of `psi` multiplies it) and add a helper to compute it.
4. ~~**`'s'` rows differ from the notes**~~ **SOLVED and tested:** the `'s'` rows are now exact. They are slip at the
   collocation centre, computed from the hat basis (centre row of the local slip node matrix times `T`), instead of a
   +-1e-9 displacement jump. They are not the identity in general, because neighbouring hats are non-zero at each
   other's centres; one-ring hats are cardinal only for their own centre (`test_each_hat_has_unit_slip_at_its_own_centre`).
   Test: `test_alpha_zero_gives_no_force_and_the_prescribed_slip`. Original: Notes say `I s = s_given`. Code evaluates slip as a displacement
   jump across `+/-1e-9` from both slip and force kernels (force kernels are continuous so it cancels, but
   it is wasteful). Check the hat-basis DOF really equals slip at the central element, and that the `'s'` row can
   be replaced by the identity.
5. ~~**Test script calls `bemcs.plot_BEM_field`**~~ **SOLVED** (export added; the script runs end to end). Original: the script calls it, but it is defined in `bemAssembly` and not exported from
   `bemcs/__init__.py`. Confirmed by running the script in the `bemcs` env: it solves fine (cond. number ~1.18) and then fails at the first plot call with `AttributeError`. Use `GF.plot_BEM_field`, or export it.
   **Fixed (stage 2):** `plot_BEM_field` is exported from `bemcs`; the script runs end to end.
6. ~~**Trapezoidal force basis was previously flagged as unreliable**~~ **SOLVED and tested (2026-11-16).**
   `tests/test_operators.py` checks continuity in the global frame and the taper to 0 at open ends, and that
   `A T = 0`. `tests/test_get_kernels.py` checks that trapezoids are exact combinations of the linear-force kernels,
   which match Kelvin/Laplace point-force quadrature in both geometries. `tests/test_near_field.py` checks they are
   bounded near every support node. Original note: (commit `8fd5a0f`: "weird issues"; later
   commits fixed stride and arctan bugs). It still needs a standalone check against the constant-force or
   line-force result.
7. ~~**Duplicated functions**~~ **SOLVED in stage 3 (2026-11-16):** the duplicates were deleted, and `test_no_redundancy_between_modules` passes. Original: duplicated functions in `bemcs.py` and `bemAssembly.py` (audited, see section 6). The 3 slip/gradient
   helpers and the 2 antiplane kernels agree numerically. The plane-strain force kernel does not (item 9).
8. **Minor** (partly solved: `pkg_resources` was replaced by `importlib.metadata` in stage 3): a `1-2nu` typo in the notes' last sentence, `__init__.py` uses deprecated `pkg_resources`, and
   `solve_planestrain_heterogeneous` has `print(np.linalg.cond(A))` on every call.
9. ~~**`bemcs.get_displacement_stress_kernel_force_planestrain` is wrong for rotated elements**~~ **SOLVED in stage 3 (2026-11-16):** deleted; plane-strain force goes through the generated kernels, which are tested against Kelvin quadrature at 4 angles. Original: the `bemcs.py` version is wrong for any
   non-horizontal element (confirmed).** It agrees with the numba version only for horizontal elements, and differs
   by up to ~0.16 (stresses) for a 30-degree element. The numba version in `bemAssembly` matches an independent
   reference (Gauss-Legendre integration of the Kelvin point source) to ~1e-9. Cause: `rotate_vector` applies
   `rot_mats_inv` (global->local) to displacements instead of `rot_mats`. The stress rotation also differs for
   general angles. Nothing in the repo calls the buggy version (the heterogeneous solver goes through
   `get_kernels_trapezoidalforce_planestrain`, which uses the correct numba kernel), but it is public API.
   Its docstring also lists the wrong return order (`sxx, sxy, syy, ux, uy`; it actually returns `ux, uy, sxx, syy, sxy`).
10. ~~**numba is a hard dependency; 140 s JIT**~~ **SOLVED in stage 3 (2026-11-16):** numba is optional: without it there is a no-op `njit`, the numpy path works, and
    `use_numba=True` raises `ImportError` (tested in a subprocess). All kernels are cached and compile in 1-2 s; the
    full `--numba` suite takes 12 s from a cold cache. Original: numba was a hard dependency despite the `try/except` at the top of `bemAssembly.py`: line 16
    `from numba import njit, prange` runs unconditionally. The plane-strain force numba kernel has no `cache=True`,
    so it costs **~140 s of JIT compilation** in every fresh Python process (it dominates the test-suite runtime).
11. ~~**Unexported public functions**~~ **SOLVED in stage 3 (2026-11-16):** `KNOWN_UNEXPORTED` is empty. The Kelvin/Laplace point sources are exported;
    the triangle helpers were renamed `_private`. Original: 11 public functions in `bemcs.py` were not exported from `bemcs/__init__.py`, including
    `kelvin_point_source_disp/_stress` and `get_displacement_stress_kernel_constant` (list in `tests/test_api.py::KNOWN_UNEXPORTED`).
12. ~~**A third copy of kernels lives in `examples/antiplane/GF.py`**~~ **SOLVED** (file removed) (it also defines
    `get_displacement_stress_kernel_force_planestrain`, among others).
13. ~~**Invalid escape sequences**~~ **SOLVED in stage 3 (2026-11-16):** raw strings; the package compiles with warnings as errors. Original: `bemcs.py` lines 3009, 3036 and 3063 use `"$\sigma_{..}$"` in non-raw strings (invalid escape, `DeprecationWarning`,
    a `SyntaxWarning` on Python ≥ 3.12).
14. ~~**`get_kernels_trapezoidalforce_antiplane` ignores `mu`**~~ **SOLVED in stage 3 (2026-11-16):** deleted; `get_force_kernels` scales with `1/mu` (`test_displacement_scales_with_one_over_mu`). Original: it ignores its `mu` argument (it always passes `mu=1.0` to the element
    kernel), so displacements are wrong by a factor `1/mu` for `mu != 1`. `get_kernels(..., basis="trapezoid")` passes `mu`
    through; the old function is compared at `mu = 1` only.
15. ~~**Debug prints in library code**~~ **SOLVED in stage 3 (2026-11-16):** `label_nodes` defaults to `verbose=False`, the old builders are gone, and
    `import bemcs` prints nothing (`test_import_is_quiet`). Remaining, user-facing: a "no connectivity file" note in
    `solveAntiplaneBEM`, and the condition-number prints in `examples/planestrain/utilfunctions.py` (T1). Original: `label_nodes` prints node counts, and the hat/trapezoid coefficient builders print
    `patch number:` / `trapezoid number:` for every patch. `bemAssembly` prints the numba version on import.
    Partly fixed: `label_nodes(els, verbose=True)` gained a `verbose` flag, and the new code calls it quietly.
    The trapezoid builder no longer prints. The old hat builder (deleted in stage 3) and the numba import message remain.
18. ~~**Points on an element return the two-sided average**~~ **SOLVED and tested (convention, both
    geometries, slip and force:** `test_points_on_an_element_give_the_two_sided_average`). In-plane stress (and slip
    displacement) jumps across an element. The old kernels returned one side or the other, decided by the sign of a
    ~1e-17 rounding error in the local y. Plane-strain kernels now treat |y_local| <= 1e-12 (a + |xi|) as on the
    element and return the mean of the two sides: exactly for slip (y = 0), and by evaluating at y = +-1e-300 for
    force. To get one side, offset the point explicitly (as the solvers do with `dr`). Antiplane slip and force
    follow the same convention (two-sided average via y = +-1e-300) since the near-field work below.
19. ~~**Force-element collocation count**~~ **SOLVED (decision: trapezoids only; enforced and tested by
    `test_mesh_continuity_operator_rejects_linear_force`).** Original question: Linear force with continuity rows (open ends
    force = 0, overlaps continuous) plus one collocation row per element and direction over-determines an open chain
    by 2 rows per direction. A closed loop is square. Options: collocate at interior nodes instead of element centres
    (square for chains and loops; the nodal "tent" functions are then the force analogue of slip hats), or keep
    trapezoids (1 DOF per 3-element patch). Decide before building the general-form `'h'` rows.
    **Decided 2026-11-16: force uses trapezoids only.** There is no general-form force solve:
    `Mesh.continuity_operator` raises for basis "linear" force. Linear force remains the element kernel that
    trapezoids are built from, and `continuity_rows(kind="force")` is kept to check that trapezoids are continuous.
16. ~~**Quadratic slip kernel loses float64 precision near every element endpoint**~~ **SOLVED and tested
    (2026-11-16):** both geometries are at the float64 floor (`tests/test_kernel_precision.py`,
    `tests/test_near_field.py`); the optional curvature form stays in the task list. History:
    `quadratic_kernel_farfield` (via `displacements_stresses_quadratic_no_rotation`) vs the same formula at 50 digits:
    relative error 1e-11 at d = 1e-2 L, 1e-7 at 1e-3 L, 2e-4 at 1e-4 L, O(1) at d <= 1e-5 L, and NaN directly above
    an endpoint (probably `np.sign(y / (a - x))` with `a - x` rounding to 0; not confirmed). Cause: large
    polynomial x log/arctan terms nearly cancel. Element-centre collocation is safe. Points within ~1e-3 L of a node
    are not, e.g. force collocation next to a fault that ends on an interface. No basis (hats, `null_space`, full 6N)
    avoids this, since they all combine the same element kernels.
    **Mostly fixed (2026-11-16).** The main loss came from evaluating `log` and the denominators from the *expanded*
    polynomial `a^2 -+ 2ax + x^2 + y^2` (and its square). The new generated formulas (`_quadratic_slip_f`,
    `_linear_force_g` in `bemAssembly`) compute `R = (a -+ x)^2 + y^2` directly. Error vs 50 digits: 4e-12 at 1e-2 L,
    1.5e-10 at 1e-3 L, 4e-8 at 1e-4 L, 3e-7 to 2e-5 at 1e-5 L, with no NaN directly above the endpoint
    (`tests/test_kernel_precision.py`). Below ~1e-6 L the remaining polynomial x log cancellation dominates again.
    **Fixed for slip (2026-11-16, endpoint rewrite).** `_quadratic_slip_f` is now expanded about the element endpoint
    nearest the observation point (`x = s a + xi`, `s = +-1`). The drivers compute `(xi, y)` by rotating
    `obs - endpoint`, so near-node quantities are never differences of large numbers. Per element: ~4e-16 relative
    down to 1e-8 a, and unchanged mid-element and in the far field (1e-10 at 100 a, 2e-7 at 1000 a; far-field
    cancellation remains). The force formulas were already well conditioned near endpoints (2e-10 at 1e-8 a) and
    expanding them made them worse, so they were kept.
    **Sum over elements** (hat basis approaching a shared node, float64 vs 50 digits): the error now follows the float64
    floor ~1e-16 L/d, i.e. 8e-13 at 1e-4, 8e-11 at 1e-6, 9e-9 at 1e-8. Before: 1e-5 to 1e-7 on a straight fault, and
    24 / 3e4 / 7e6 on a 30-degree kink. Only analytic cancellation at the node (curvature form) could go below
    ~1e-16 L/d.
    **Kinks: physics, not numerics (decided).** The exact stress near a kinked node grows ~0.9 per decade (log
    singularity) and is now resolved accurately. The mesh is C0, so geometry-induced singularities remain at kinks and
    triple junctions, with strength set by the angle between neighbouring elements (discrete curvature). True
    cancellation holds only for co-planar elements. This is documented in the README and in the `get_kernels`,
    `get_slip_kernels` and `element_kernels_slip` docstrings.
17. ~~**Triple junctions with all branches in or all out crash**~~ **SOLVED and tested:** `label_nodes` raises a
    `ValueError` naming the junction, its elements and their orientation, and says to reverse one branch and flip its BC
    signs (`test_unsupported_triple_junction_orientation_is_rejected`; good orientations still give full-rank rows).
    Original: crash, confirmed 2026-11-16.
    `label_nodes` assigns the junction signs only when 1 element ends at J and 2 start there, or the reverse. With all 3
    starting at J or all 3 ending there, the junction gets no negative sign, and `construct_smoothoperator` fails with
    `ValueError: could not broadcast input array from shape (0,36)`. Tested on a 120-degree star:
    1-in/2-out and 2-in/1-out both give 24 full-rank rows; all-in and all-out both crash.
    **Decision:** do not handle these orientations. Element orientation is the user's input. Reversing a branch also
    flips its shear/normal sign convention, so its BC values must change with it, and the user should make that choice.
    Reject such meshes with a clear error instead (task in T2).

---

## 5. Task list

### T0. Environment
- [x] Use the `bemcs` conda env (numpy 2.3.4, numba 0.62.1; `import bemcs` works). Do not use the base env (broken numpy).

### TA. Code audit and API consolidation (do before T1)
- [x] Add a `pytest` suite skeleton (`tests/`, `pytest.ini`); run with
  `/Users/mallickrishg/miniconda3/envs/bemcs/bin/python -m pytest`. Current status (after stage 3): 270 passed and 28 skipped by
  default, ~5 s; with `--numba` 298 passed in ~12 s from a cold cache.
  **Numba tests are opt-in.** They are marked `@pytest.mark.numba` and run only with `pytest --numba`. While developing,
  never trigger JIT; test through the numpy path. Pay the compile once, when the code is functional. (The numba cache
  is invalidated by any edit to `bemAssembly.py`; we keep the kernels in that module anyway.)
- [x] Static API checks (`tests/test_api.py`): `__init__` exports resolve, no function defined twice in a module,
  no undefined global names, list of duplicates across modules, list of unexported functions.
- [x] Numerical equivalence of duplicated pairs (`tests/test_duplicates_consistent.py`).
- [x] Plane-strain force kernels vs an independent Kelvin reference (`tests/test_force_kernels_reference.py`),
  at 0, 90, 30 and -40 degrees.
- [x] Agree on the package layout and the unified kernel API (see section 7).
- [x] Stage 1: `bemcs.get_kernels` + `KernelInfo` in `bemAssembly.py`, tested in `tests/test_get_kernels.py`
  (argument errors, shapes/info, mixed labels, numpy vs numba, old vs new) and against the Kelvin reference.
- [x] Stage 1b: three levels in `bemAssembly.py`:
  - level 1: `Mesh`, `get_kernels(x, y, mesh, geometry=, mu=, nu=)`;
  - level 2: `get_slip_kernels`, `get_force_kernels`;
  - level 3: `element_kernels_slip/force`, `basis_transform`, `hat_coefficients`, `trapezoid_coefficients`.

  Details:
  - Constant bases are derived. Derived constant slip matches Crouch & Starfield to 1e-15.
  - Every basis works in both geometries; antiplane hat is new.
  - `get_displacement_stress_kernel_constant` moved to `tests/reference_kernels.py`.

  Tests (`tests/test_get_kernels.py`, `tests/test_force_kernels_reference.py`):
  - Lagrange-node jumps of quadratic slip;
  - continuity of force kernels;
  - smoothness of the hat and trapezoid transforms;
  - mu scaling;
  - linear and constant force vs point-force quadrature, in both geometries and both backends;
  - mesh validation, ordering and global connectivity.

  Note: `hat` still takes a user-supplied 3-element `connectivity`; the one-ring hats of section 7.4 replace it in T2.
- [x] Stage 2 (2026-11-16):
  - **Level-3 operators**, all in interleaved columns:
    - `slip_node_matrices`, `force_node_matrix`;
    - `continuity_rows`, with clamped open slip nodes (slip and gradient, section 7.4);
    - `tip_elements`, `traction_kernels`, `kernel_matrix`, `values_on_elements`.
  - **Mesh methods:** `continuity_operator`, `tip_elements` / `bc_elements` (tip elements get no BC row),
    `n_src` / `columns` / `n_columns`, `element_coefficients`, `values_on_elements`.
  - **`hat_coefficients`** is rebuilt on the interleaved node matrices, one implementation for both geometries.
  - **Antiplane slip kernel** split into a formula plus numba/numpy drivers. The numpy path had been 30x slower
    through `.py_func`.
  - **Callers migrated** to the new API, with no calls left to functions scheduled for deletion:
    - all 8 `examples/*.py` scripts and `examples/utilfunctions.py`;
    - `solveAntiplaneBEM` and `construct_linearoperator_slip_antiplane`.
  - **Checks against pre-migration baselines.** With the old open-node rule, every example reproduced its baseline
    (slip and fields to 1e-16 to 3e-8 relative, except on-element in-plane stress; issue 18).
  - **Effect of clamped tips** (by design): 0.7-6% changes in examples whose faults have tips (19% for `sxy` in the
    free-surface example); the split box has no tips and is unchanged. Short faults change a lot: the
    3-element faults in `example_non_planar_fault_plotting.py` and `HeterogeneousElasticity_forces.py` lose the
    BCs of both or one end element. Their meshes should get one extra element per tip (decision for the user).
  - **Tests:** `tests/test_operators.py` checks against `construct_smoothoperator*` plus gradient rows at open nodes,
    `A T = 0` for hats and trapezoids, and the equation count `pe (N - N_tip) + rows = 3 pe N`.
  - **Not done here (T2):** one-ring hats; rejecting all-in/all-out junctions (issue 17). Force collocation count:
    issue 19.
- [x] Stage 3 (2026-11-16): deleted the old functions.
  - **From `bemcs.py`:**
    - the 4 `els`-based kernels, including the buggy plane-strain force kernel;
    - `get_matrices_slip_slip_gradient*` and `construct_smoothoperator*`;
    - `coeffs_to_disp_stress`, `get_traction_kernels*`, `get_slipvector_on_fault*`, `rotate_stress_antiplane`.
  - **From `bemAssembly.py`:**
    - the 3 monolithic numba kernels and the old patch kernels;
    - the old hat/trapezoid coefficient builders;
    - the duplicated slip helpers.
  - **`trapezoid_coefficients`** was rebuilt on `force_node_matrix`, one implementation for both geometries,
    identical to the old builders (difference 0.0).
  - **Tests:**
    - the old-vs-new comparisons and `tests/test_duplicates_consistent.py` were removed;
    - the frozen old continuity operators were kept as references in `tests/reference_operators.py`;
    - traction and slip-evaluation tests now compare against explicit formulas;
    - `KNOWN_DUPLICATES` and `KNOWN_UNEXPORTED` are empty, and the strict xfail now passes as a normal test.
  - **Examples:** all 8 give the same results as before the deletions.
  - **Note:** quadratic slip coefficients are slip values at s = -2w/3, 0, +2w/3, not at the element ends (documented
    in `KernelInfo`). The free-surface example had used them as end-node slip; fixed.
- [x] Fix issue 9 (numpy plane-strain force kernel): deleted in stage 3.
- [x] Add `cache=True` to every `@njit` (issue 10, first half).
- [x] Make numba an optional dependency (issue 10, second half; stage 3).
- [x] Rewrite the plane-strain force kernel and add a plane-strain slip numba kernel (2026-11-16):
  - **Formulas:** `_quadratic_slip_f` and `_linear_force_g` are scalar formulas generated with sympy from the old closed
    forms (common subexpressions only, plus factored `R`; see issue 16). They are shared by numba drivers
    (`_slip_planestrain_numba`, `_force_planestrain_numba`: `prange` over observation points, scalar maths,
    `error_model="numpy"`) and numpy drivers (vectorized over points, via `.py_func`).
  - **Results:** compile 1.9 s (slip) and 1.1 s (force), down from 142 s for the old force kernel. Runtime for
    5000 observation points x 200 elements:

    | | old numpy | new numpy | new numba |
    | --- | --- | --- | --- |
    | slip | 39 s | 0.89 s | 0.04 s |
    | force | – | 0.39 s | 0.02 s |

  - **Agreement:** numba vs numpy 3e-15; also matches the old kernels, the Kelvin quadrature and the analytic constant slip.
  - **Generator:** a scratchpad script (not in the repo) reads the *old* formulas. Once those are deleted in stage 3,
    the generated functions are the source of truth.
  - Original analysis, kept for reference. Measured first-call compile (no cache):
  plane-strain force 90 s serial / 142 s `parallel=True`; antiplane kernels 2-3 s. Cause: 10 inner closures with
  long symbolic expressions (62 log/arctan calls) are inlined into one function, and they operate on whole arrays,
  so numba builds a separate loop and temporary for every sub-expression (and `parallel=True` then analyses each
  one for fusion). Fix: loop over observation points with scalar math, put the formulas in small `@njit` helpers,
  and compute the shared logs/arctans once per point. Target: compile in seconds and no slower at run time. Guarded by
  `tests/test_force_kernels_reference.py` (Kelvin reference) and `test_numba_backend_matches_numpy`.
  Apply the same pattern to the antiplane kernels if it helps.
- [ ] Add reference tests for the remaining kernels: slip (plane strain and antiplane, constant and quadratic),
  antiplane force, triangle forces, hat-slip and trapezoid-force patches.
- [x] Near-field tests of the singularity-free patch bases (`tests/test_near_field.py`, 2026-11-16):
  - **Setup:**
    - hat slip and trapezoid force, in plane strain and antiplane, on a straight and a 30-degree kinked fault;
    - approach every node of the patch support from 4 directions (one grazing the line, 0.1 degree off), down to
      1e-8 L, against the same formulas in 50-digit mpmath.
  - **Asserts:**
    - straight fault: the exact stress is bounded and converges;
    - float64 matches the exact value to `1e-12 + 2e-15/d`, on both meshes;
    - kink node: log-singular (constant stress increment per decade) and still resolved in float64; other nodes
      bounded;
    - controls: a lone quadratic node basis grows like 1/d and a lone force ramp like log d.
  - **Mutation check:** swapping the original plane-strain closed form back in fails with 2e7 to 2e8 relative errors.
  - **Antiplane fix needed for these tests to pass:**
    - The antiplane slip formula had the same problem as the old plane-strain one: 3e6 error at 1e-8 L. It is now
      generated, expanded about the +w endpoint only (s = 1). Points nearer -w use the mirror symmetry: swap shape
      functions 1 and 3 and negate the x-derivatives, verified to 6e-16.
    - Keeping s symbolic left split terms such as `s w/R - w/R` that cancel catastrophically.
    - The antiplane force formula was extracted unchanged into `_antiplane_force_point` with numpy/numba drivers.
      It was already accurate near endpoints.
    - Antiplane drivers compile in about 1 s; numba vs numpy 2e-16; examples unchanged (<= 3e-11).
- [x] Fix near-endpoint precision of the quadratic slip kernel (issue 16): endpoint rewrite, see issue 16. Remaining,
  optional: the curvature form (section 8.4) to go below the ~1e-16 L/d float64 floor, and a check of the slip formula
  itself against quadrature of the point-dislocation solution. Previous status: mostly done: factored `R` gives
  ~1e-7 to 1e-5 relative error at 1e-5 L, and `tests/test_kernel_precision.py` sweeps 1e-1 ... 1e-5 L against 50-digit mpmath.
  Remaining: below ~1e-6 L (section 8.4, curvature form), and checking the formula itself against quadrature of the
  point-dislocation solution.
- [x] Resolve `KNOWN_UNEXPORTED` and `KNOWN_DUPLICATES` until both are empty (stage 3); the xfail marker is removed.
- [x] Remove `examples/antiplane/GF.py` (issue 12). It was removed with the antiplane examples that used it.

### T1. Correctness of plane-strain heterogeneous solver
- [x] Fix issue 4.1 (compliance) and re-run `testing_HeterogeneousMaterialsPlanestrain.py` (runs; same solution, see issue 1).
- [ ] Resolve 4.2 and 4.3 (side of evaluation, definition and helper for `alpha`).
- [x] Fix 4.5 so the test script runs end to end.
- [ ] Validate: ~~(a) `alpha = 0` must reproduce the homogeneous hat-slip solution~~ (done:
  `tests/test_heterogeneous.py` also checks that the `'h'` and `'t'` rows hold for the solved fields); (b) the antiplane
  limit; (c) a circular/layered inclusion with a known analytic solution, or an independent FEM result;
  (d) symmetry `mu1 <-> mu2` with sign-flipped `alpha`; (e) traction continuity across the interface.
- [x] Add a `'t'`-row test (prescribed projected strain): `test_t_rows_prescribe_the_projected_strain`.
- [x] Verify the trapezoidal-force kernels in isolation (issue 4.6; see the issue for the tests).

### T2. Package integration
- [x] Move `solve_planestrain_heterogeneous` into `bemcs/bemAssembly.py` (mirroring `solveAntiplaneBEM`). It uses the new
  kernel API, has exact `'s'` rows, and prints the condition number only when `verbose=True`.
- [ ] Build the mesh/connectivity interface (CSV: `x1,z1,x2,z2,BC_type,value`, `'h'` rows carry `alpha` or `(mu1, mu2)`).
- [ ] Support multiple patches, and patches that share nodes (junctions between slip and interface elements).
- [x] One-ring hats (section 7.4), 2026-11-16:
  - **API:** `basis="hat"` *without* connectivity builds one hat per non-tip element from the mesh topology
    (`one_ring_hat_coefficients`, `hat_elements`), at level 2 and in `Mesh`. An explicit 3-element connectivity still
    works, and equals the one-ring hat of the middle element on a straight run.
  - **Tests** (`tests/test_one_ring_hats.py`) on a straight fault, a 30-degree kink, a splay (1 in / 2 out) and a Y
    (2 in / 1 out), in both geometries:
    - equal to 3-element hats on a straight run;
    - number of hats = dim null(A); `A T = 0`; `rank T` full;
    - unit slip at their own centre;
    - the reduced solve equals the general solve;
    - a splay hat spans both main-fault elements at J.
- [x] Remove the duplicated functions (issue 4.7) and decide on the public API in `__init__.py` (stage 3).
- [x] Reject unsupported triple-junction orientations (issue 17) in `label_nodes` (and so in
  `mesh.continuity_operator()`). If all 3 elements at a junction start there or all 3 end there, raise a `ValueError`
  that names the junction coordinates and the 3 element indices, with their orientation (start/end at J). It should say
  that one branch must be reversed, and that its BC values change sign with it. Add a test for both bad cases, and
  check that 1-in/2-out and 2-in/1-out still pass.

### T3. Tests and docs
- [ ] Extend `tests/` with the validation cases from T1.
- [ ] Document heterogeneous elasticity (sections 2.1 and 2.2) in the README or wiki, plus a plane-strain tutorial.
- [x] Tidy up: `environment.yml` gains `mpmath` (the tests evaluate 50-digit references with it), `pkg_resources`
  is replaced, and library debug prints are removed.

### T4. Possible extensions (later)
- [ ] Spatially varying `nu` (currently assumed uniform).
- [ ] Heterogeneous domains with topography/free surface, combining with existing non-planar fault tools.
- [ ] Numba acceleration for the kernels that do not have it yet (plane-strain slip, triangles).

---

## 6. Duplicate-function audit (bemcs.py vs bemAssembly.py) — resolved in stage 3, kept for history

| Name | `bemcs.py` | `bemAssembly.py` | Agree? |
| --- | --- | --- | --- |
| `slip_functions`, `slipgradient_functions` | `(x, a)` | `(x, a, basis="quadratic")` | yes |
| `get_matrices_slip_slip_gradient` | `(els, flag, reference)` | `+ basis="quadratic"`; only `flag="node"` | yes (`flag="node"`) |
| `get_displacement_stress_kernel_slip_antiplane` | numpy, takes `els` | numba, takes arrays | yes |
| `get_displacement_stress_kernel_force_antiplane` | numpy, takes `els` | numba, takes arrays | yes |
| `get_displacement_stress_kernel_force_planestrain` | numpy, takes `els` | numba, takes arrays | **no**: numpy version wrong (issue 9) |

---

## 7. Unified kernel API (agreed design)

Decisions 2026-11-15:
- keep two modules;
- remove the old kernel functions (no deprecated wrappers);
- return displacements and stresses as N-D arrays plus an info object.

Decisions 2026-11-16:
- a three-level structure (mesh → source type → basis);
- constant bases derived from the higher-order ones;
- a general continuity-operator path alongside the reduced patch bases;
- a reference `mu` and a uniform `nu`, both required.

Work happens on branch `coderevamp` (from `main` after merging `het-planestrain`). All `examples/**/*.ipynb`
were converted to percent-format `.py` scripts (`# %%` cells, outputs dropped, display/autoreload magics
commented out), and `notebooks/`, `point_source/` and `examples/hrf/` are untracked and git-ignored (personal work).
`examples/antiplane/` keeps only `antiplane101.py` (+ `example_mesh.csv`), `ConstructMesh.py` and
`HeterogeneousElasticity_forces.py` (run `ConstructMesh.py` first to generate its two CSV inputs); both run cleanly.
When old functions are removed, the `examples/**/*.py` scripts that call them must be migrated.

### 7.1 Module rule

- **`bemcs/bemcs.py`: single-element physics and geometry.** It covers `els` construction and standardization,
  analytic element solutions in the local frame, basis/shape functions, rotations and point sources. It **never
  imports `bemAssembly`** (enforced by `tests/test_api.py`).
- **`bemcs/bemAssembly.py`: everything that combines elements.** That means all three levels below, the numba
  backends, continuity and BC operators (`construct_smoothoperator*` moves here), solvers and field plotting.
- `bemcs/__init__.py` re-exports the public API.
- Reference solutions used only to check the code live in `tests/` (`tests/reference_kernels.py`). These are the
  analytic constant-slip kernels (Crouch & Starfield) and the Kelvin point-force quadrature.
  `get_displacement_stress_kernel_constant` moves there; it is no longer part of the package.

### 7.2 Three levels

```
Level 1  mesh       bemcs.Mesh(...)  +  bemcs.get_kernels(x_obs, y_obs, mesh, geometry=, mu=, nu=)
                    what users work with: splits the mesh by source type, calls level 2, stacks [slip; force]
Level 2  source     bemcs.get_slip_kernels(...)   bemcs.get_force_kernels(...)
                    public, but users are pointed to level 1; tested extensively in their own right
Level 3  basis      element kernels (quadratic slip, linear force) x basis transform T
```

**Level 3: basis.** There are only two element kernel families, which are the highest-order bases:
- **Quadratic slip:** 3 nodes per element.
- **Linear force:** 2 nodes per element.

Every basis is an element kernel times a transform `T` (`n_elem_dof x n_basis_dof`, with a direction axis in plane
strain):

| kind | basis | DOF per slip/force component | `T` |
| --- | --- | --- | --- |
| slip | `quadratic` | 3 per element | identity |
| slip | `constant` | 1 per element | weights `[1, 1, 1]` on the element's nodes |
| slip | `hat` | 1 per element (one-ring patch, section 7.4) | one-ring hat coefficients from mesh topology |
| force | `linear` | 2 per element | identity |
| force | `constant` | 1 per element | weights `[1, 1]` on the element's ramps |
| force | `trapezoid` | 1 per 3-element patch | trapezoid coefficients from `connectivity` |

- Constant bases exist **only** as derived bases. The analytic constant kernels are kept in `tests/` to check that the
  quadratic/linear kernels and their weights are right.
- Every basis is available in both geometries. Antiplane gets constant slip and hat slip for free.
- Level-3 functions (`element_kernels_*`, `basis_transform`, `hat_coefficients`, `trapezoid_coefficients`) are public in
  `bemAssembly` but not re-exported from `bemcs`.

**Level 2: source type.**
```python
U, S, info = bemcs.get_slip_kernels(x_obs, y_obs, els, *, geometry, mu, nu=None,
                                    basis="quadratic", connectivity=None, use_numba=False)
U, S, info = bemcs.get_force_kernels(x_obs, y_obs, els, *, geometry, mu, nu=None,
                                     basis="linear", connectivity=None, use_numba=False)
```
- `els` contains only elements of that type.
- `connectivity` (n_patches, 3) is required for patch bases and rejected otherwise.

**Level 1: mesh.**
```python
mesh = bemcs.Mesh(els, labels,                         # labels: "s"/"f" per element
                  basis={"slip": "hat", "force": "trapezoid"},
                  connectivity={"slip": ..., "force": ...},   # global element indices
                  bc_type=None, bc_value=None, alpha=None)    # filled in with T1/T2
U, S, info = bemcs.get_kernels(x_obs, y_obs, mesh, *, geometry, mu, nu=None, use_numba=False)
A, rows = mesh.continuity_operator()                   # element-DOF bases only, see 7.4
```
- Default bases are `quadratic` (slip) and `linear` (force), i.e. the general element-DOF formulation.
- The source axis is ordered `x = [slip; force]`.
- CSV input and automatic patch generation from shared nodes come with T2.

### 7.3 Arguments, outputs and errors

- **Required keyword arguments.** `geometry` (`"antiplane"` | `"planestrain"`) and `mu` are required everywhere, with no
  defaults; `nu` is required for plane strain.
  - `mu` is the user's **reference** shear modulus and `nu` is uniform over the medium (variable `nu` is out of scope).
  - Heterogeneity is not a kernel argument. It is mesh data: `alpha = d(ln mu)/dn` on `'h'` (interface) elements.
    For a sharp interface alpha integrates to `ln(mu2/mu1)`; which side's `psi` it multiplies is still open
    (issues 4.2, 4.3).
- **Outputs.** Axis order follows the notes' `Sigma_ijkl` (observation, source coefficient, component, direction):

  | | displacements `U` | stresses `S` |
  | --- | --- | --- |
  | planestrain (4-D) | `(N_obs, N_src, 2, 2)`: `[ux, uy]` x `[t, n]` | `(N_obs, N_src, 3, 2)`: `[sxx, syy, sxy]` x `[t, n]` |
  | antiplane (3-D) | `(N_obs, N_src, 1)`: `[uz]` | `(N_obs, N_src, 2)`: `[sxz, syz]` |

  - `t`/`n` are element-local (shear/normal slip; along/normal force).
  - `N_src` is element-major then basis node, or patch index for patch bases.
  - A system-matrix block is `S[:, :, k, :].reshape(N_obs, -1)`, with columns interleaved `[t0, n0, t1, n1, ...]`.
    `get_matrices_slip_slip_gradient`, the continuity operator and the hat-coefficient builder all use this same order.
- **`info`** (`KernelInfo`) records:
  - geometry, basis, `mu`, `nu` and backend;
  - axis labels;
  - a per-`N_src` table: source label, element/patch index and basis node (`-1` for derived/patch bases).
- **Errors.**
  - Missing required keywords: `TypeError`.
  - Unknown values, unsupported combinations, or a patch basis without `connectivity`: `ValueError`.
  - `use_numba=True` without a numba kernel (plane-strain slip): `NotImplementedError`, with no silent fallback.

### 7.4 Two ways to impose continuity

The same element kernels support both. They differ only in where the continuity conditions go.

| | general (element DOFs) | reduced (patch bases) |
| --- | --- | --- |
| unknowns | element coefficients `c`: plane strain 6N (slip), 4N (force) | patch amplitudes `a`: 2(N - N_tip) for slip |
| continuity | explicit rows `A c = 0` stacked under the BC rows | built in: `c = T a` with `A T = 0` |
| triple junctions | handled by `A` | one-ring hats (below) |

Continuity rules (`mesh.continuity_operator()`, built on `label_nodes`):
- **Slip** (as in today's `construct_smoothoperator`):
  - open nodes (fault tips): **clamped**, slip = 0 and slip gradient = 0 (4 rows). Decided 2026-11-16: no singular
    crack tips. A tip element is a taper element: it gets no hat and no BC/collocation row, in either form, so a fault
    should extend one element past the region where BCs matter. (Today's `construct_smoothoperator` imposes slip only,
    2 rows; change it.)
  - overlap nodes: continuity of slip and slip gradient (4 rows);
  - triple junctions: slip kinematics and gradient (6 rows).
- **Force:** solved with **trapezoids only** (decided 2026-11-16, issue 19). They satisfy these rules by construction:
  - overlap nodes: force continuous;
  - open ends: force tapers to 0;
  - force elements never form triple junctions (raise if found).
- **Slip-force contact.** A node where a slip element meets a force element is treated as **open for both**: each tapers to
  0, and no continuity is enforced between the two element types.
- **One-ring hats** (reduced form for slip; verified in section 8.3). One hat per element and slip component:
  - support: the element plus every element sharing one of its end nodes (2 neighbours at a triple junction);
  - inside the support: the slip continuity rules above (overlap 4 rows, triple junction 6 rows);
  - patch-boundary nodes, including fault tips: slip = 0 and gradient = 0;
  - tip elements get no hat (clamped tips, above);
  - normalisation: unit shear or normal slip, local frame, at the element centre.

  Every patch system is square, including at triple junctions of any angle (section 8.3). With clamped tips in both
  forms, the hats span exactly the null space of `A`, so the reduced form gives the same solution as the general
  form, with 2(N - N_tip) unknowns. On a straight run of elements they equal today's 3-element hats; the new part
  is the patch that crosses a triple junction.

### 7.5 Implementation notes and stages

- numpy and numba share one source: `use_numba=False` calls the jitted function's `.py_func`.
- The numpy `els`-based duplicates in `bemcs.py` (including the buggy plane-strain force kernel, issue 9) are deleted,
  not fixed.
- Triangle (area) force sources are not `els`-based and stay separate for now.
- **Stages:**
  1. **[done]** A first `get_kernels(x_obs, y_obs, els, geometry, source, ...)` with tests (uncommitted at
     the time of this redesign).
  2. **1b:** restructure into the three levels above, with:
     - derived constant bases;
     - required `mu`/`nu`;
     - `Mesh` (labels, bases, connectivity);
     - reference kernels moved to `tests/`.
  3. **Tests for each level:**
     - level 3: weights/transforms (constant = sum of nodes vs analytic Crouch & Starfield; Kelvin reference for
       linear and constant force; hat and trapezoid satisfy `A T = 0`);
     - level 2: argument errors, shapes/info, numpy vs numba, old vs new;
     - level 1: mixed labels, global connectivity, ordering.
  4. **Stage 2:**
     - `mesh.continuity_operator()` in interleaved columns, replacing `construct_smoothoperator*`;
     - migrate the solvers and `examples/*.py`.
  5. **Stage 3:** delete the old functions and the old-vs-new comparison tests; empty `KNOWN_DUPLICATES`.

---

## 8. Findings: basis choice, triple junctions, near-node kernel precision (2026-11-16)

Scratch experiment (not in the repo): straight, 30-degree kinked, and triple-junction meshes. Compared the full 6N
system with constraint rows, `scipy.linalg.null_space` of the constraints (raw and row-scaled), the cardinal basis
`Z (C Z)^-1`, and "one-ring" hats (below).

### 8.1 Basis choice does not affect singularity cancellation
- All bases give the same centre slip, to ~1e-14 relative, for a crack under uniform shear traction. Constraint
  residuals are 1e-15 (hats) and 1e-14 (`null_space`), too small to leave a visible 1/r term.
- Off-fault stress approaching a node is also identical across bases, to 5-9 digits, *including where it is wrong*.
  Straight crack: sxy goes 0.96, 0.997, 1.0004, 1.0008 for d = 1e-1 ... 1e-4 L, then every basis gives 15.7 at
  1e-4 L and NaN or about +/-1e6 closer in.
- The `null_space` reduction is therefore not worse than hats. The error is in the per-element kernel columns, which
  every basis shares.

### 8.2 Root cause: float64 cancellation inside the closed-form quadratic slip kernel
- One element's kernel (`quadratic_kernel_farfield` via `displacements_stresses_quadratic_no_rotation`) compared
  against the *same formula* evaluated with 50-digit mpmath. Relative error vs distance from an endpoint, element
  length 1: 1e-13 at 1e-1, 1e-11 at 1e-2, 1e-7 at 1e-3, 2e-4 at 1e-4, O(1) at <= 1e-5.
- The formula looks correct (high-precision values show the expected log and 1/d behaviour). The problem is rounding:
  polynomial x log/arctan terms far larger than their sum nearly cancel. This holds at *every* element endpoint, not
  only fault tips.
- Separately, the result is NaN directly above an endpoint for d <= 1e-5. Probably `np.sign(y / (a - x))` when
  `a - x` rounds to 0; not confirmed.
- Safe: collocation at element centres (L/2 from the endpoints), so homogeneous solves are unaffected.
  At risk: anything within ~1e-3 L of a node, e.g. force collocation points next to a fault that ends on a material
  interface, `dr`-shifted points at nodes, and fine plotting grids.
- Not done: an independent check of the formula itself against quadrature of the point-dislocation solution.

### 8.3 One-ring hats at triple junctions (rule in section 7.4)
The first tests below used free-slope tip hats, since replaced by clamped tips. Re-checked with clamped tips (tip
elements get no hat or BC row; open nodes get slip and gradient rows) on a straight fault and all 7 junction
geometries: the number of hats equals the null-space dimension (20 and 18), rank[Z, H] is full, |A H| <= 2e-15,
cond(C H) = 1.87-1.94, and slip matches the general form to 1.5e-12 or better.
- Every patch system is square, condition number 33-48, including at the junction.
- Junction mesh: rank[Z, hats] = 2N, so the hats span exactly the constrained space.
- cond(C · hats) ~ 2, and the end-to-end slip matches the full 6N solve.
- Interior hats match `compute_coefs_quadratichatslip_planestrain` to 1e-16.
- Junction angles tested: flat + 30-degree splay, symmetric 120, T (90), Y, random (200, 17, 101), and a splay 5 degrees
  off either main branch. All square and full rank, with cond(C · hats) 1.94-1.98, and slip matching the full 6N
  solve to 2e-12 or better. The worst patch cond is 141 (5-degree cases).
- Shape: the hat of a splay element next to J peaks at J (1.14) and puts equal and opposite slip on the two other
  branches (slip conservation). The hat of a main-fault element passes straight through J.
- System cond, traction BCs at centres (constant / hat / full 6N row-scaled). 2N: 11 / 8 / 90 on a straight fault,
  11 / 7.5 / 80 at 120 degrees, 920 / 1.5e4 / 1.1e5 for a 5-degree splay. Shear only (N): hats 6-8 everywhere vs
  constant 8-11; for the 5-degree splay, 39 vs 130. The 5-degree 2N case is a real near-null mode: equal and opposite
  opening on the two nearly parallel branches, i.e. the thin wedge between them sliding as a block. Forbidding
  opening removes it.

Where these findings now live: issue 16 (kernel precision); TA task "Fix near-endpoint precision"; section 7.2 hat row
and section 7.4 (one-ring hats replace "not available" for triple junctions, and the earlier T4 idea of mixing both
forms is dropped); T2 task "Replace the 3-element hat builder".

### 8.4 Fix options for near-node accuracy (fix the kernel, not the basis)
1. **Curvature form for C1 bases** (preferred). Integrate by parts twice. For a basis with zero slip and slope at its
   support ends, the 1/r and log end terms cancel analytically. The stress becomes a sum over elements of
   piecewise-constant curvature times a bounded log-type kernel, with no cancellation. Needs a new derivation.
2. **Near-field fallback:** adaptive Gauss quadrature or an endpoint series when d < ~1e-2 L.
3. **Keep evaluation points >= ~1e-2 L from nodes.** Fine for collocation, not for plotting or forces near fault ends.

Also seen: on the kinked fault, stress keeps falling slowly toward the kink (0.68, 0.48, 0.22 for d = 1e-1 ... 1e-3).
This is probably a real log singularity at kinks that C1 continuity in the global frame does not remove; unverified.
