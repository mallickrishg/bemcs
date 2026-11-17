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

Originally implemented in `bemAssembly.solveAntiplaneBEM(fileinput, connectivityfile)`: mesh CSV columns
`x1,z1,x2,z2,BC_type,value` (for `'h'` rows `value` holds `alpha`), connectivity CSV lists 3-element force patches.
Now `bemcs.solve` with the exact interface condition and `mu_minus` / `mu_plus` columns. Demo, checked against the
exact layered half-space solution: `examples/antiplane/HeterogeneousElasticity_forces.py`.

### 2.2 Plane strain (original design; superseded, see section 3)

> The code does not use the `alpha` / projected-strain rows below. With uniform `nu`, every stiffness scales with
> `mu`, so the true stress is `(mu(x) / mu0) S` with `S` the reference-medium stress. The `'h'` row imposes exact
> continuity of the true traction (all components, volumetric part included) through `kappa`, and `'t'` rows
> prescribe the true traction. The table is kept for history.

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

Updated 2026-11-16. Details are in T1/T2 and the issue list (section 4).

- **Solver in the package.** `bemcs.solve(mesh, geometry=, mu=, nu=)` (and `bemcs.assemble`) solves both geometries
  in either form:
  - general: quadratic slip plus continuity rows;
  - reduced: one-ring hats.

  Force is always trapezoids, by default centred on every other element along each force chain. The old
  `examples/utilfunctions.py::solve_planestrain_heterogeneous` and `solveAntiplaneBEM` are gone.
- **Formulation as implemented** (it differs from section 2.2 in three places):
  - `'h'` rows use the exact sharp-interface factor `kappa` from (`mu_minus`, `mu_plus`) (`bemcs.interface_kappa`),
    not `alpha = mu_,n/mu`;
  - `'t'` rows prescribe the true traction (mu_side / mu0) S.n, not a projected strain;
  - the compliance uses `nu` (issue 1).
- **Input.**
  - Every element carries `mu_minus` / `mu_plus`.
  - BC types are `s`, `u`, `t`, `t_local`, `h`, per element or per component.
  - `Mesh.from_csv` reads `x1, z1, x2, z2, mu_minus, mu_plus` plus BC columns.
  - Fault tips are clamped and carry no BC row.
- **Example.** `examples/planestrain/testing_HeterogeneousMaterialsPlanestrain.py` runs on `Mesh` / `solve`.
- **Validated** (`tests/test_solve.py`, `tests/test_validation.py`):
  - no contrast gives no interface force;
  - high-contrast limits match a rigid boundary and a free surface;
  - plane-strain true traction is continuous across the interface;
  - the solution does not depend on the sampled side;
  - reversing the interface elements, or mirroring the problem, gives identical or mirrored fields;
  - the general and one-ring forms agree;
  - in antiplane only, the bimaterial image solution (h^2 convergence).
- **Not yet done:** see "P. Readiness for plane-strain heterogeneous elasticity" in section 5.
  - ~~a meaningful fault-crossing-an-interface case~~ done in antiplane: the layered half-space example (T2), with
    both faults at about 2.5e-5;
  - ~~the collocation-proximity warning (T3)~~ done, and `TrapezoidCrossingWarning` too.
- **Accuracy:** the interface condition is exact at collocation points and first order in h between them.

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
2. ~~**Observation-point side for `'h'` rows**~~ **SOLVED and tested (2026-11-16, exact sharp-interface condition).**
   - **Derivation.** With uniform nu, sigma = mu C with C = 2 eps + 2nu/(1-2nu) tr(eps) I. The true traction
     t = mu C.n is continuous across the interface, so the equivalent line force in the reference medium mu0 is
     F = mu0 t (1/mu_minus - 1/mu_plus).
   - **Sampling.** Sampling the reference traction (S_ref.n) on side s gives F = kappa (S_ref.n)_s, with
     kappa = mu_s (1/mu_minus - 1/mu_plus): `bemcs.interface_kappa`. That is 1 - mu-/mu+ on the -n side and
     mu+/mu- - 1 on the +n side; both sides give the same solution (`test_solution_does_not_depend_on_the_sampled_side`).
   - **Kernel sign conventions.** Plane-strain Kelvin acts along f, so the row is f_local - kappa R^T T = 0. The
     antiplane kernel has u = +f ln r/(2 pi mu), so the row is f + kappa T = 0.
   - **The force density is taken at the collocation centre from the trapezoid basis, not as the identity.**
     Overlapping trapezoids add 0.5 a_{j+-1} there.
   - **Inputs.** Every element now carries `mu_minus` / `mu_plus` (Mesh, CSV columns), and `solve` requires them.
   - **Validation:**
     - antiplane fault near a bimaterial interface vs the exact image solution, h^2 convergence (1.8e-3, 3.5e-4,
       9.0e-5), identical from both sides;
     - plane-strain true-traction continuity at collocation points, 1e-6 (the reference traction jumps by 150%),
       converging between them.
   - **The previous rows were wrong:**
     - antiplane `beta = -(mu_{i+1} - mu_i)/mu_{i+1}` in the example had the wrong sign and size for its +y normals
       (+0.667 where the exact value is -0.4);
     - plane strain lacked the factor 2 mu0 and had the wrong sign;
     - both used the identity for the force density.

   Original: (issue 2) Plane strain uses `x_c + dr*n` with `dr = -1e-9`
   for slip and force elements alike. Antiplane uses `x_c - dr*n` for force elements. Stress from a
   line force is discontinuous across the interface, so the side matters. Decide
   what `psi` should be sampled on (probably each side's value, or an average, related to how `alpha` is defined) and
   document it.
3. ~~**Definition of `alpha` at a sharp interface**~~ **SOLVED with issue 2** (exact kappa from (mu_minus, mu_plus)). Original: `alpha = mu_,n/mu` is a delta function at a jump.
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
8. ~~**Minor**~~ **SOLVED in code:**
   - `pkg_resources` was replaced by `importlib.metadata`;
   - the per-call `print(np.linalg.cond(A))` went with the old `solve_planestrain_heterogeneous` (`solve` has
     `return_system=True`).

   Left for the LaTeX notes (not in the repo): the `1-2nu` typo in the last sentence, and the plane-stress compliance
   of issue 1. Original: a `1-2nu` typo in the notes' last sentence, `__init__.py` uses deprecated `pkg_resources`,
   and `solve_planestrain_heterogeneous` has `print(np.linalg.cond(A))` on every call.
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
    BCs of both or one end element. Decided (2026-11-16): no extra elements. The non-planar fault is a toy case
    showing continuity of slip and traction, and the heterogeneous antiplane example is not built properly yet
    (open task in T1/T3).
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
- [ ] Add reference tests for the remaining kernels. **Only the triangle-force kernels remain** (lowest priority,
  tracked in T4); everything else below is done.
  - **Done:**
    - ~~constant slip (plane strain)~~ against Crouch & Starfield;
    - ~~linear and constant force (both geometries)~~ against point-force quadrature;
    - ~~hat-slip and trapezoid-force patches~~: smoothness, `A T = 0`, near field.
  - **Remaining:**
    - ~~the quadratic slip formula itself against quadrature of the point-dislocation solution, in both
      geometries~~ **done (2026-11-16, `tests/test_slip_kernels_reference.py`):**
      - **Reference:** textbook edge and screw dislocations (Hirth & Lothe), first checked against elasticity
        (Hooke from the displacement gradient, equilibrium, jump b across the cut). Quadratic slip = end
        dislocations ±s(±a) + density −s′, by composite Gauss–Legendre; no package code involved.
      - **Agreement:** every node and direction, displacement and stress, 4 orientations, element level and
        `get_slip_kernels` on a curved mesh: ~1e-14 in the near field (down to 0.05 a off the element and
        1e-3 a off the line beyond the tips). The far field loses digits as ~eps (r/a)^3 (1e-9 at 47 a,
        7e-8 at 180 a) from cancellation; the tolerance follows that. A change of 1e-6 in ν fails at every point.
      - **Sign convention pinned:** slip = u(+n side) − u(−n side). The Lagrange-node test checked only |jump|.
    - the triangle-force kernels, antiplane and plane strain: **open, lowest priority**. Tracked in T4
      ("Triangle force sources").
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

### P. Readiness for plane-strain heterogeneous elasticity (reviewed 2026-11-16)

**In place:** solver, exact interface condition (kappa, true traction), mesh/BC/moduli input, connectivity files,
warnings (`CollocationProximityWarning`, `TrapezoidCrossingWarning`), numba kernels, LU solve. The slip and force
kernels are verified against independent references in both geometries. The plane-strain heterogeneous solve is
checked by:
- the high-contrast limits (rigid and free, first order);
- traction continuity at the collocation points;
- side independence, element reversal and the mirror test;
- general = hat forms.

**Missing before plane-strain results can be trusted at a finite contrast:**
- [x] ~~**P1. A finite-contrast exact solution in plane strain.**~~ **Done 2026-11-16:** (a) circular inclusion and
  (b) Dundurs, both below.
  - **(a) Circular inclusion: done** (`tests/test_circular_inclusion.py`, `tests/reference_solutions.py`).
    - **Reference:** Muskhelishvili potentials, with the constants solved from continuity at boundary points
      (residual 1e-15). It matches the closed-form inclusion stress, Kirsch for a hole, Hooke and equilibrium.
    - **BEM results at 128 elements** (mu_i / mu_m = 4 and 1/4, nu = 0.27):
      - stress inside: 7.6e-3 (stiff) and 7.6e-4 (soft);
      - stress outside (>= 1.5 a): about 9e-4;
      - displacement: about 6e-4.
    - **Convergence:** at the centre (stiff) 1.6e-2, 5.5e-3, 2.1e-3 for 32, 64, 128 elements.
    - **Needed a new feature:** a remote load had no way into the `'h'` rows. `uniform_background` plus
      `solve(..., background=)` and `evaluate_fields(..., background=)` now handle it: the total field is
      background + BEM. Tested against prescribing t = -sigma_inf . n.
    - **Exposed a meshing trap:** a circle built with `linspace(0, 2 pi)` is not exactly closed
      (sin(2 pi) = 2.4e-16). It is treated as an open chain with tapered tips, giving errors of 100% at
      32 elements. `Mesh` now warns about end points that nearly coincide (`MeshGapWarning`, `node_gaps`).
    - **Finding: first order on curved interfaces.** The default trapezoids converge at about first order on the
      closed circle, in both geometries (the antiplane inclusion halves its error per refinement); the planar
      antiplane bimaterial gave h^2. See the new item below. This is the plane-strain counterpart of the
  antiplane image series; today nothing checks the answer at, say, mu2/mu1 = 4. Two candidates, ideally both:
  - **(a) Circular inhomogeneity under a uniform remote stress.** Exact (Muskhelishvili; the stress inside is
    uniform). It also exercises a **closed, curved interface**, the normal and volumetric parts of the condition,
    and the remote load. Gives h-convergence.
  - **(b) Done 2026-11-16** (`tests/test_dundurs.py`, `tests/reference_solutions.py`).
    - **Reference:** an edge dislocation near a bonded planar bimaterial interface (Dundurs 1969), with the image
      and transmitted potentials solved from continuity (residual 1e-15). It has Burgers vector b, satisfies Hooke
      and equilibrium, has continuous traction, and gives the free-surface and rigid limits. A buried fault with
      tapered shear and opening slip is the dislocation density -s' e, by quadrature.
    - **BEM setup:** an "s" fault in material 1 (horizontal, or dipping 40 degrees), an "h" interface out to
      +-300, no background.
    - **No contrast:** 2.1e-4, 9.3e-6, 5.3e-7 for h = 0.1, 0.05, 0.025. This validates the reference, the signs
      and the quadrature.
    - **mu2/mu1 = 4 and 1/4:** 2.4e-2, 1.0e-2, 5.4e-3 and 1.8e-2, 6.6e-3, 2.7e-3, about first order everywhere
      (not only near the interface). Truncation (L = 30 to 3000) makes no difference.
    - **Diagnosed:** the interface rows are only O(h) consistent in plane strain even on a straight line (residual
      at the exact force 1.1e-2, 4.9e-3, 2.3e-3). In plane strain a tangential force produces a normal traction on
      the line through a 1/x kernel, and vice versa. The trapezoids hold the force constant over each collocated
      element, and the odd error f'(x - c) there gives O(h). The same density linearly interpolated gives O(h^2)
      (traction error 1.25e-2 -> 5.8e-3 vs 3.8e-3 -> 9.6e-4). Antiplane has no such coupling along the line,
      hence h^2 there.
- [x] **P1d. Plane-strain force basis: option 1 chosen (user, 2026-11-16) and implemented: `basis={"force": "nodal"}`.**
  - **Basis:** continuous piecewise-linear force, one unknown (a global force vector) per node, collocated at the node
    with normal = the average of the two element normals (`nodal_force_layout`, `nodal_force_coefficients`,
    `Mesh.nodal_layout`). Trapezoids remain available for comparison.
  - **Break points:** at `Mesh.trapezoid_restart_points()` (fault crossings, open force ends, overlap ends of non-"s"
    slip) the force may differ on the two sides: a half-hat on each side, collocated at its element's centre. This
    is automatic, since there is no connectivity for this basis. Tapered tips have no unknown.
  - **Accuracy:**
    - **Dundurs (straight interface):** second order. 1.0e-2, 2.6e-3, 6.5e-4 (mu2 = 4, horizontal fault), against
      the trapezoids' 2.4e-2, 1.0e-2, 5.4e-3.
    - **Antiplane layered half-space, fault through the interface:** 1.1e-4, 3.3e-5, 1.7e-5, without any
      connectivity file.
    - **Rybicki contact:** as the trapezoids (corner-limited, P3).
  - **Kinks (curved interfaces as polygons):** the straight-element force layer is log-singular at a kink node, so
    collocating there (offset `dr` = 1e-9 element lengths) fails: 3.9, 1.2, 0.35, 0.15 on the circular inclusion.
    `solve(kink_offset=0.5)` samples kink nodes 0.5 element lengths off the interface on the sampled side:
    4.1e-3, 2.4e-3, 1.3e-3, 6.6e-4 for n = 32 to 256. That is first order (the polygon), but better than the
    trapezoids (1.6e-2, 5.5e-3, 2.1e-3). Straight nodes keep `dr`.
  - **Overlapping slip and force meshes (user):** the force mesh may be much denser than the slip mesh where they
    overlap. Recommended: an integer number M >= 1 of force elements per slip element, i.e. every slip node in the
    overlap is also a force node. Not enforced: `Mesh` warns (`OverlapRefinementWarning`,
    `overlap_refinement_issues`).
    - **Antiplane interface crack:** M = 2 and 4 make the contrast free (5.84e-2 at h = 0.05, the same crack with
      no contrast 5.8e-2); M = 1: 8.1e-2; M = 1.5 warns, 6.4e-2.
    - **Plane-strain interface crack (M = 2):** cond <= 7e2, traction continuous to 1e-5 at the bonded nodes,
      converging.
    - The proximity check now skips collinear overlaps, and works on collocation points (nodes included).
  - **`assemble`'s `rows`** gains "x" and "y" (the collocation point of each row).
  - **Tests:** `tests/test_nodal_force.py`.
  - **Decision (user, 2026-11-16): trapezoids stay the default force basis** for their simplicity; "nodal" (tents)
    is an option for plane-strain cases that need second order on straight interfaces.
    - **Tent apex:** on a straight interface the stress of a tent is finite at its apex (it tends to f/2); at a
      geometric bend it is log-singular (measured: 0.50 at all distances when straight, but 0.49 -> 1.05 from
      d = 1e-1 to 1e-9 when bent 20 degrees), hence `kink_offset`.
    - **Side independence:** the tent solution does not depend on the sampled side (1e-9).
- [x] **Open force ends only at the free surface (user, 2026-11-16).** `force_open_ends(els, labels, bc_type)`: a force
  chain tip is open (free force value) only if a slip element with a traction condition ("t", "t_local") touches it.
  The cases are told apart:

  | the chain meets slip elements | treatment |
  |---|---|
  | its end lies on the free surface ("t"), incl. a node shared by the surface and a fault | open end |
  | its end lies on a fault only ("s"), e.g. a layer offset across a fault | ordinary tapered tip |
  | a fault cuts through it (an interior node of the chain lies on the fault) | crossing: `fault_crossings`, `TrapezoidCrossingWarning`, `split_force_trapezoids()` |

  Without boundary conditions (a kernel-only mesh) there are no open ends.
  Test: `test_open_ends_only_at_the_free_surface_not_at_faults`.
  - **Original note:** (b) Edge dislocation, or a short fault with smooth slip, near a planar bimaterial interface. Dundurs image
    dislocations. This is the plane-strain analogue of the antiplane layered case, with fault-interface coupling
    through both shear and normal components.
- [x] ~~**P1c. First-order convergence on curved interfaces.**~~ **Settled 2026-11-16.** Diagnosed (polygon kinks),
  and the user decided to keep the trapezoids and accept first order on curved interfaces: refine the force mesh
  for accuracy. Realistic interfaces are non-planar, where the tents lose their advantage too.
  - **Basis is not the bottleneck.** On the circular inclusion (antiplane), representing the *exact* force density
    in the trapezoid space, without solving, gives interior errors 1.2e-2, 3.1e-3, 7.7e-4, 1.9e-4 for n = 32 to
    256: second order. That equals plain linear interpolation on half as many elements, as expected from half the
    unknowns. The solved trapezoids give 5.9e-2 to 6.4e-3: first order, and 5-30x worse.
  - **The equations are first-order consistent on a polygon.** Plugging the exact force density into the discrete
    `'h'` rows leaves a residual of 7.1e-2, 3.6e-2, 1.8e-2, 9.2e-3: O(h). It is the same with element-linear
    interpolation of the exact density, so the trapezoid shape is not the cause.
  - **On a straight line the same equations are second-order consistent:** 6.1e-2, 1.6e-2, 3.9e-3, 9.8e-4.
  - **Cause:** the polygonal (C0) interface has a kink of 2 pi / n at every node. The traction of the neighbouring
    elements, evaluated at the collocation points, picks up an error proportional to the kink angle, which is O(h).
    This is the same mechanism as the slip-kink limitation in the README (cancellation only for co-planar
    elements), here for force elements.
  - **Remedies to consider:**
    - curved (higher-order geometry) elements;
    - an analytic kink or curvature correction to the traction at collocation points;
    - accept first order on curved interfaces and document it. Planar interfaces keep h^2.
- **Note on background fields (user, 2026-11-16):** remote loading (`background=`) is an edge case, used here
  because the inclusion is the cleanest exact solution. The typical heterogeneous problem is an imposed slip source
  with prescribed tractions elsewhere (a free surface), so the remaining checks (P1b, P2, P3, P6) should use that
  setup, without a background.
- [x] ~~**P2. A fault crossing an interface in plane strain.**~~ **Done 2026-11-17** (`tests/test_fault_crossing.py`).
  - **Exact reference after all (full space):** a dislocation above the interface is the mirror image of one below
    with the materials swapped (the reflection also reverses the Burgers circuit: b -> (-bx, by)), so the Dundurs
    solution integrates along a fault that runs through the interface. The image-constant fit was ill-conditioned
    within 1e-3 of the interface (residual 1e-2 at 1e-4); column scaling in
    `edge_dislocation_bimaterial_constants` brings it to round-off.
  - **Setup:** fault through (0.15, 0), s from -0.7 to 0.5 along dip, cos^2 slip (shear 1, opening 0.5), dips
    90/60/30; interface out to 300 (log-spaced). Stress error at points >= 0.15 from the fault and the interface,
    relative to the largest; h = 0.1, 0.05, 0.025.

    | layout | no contrast | mu2/mu1 = 4 or 1/4 |
    |---|---|---|
    | any | 2.7e-4, 1.3e-5, 6.7e-7 (all dips) | |
    | trapezoids tapered to zero at the crossing (old) | | first order, 3e-2 to 1e-1 at h = 0.025, worse at gentle dips |
    | trapezoids + free half-hat each side (new default split) | | 3e-3 to 5e-3 at h = 0.025 (ratios 2.3-3.6) |
    | nodal | | 1e-3 to 2e-3 |
  - **Zero contrast, half-space (free surface + layer):** identical to the mesh without the interface (1e-15).
  - **Half-space with contrast (self-convergence of surface displacement vs nodal at h = 0.00625, dip 60):**
    the surface sees a net-force error as a near-rigid offset (log L), invisible in stresses.
    - trapezoids, uniform: 7.3e-2 (stiff layer, mu 4) and 1.1e-2 (soft layer) at h = 0.025. Net interface force
      decays at first order (full space: 2.3e-2 -> 2.9e-3 for h = 0.1 -> 0.0125; nodal -> 4e-4);
    - the error is spread over |x - xc| < 0.5, where the force density changes by ~5 over ~0.2 (it jumps at the
      crossing and has a cusp-like approach, ~r^0.7): the known first-order plane-strain trapezoid error, scaled by a
      steep gradient. Grading only at the crossing node, or refining the fault, does not help;
    - **remedy (meshing, the user's choice):** refine the trapezoids over a zone around the crossing. M = 4 within
      |x - xc| < 1: 1.2e-2 / 2.6e-3 (dip 60), 2.0e-2 / 4.6e-3 (dip 90), 9.1e-3 / 1.6e-3 (dip 30) at h = 0.025 for the
      stiff / soft layer, on par with or better than nodal (1.1e-2 / 7.7e-3 at dip 60).
  - **Antiplane cost:** in the antisymmetric layered case the exact force at the crossing is zero, so the half-hats
    cost a little: 6.1e-5 vs 2.6e-5 at h = 0.05, 2.3e-5 vs 1.8e-5 at 0.025.
  - **Warning (user, 2026-11-17):** `solve` issues `CrossingRefinementWarning` at a true crossing (not an overlap
    end) of an interface with a contrast when the force elements there are longer than half the fault elements,
    recommending ~4x finer force elements within about one fault length. It does not change the mesh.
  - **Open:** document zone refinement near crossings in the user docs; a stiff layer over a soft half-space is the
    hardest case (1-2e-2 at h = 0.025 even refined; a finer zone or M = 8 would be the next thing to try).
- [x] ~~**P3. An interface that meets the free surface**~~ **Done 2026-11-16**: the force chain has an open (free)
  end at the free surface. The remaining first-order corner error is handled by refining towards the corner
  (documented in the Rybicki example). Below, the original item (outcropping or dipping layers, basins). **Confirmed
  2026-11-16** with the vertical-contact case (`examples/antiplane/BimaterialContact_Rybicki.py`,
  `tests/test_heterogeneous.py`):
  - **Exact reference:** a strike-slip fault at x = 0 in a half-space, contact at x = d, mu2/mu1 = 4.
    u = S + k S(2d - x) for x < d and (1 + k) S for x > d (Rybicki 1971).
  - **Result:** first order, 3.3e-2 -> 1.7e-2 (d = 0.5) at h = 0.1 -> 0.05. The force density is exact at the
    collocation points at depth, but the top contact element, touching the free surface, carries about half the
    exact force, because of the slip-force contact taper. The missing net force (O(h)) shifts the whole surface
    displacement (a smooth, broad error) and grows slowly with the extent L, like log L.
  - **Remedy done (user's decision, 2026-11-16): open force ends.**
    - **Slip side unchanged:** the slip elements are treated as regular elements. Slip topology is that of the
      slip elements alone, so a free surface meeting a fault is an ordinary slip triple junction, with no taper.
      The earlier "both taper" wording was wrong for slip.
    - **Force side:** a force chain that ends on a slip element has an open end (`force_open_ends(els, labels)`,
      `mesh.force_open_ends`).
      - **Open-end patch (user's choice: a free unknown):** the connectivity row [e, e, e] is a half-hat on the
        tip element e, 1 at the tip node and 0 at the next node, with its own collocation point at e's centre.
        The neighbouring trapezoid keeps its zero at the tip.
      - **Default layout:** `trapezoid_connectivity(..., breaks=...)` adds these rows at tip breaks and counts the
        stride from the tip. The `Mesh` default and `split_force_trapezoids()` include them.
    - **Alternatives compared** (Rybicki, surface error, h = 0.1, 0.05, 0.025):

      | end treatment | d = 0.5 | d = 0 |
      |---|---|---|
      | tapered | 3.3e-2, 1.7e-2, 8.9e-3 | 1.3e-1, 6.8e-2, 3.4e-2 |
      | constant over the tip element | 1.3e-2, 6.4e-3, 3.5e-3 | 6.0e-2, 2.5e-2, 1.1e-2 |
      | free node (chosen) | 1.1e-2, 6.1e-3, 3.4e-3 | 4.7e-2, 2.3e-2, 1.1e-2 |
    - **Still first order: the corner** (confirmed). Grading the free surface and the contact towards the corner
      only (geometric halving):
      - d = 0.5, h = 0.1: 1.3e-2 -> 2.7e-3 (3 levels) -> 9.8e-4 (6 levels);
      - d = 0, h = 0.05: 2.5e-2 -> 2.6e-3 (3 levels).
      - **Cause:** a force layer with non-zero density at its end has a log-singular stress there, which the free
        surface must cancel; uniform hats do it only to O(h). Corner grading is a meshing choice for the user
        (document it); a corner correction would be the code-side alternative.
  - **Original note:**
  - **Gap:** where a force chain meets a slip (free-surface) element, the force tapers to zero ("slip-force contact:
    both taper"). The true force density there is generally nonzero, so expect a local, first-order error. Untested
    in either geometry.
  - **Plan:** test against a high-contrast limit or by convergence, then decide whether to warn
    (warn-don't-fix rule).
- [x] **P4. Rebuilt (2026-11-16) on the Dundurs setup:**
  - **Setup:** a buried 40-degree fault with tapered shear and opening slip in material 1, below a bonded interface
    with material 2 (mu2/mu1 = 4).
  - **Script:** self-contained, with its own exact dislocation-integral solution (continuity residual 5e-15).
  - **Figures:** geometry and prescribed slip; true stresses, BEM (trapezoids) vs exact; profiles above and below
    the interface with trapezoids and tents; convergence.
  - **Results** (h = 0.1, 0.05, 0.025):
    - trapezoids 9.9e-3, 2.8e-3, 1.5e-3;
    - tents 4.1e-3, 1.1e-3, 2.7e-4;
    - no contrast 1.8e-4, 7.9e-6, 4.1e-7.
  - Runs in about 11 s.
  - **Original note:** rebuild `examples/planestrain/testing_HeterogeneousMaterialsPlanestrain.py`. It is still a toy
  (3 slip and 3 force elements, a 3-element interface), like the old antiplane example. Rebuild it on the P1 case(s)
  with log-spaced lines and connectivity files, the same way as the antiplane example.
- [x] ~~**P6. Fault on a material interface: slip and force meshes that overlap exactly**~~ **Done 2026-11-16**
  (antiplane and plane strain). Classification, warnings, denser force meshes with integer M, and tests
  (`tests/test_overlapping_meshes.py`, `tests/test_nodal_force.py`). Optional later: the exact plane-strain
  interface-crack solution (England 1965; Rice 1988).
  - **Why:** the natural setup for a bimaterial fault (the fault is the contact between two materials). Slip
    elements and force (`'h'`) elements then share every node and every collocation point. Physically important:
    in plane strain, slip on a bimaterial interface changes the normal stress (bimaterial coupling).
  - **What the code does today** (probed 2026-11-16, straight fault on a straight interface):
    - `Mesh` / `solve` accept it in both geometries and both slip forms (hat, quadratic), and the system is well
      conditioned (cond 4 to 1.5e2).
    - **`bemcs.label_nodes(els)` raises** ("overlapping nodes ... same circulation direction"): 2 slip + 2 force
      elements at every node. `solve` works on per-type subsets, so it is not affected, but anything calling
      `label_nodes` on the whole mesh is.
    - **`TrapezoidCrossingWarning` fires at every interface node.** `fault_crossings` treats every node as a
      crossing, which is a false positive for a coincident fault. Coincident (and partly coincident) slip and force
      elements need their own detection: a different warning, or none.
    - **Slip-force contact rule:** "a node shared by a slip and an interface element is open (tapered) for both".
      Check what this does at the ends of the fault, where the interface continues.
    - **Collocation:** `'h'` rows sample the traction at the same point where the fault's own kernels are
      evaluated (offset `dr`, `side`). A displacement discontinuity has continuous traction, so this should be
      fine, but in plane strain the in-plane stress parallel to the fault jumps across it. Check that the `side`
      choice is consistent with the interface `kappa`.
  - **First antiplane check:**
    - **Exact reference:** slip s on an infinite straight bimaterial interface gives the homogeneous slip field
      scaled by c+ = 2 mu- / (mu- + mu+) on the +n side and c- = 2 mu+ / (mu- + mu+) on the -n side.
    - **Setup:** fault on [-2, 2], interface to +-200 (log-spaced), mu+/mu- = 3, hats + trapezoids.
    - **Result:** off-fault displacement errors 5.5e-3, 3.5e-3, 3.2e-3 for 16, 32, 64 fault elements (|u| up to
      0.57). It levels off, with the same error on both sides (a common offset?). Investigate: truncation of the
      interface, the force taper where the fault ends on the interface, or the coincident collocation.
    - **Pitfall:** an interface no longer than the fault is a different problem (a fault on a finite material
      strip), not the bimaterial one.
  - **Vertical bimaterial fault (d = 0 in the Rybicki case):** the exact solution scales the half-space field by
    2 mu2/(mu1 + mu2) on the mu1 side and 2 mu1/(mu1 + mu2) on the mu2 side. Today: 1.3e-1 -> 6.8e-2 (first
    order), dominated by the P3 taper at the free surface, plus the false-positive `TrapezoidCrossingWarning`.
    Revisit after P3.
  - **Plane strain:** exact references for a dislocation, or a constant-slip segment, on a bimaterial interface
    exist (Dundurs 1969; Comninou). Unlike cracks there are no oscillatory singularities for a prescribed smooth
    slip. Compare the bimaterial normal-stress change on the fault.
  - **Overlap handling, done (2026-11-16)** (`tests/test_overlapping_meshes.py`):
    - **Classification** of force nodes on slip elements:
      - inside an overlap (all neighbouring force elements on slip elements): fine, no warning;
      - end of an overlap, or a true crossing: `mesh.fault_crossings`;
      - chain tip on slip: `mesh.force_open_ends`.
      `coincident_force_elements(els, labels)` lists the overlapping force elements.
    - **Restart points** (`mesh.trapezoid_restart_points()`, used by `TrapezoidCrossingWarning` and
      `split_force_trapezoids`): crossings, open ends, and overlap ends, except those formed by prescribed-slip
      ("s") elements. Those taper smoothly at their tips, so the force has no step there; splitting at the bottom
      of the Rybicki d = 0 fault made it slightly worse (3.2e-2 vs 2.5e-2). Traction ("t") cracks have singular
      tips, so their overlap ends do restart.
    - **Antiplane interface crack** ("t" on "h"; exact slip tau (1/mu1 + 1/mu2) sqrt(a^2 - x^2)):
      - converges; with the stride restarted at the crack tips the contrast adds almost nothing to the crack's own
        error: 4x contrast 1.4e-1, 6.4e-2, 3.0e-2 vs no contrast 1.2e-1, 5.8e-2, 2.9e-2;
      - without the restart: 8.2e-2 at h = 0.05, and it warns;
      - the crack's own first-order error is the clamped-tip hats vs the sqrt crack tip, a separate limitation of
        the hat basis for traction-loaded cracks.
    - **Plane-strain interface crack:**
      - no contrast: the interface carries exactly zero force, and the slip is Griffith's (5.8e-2 at h = 0.05,
        same as the crack alone);
      - 4x contrast: converges (changes 7.3e-2, 3.2e-2, 1.6e-2), cond <= 8e2, and the true traction is continuous
        to about 1e-5 at the bonded collocation points;
      - the exact oscillatory solution (England 1965; Rice 1988) is a later check, with the Dundurs work.
    - `bemcs.label_nodes(els)` on the whole overlapping mesh raises (it cannot label 2 slip + 2 force elements at
      one node). The solver never calls it on the whole mesh.
  - **Options if the overlap formulation stays awkward:** a combined "bimaterial fault" element (slip with
    mu_minus != mu_plus that also carries the interface condition), instead of two coincident meshes.
- [ ] **P5. Documentation of the heterogeneous formulation as implemented** (T3). Can follow P1-P4.

**Not blockers:** triangle force sources (lowest priority), far-field kernel expansion (only for very long meshes),
spatially varying nu (T4), wiki updates.

### T1. Correctness of plane-strain heterogeneous solver
- [x] Fix issue 4.1 (compliance) and re-run `testing_HeterogeneousMaterialsPlanestrain.py` (runs; same solution, see issue 1).
- [x] Resolve 4.2 and 4.3: exact kappa from (mu_minus, mu_plus) and the sampled side, `bemcs.interface_kappa` (see issue 2).
- [x] Fix 4.5 so the test script runs end to end.
- [x] Validate (`tests/test_solve.py`, `tests/test_validation.py`, 2026-11-16):
  - [x] (a) no contrast gives no interface force.
  - [x] (b) limits, interpreted as the high-contrast limits in both geometries (say if (b) meant something else).
    mu+/mu- = 1e4 matches a rigid boundary (homogeneous BEM with u = 0 on the line), and 1e-4 matches a free surface
    (t = 0); compared by stresses, because displacements also reflect the different mesh truncations. Measured from
    h = 0.1 to 0.05 (first order, the rate of the default trapezoids):
    - plane strain: rigid 5.0e-3 -> 2.5e-3, free 7.3e-2 -> 3.1e-2;
    - antiplane: about 1e-3 (rigid) and 4e-3 (free).
  - [x] (c) analytic layered case: antiplane bimaterial image solution (h^2 convergence, both sides). Plane strain
    relies on (b) and (e); a Dundurs bimaterial dislocation solution would be a further check.
  - [x] (d) symmetry:
    - reversing the interface elements (n -> -n) while swapping mu- <-> mu+ gives identical fields (1e-10);
    - the bimaterial mirrored about the interface gives mirrored fields. For plane strain the mirrored fault takes
      the opposite local shear slip, because t' = -R t.
  - [x] (e) plane-strain traction continuity across the interface: exact at collocation points, first order between
    them.
- [x] `'t'` rows now prescribe the true traction (mu_side / mu0) S.n; prescribed projected strain is no longer a BC type.
- [x] Verify the trapezoidal-force kernels in isolation (issue 4.6; see the issue for the tests).

### T2. Package integration
- [x] Move `solve_planestrain_heterogeneous` into `bemcs/bemAssembly.py` (mirroring `solveAntiplaneBEM`). It uses the new
  kernel API, has exact `'s'` rows, and prints the condition number only when `verbose=True`.
- [x] Mesh/connectivity interface: `Mesh.from_csv` reads `x1, z1, x2, z2, mu_minus, mu_plus` and the BCs as
  `BC_type, value` (antiplane), `BC_type_1, value_1, BC_type_2, value_2` (plane strain, one type per component;
  x/y for global types, t/n for local ones, `h` on both), or the shorthand `BC_type, value[, value2]`
  (`length_scale=`). Patches default from the topology:
  - one-ring hats for slip;
  - trapezoids centred on every other element along each force chain, as in the original design.

  **Rule (enforced, `_check_trapezoid_overlap`):** a trapezoid is a linear basis spread over 3 elements, so
  neighbouring trapezoids share exactly one element ([0,1,2], [2,3,4], ...), whereas neighbouring quadratic hats
  share two. A connectivity in which two trapezoids share two or more elements raises `ValueError` (tested).
  Every-element trapezoids have an alternating near-null mode that centre collocation cannot see. It blew up to +-49
  where a fault crosses an interface; the default layout removes it, and the condition number drops from 1.6e5 to 2.7e3.
- [x] Unified solver `bemcs.solve(mesh, geometry=, mu=, nu=)` (and `bemcs.assemble`), 2026-11-16:
  - **Formulation from the mesh basis:** general (quadratic slip + continuity rows) or reduced (one-ring hats);
    trapezoids for force.
  - **BC types:** `s, u, t, t_local, h`, per element or per component.
  - **Exact rows** throughout; least squares; returns the coefficients in kernel order.
  - **Replaces** `solveAntiplaneBEM`, `construct_linearoperator_slip_antiplane`, `solve_planestrain_heterogeneous`
    and the solvers in `examples/planestrain/utilfunctions.py` (file deleted).
  - **Plotting:** `bemcs.evaluate_fields`, `bemcs.plot_fields` (geometry-aware components, `derived=`, `mask=`, true
    stresses with `mu_field=`) and `bemcs.plot_mesh` (normals, mu_minus / mu_plus dots on each side). These replace
    `plot_BEM_field` and `plot_displacements_stresses_els`.
  - **Examples:** all use these (the split box keeps its custom friction assembly). Homogeneous examples reproduce
    their previous solutions to 1e-9 (slip with u BCs: 6e-7, from the sampling offset).
  - **Traction boundaries need their own modulus (2026-11-16).** A free surface (or any `t` element) is a slip
    element whose row uses the true traction (mu_side / mu0) S.n. Its mu_minus / mu_plus must be the modulus of
    the material it bounds, matching the adjacent `h` side; the reference mu0 is arbitrary (1 is fine).
    - **Not enforced:** `plot_mesh` makes it visible on one colour scale, with dots on both sides of `h`
      elements and a square at the centre of `t` elements.
    - **The user checks** that every closed polygon of `t` / `h` elements encloses one uniform modulus.
    - `ConstructMesh.py` already gives the free surface the top-layer modulus.
  - **Accuracy note:** with the default trapezoids the interface condition is exact at collocation points and first
    order in h between them (force constant on every other element).
- [x] ~~Support multiple patches, and patches that share nodes (junctions between slip and interface elements)~~.
  Done:
  - multiple patches: one-ring hats on every non-tip slip element, and trapezoids along every force chain;
  - a node shared by a slip and an interface element is open (tapered) for both (decided; `continuity_operator`).

  Whether that treatment is physically adequate where a fault crosses a material contrast is part of the rebuild
  below.
- [x] ~~Rebuild `examples/antiplane/HeterogeneousElasticity_forces.py` (and its `ConstructMesh.py`)~~ done
  (2026-11-16):
  - **Problem:** a strike-slip fault breaking the free surface of a compliant layer (mu1, thickness H) over a stiff
    half-space (mu2 = 4 mu1), with smooth tapered slip. Two faults: one inside the layer (D = 0.6 H), one through
    the interface (D = 1.63 H). The fault has no node on the interface: it crosses inside one of its elements, at
    an interface node.
  - **Reference:** the exact image series for the surface displacement (Rybicki 1971; Segall 2010 ch. 5), for
    both depth branches, integrated over the slip profile.
  - **Files:** `ConstructMesh.py` holds the model (parameters, `mesh_table`, `layered_mesh`) and writes two CSVs.
    The example reads them, solves, and compares. Panels:
    - surface displacement vs the series and the homogeneous half-space;
    - error along the surface;
    - convergence with element length;
    - the true stress field;
    - traction continuity on the interface.
  - **Mesh:** the free surface and the interface extend to |x| = 1000 H. Elements are log-spaced (8% growth) and
    capped at 5 H, so the parallel lines H apart stay resolved and no proximity warnings fire. The force trapezoids
    are mirror-symmetric about the fault (`force_connectivity`: the stride restarts at the crossing node).
  - **Results:** both faults match to about 2.5e-5 at h = 0.05 H. Inside the layer the error falls with h (1.8e-4,
    2.4e-5, 7e-6); through the interface 8.6e-5, 2.5e-5, 1.7e-5. Truncation stops mattering beyond about 100 H
    (2.6e-4 at 10 H). The traction is continuous at the collocation points (2e-5), but not on the two elements
    next to the crossing.
  - **Finding: the default trapezoid layout at a crossing.** The stride-2 layout from one end puts a collocation
    point on one side of the crossing only. In this antisymmetric problem that leaves a spurious net force on the
    interface of about 0.02, whose log-r far field shifts the surface displacement by 1-2%, growing with L (1.0e-2
    at 10 H, 2.3e-2 at 1000 H). The first rebuild (L = 60 H) misread this as first-order convergence.
  - **Tests:** `tests/test_heterogeneous.py`:
    - the series branches agree;
    - both faults match with the symmetric layout;
    - the lopsided default layout produces the spurious symmetric part;
    - traction is continuous at the collocation points.
- [x] **Force trapezoids at fault crossings: warn, and suggest a fix** (2026-11-16).
  - **Principle (user's decision):** the user provides the mesh file and the connectivity file. bemcs does not
    silently repair a poorly designed mesh. It points out what the user can fix, as with
    `CollocationProximityWarning`.
  - **First version replaced:** `Mesh` first rewrote the default force connectivity at crossings; that was reverted
    the same day. `Mesh` now keeps the user's connectivity, or the plain stride-2 default.
  - **Detection:** `fault_crossings(els, labels)` finds force-chain nodes that lie on a slip element
    (`mesh.fault_crossings`).
  - **Warning:** `assemble` / `solve` issue `TrapezoidCrossingWarning` when a trapezoid spans a crossing node, i.e.
    contains both elements next to it. The message gives the location, the elements, the consequence (poorly
    represented force density, spurious net force) and the fix.
  - **Fix helpers:**
    - `mesh.split_force_trapezoids(points=None)` returns a global connectivity with the stride restarted at the
      crossings, ready to save as the connectivity file;
    - underneath, `trapezoid_connectivity(els, breaks=...)` counts the stride from each break, so collocation
      points sit 1.5 elements from it on both sides. A segment between two breaks with an even number of elements
      gets 1.5 at one end and 2.5 at the other.
  - **Connectivity files:** `Mesh` and `Mesh.from_csv` accept a CSV path per kind, e.g.
    `connectivity={"force": "x_connectivity.csv"}`, with three 0-based element indices per row and an optional
    header.
  - **Example:** `ConstructMesh.py` writes a mesh file and a connectivity file per case. The comparison without
    a connectivity file shows the warning.
  - **Warning location:** both mesh warnings point at the user's `solve(...)` / `assemble(...)` call.
  - **Tests:**
    - `tests/test_get_kernels.py`: breaks, closed loop, warning, suggested fix, connectivity file;
    - `tests/test_heterogeneous.py`: the lopsided case warns and is off by percent; the split case is exact.
- [x] ~~**Let the interface force density step where a fault crosses an interface.**~~ **Done 2026-11-17** (P2):
  `trapezoid_connectivity` adds a free half-hat [e, e, e] on each side of every interior break; the coefficients
  find its free node from the break points (`Mesh.transform` passes `trapezoid_restart_points()`).
  - **Cause:** across a fault the normal traction on an interface steps in proportion to the slip gradient, and
    the step differs on the two sides (mu1 vs mu2). So the equivalent force density steps at the crossing.
  - **Effect:** continuous trapezoids cannot represent the step. With the symmetric layout the force is zero at the
    crossing node (the average of the two sides) and both adjacent elements are interpolated, so the traction is
    O(1) off on those two elements. In the symmetric example the error stays local, but without the symmetry the
    local error carries a net force.
  - **Option:** split the chain at the crossing with a non-tapering end basis (a half-trapezoid) on each side.
- [x] **Displacement BCs in local and global frames** (2026-11-17, `tests/test_displacement_bc.py`).
  - `"u_local"` (local shear, normal displacement) added next to `"u"`; `"u_global"` and `"t_global"` are accepted
    aliases of `"u"` and `"t"` (stored under the plain names). Mixed components work, e.g. a roller
    ["t_local", "u_local"] = (0, 0).
  - Checked: global and local forms give the same solution (1e-10), with and without a background; a line
    prescribed with the fault's own displacement gets no slip; the roller holds between collocation points
    (normal displacement 4e-5, shear traction 1.2e-3 of the fault's field).
- [x] **Sample rows on the element unless the sampled quantity jumps** (user, 2026-11-17; `_needs_side`).
  - Force kernels: displacement continuous, traction jumps. Slip kernels: displacement jumps; traction continuous on
    the element's own plane, but the stress tensor jumps, so traction on another plane jumps too.
  - So the `dr` shift to `side` is applied only to: displacement rows on slip elements; traction/"h" rows on force
    elements (own or coincident) or on a slip element at an angle (a crack through another crack's collocation
    point). Free surfaces ("t" on slip) and displacement rows on force elements are sampled on the element.
  - Before, every row moved by dr = 1e-9 L; the solutions agree to ~1e-7 (the dr error itself).
- [ ] **Far-field quadrature for r/a > 100 (later stage).**
  - **Problem:** the closed-form element kernels are differences of endpoint terms (log r1 - log r2, polynomial x
    log terms). Far away these terms are large while their difference, the kernel, is small, so float64 cancellation
    costs accuracy. Measured against 60-digit evaluation of the same formulas (element half-length a, point at
    distance r), as relative error of the final kernels:

    | r/a | 1e2 | 1e3 | 1e4 | 1e5 |
    |---|---|---|---|---|
    | quadratic slip (U and S) | 4e-9 | 3e-6 | 4e-3 | O(1) |
    | linear force | 5e-13 | 3e-11 | 4e-9 | 5e-7 |

    This is about (r/a)^3 relative, (r/a)^2 absolute.
  - **Planned fix:** keep the numba / numpy element kernels exactly as they are. After building the BEM matrix (or a
    field evaluation), recompute only the entries whose source element is far from the observation point
    (r/a > 100) by a few-point Gauss quadrature of the point dislocation (slip) or point force (force), and swap them
    in. Far away the integrand is smooth, so 4-8 points reach machine precision with no cancellation. This leaves the
    jitted kernels untouched.
  - **Does it matter today? Barely.** Homogeneous half-space (fault plus free surface only, no parallel lines),
    uncapped log-spaced surface, error vs the exact solution (L = 1e3, 1e4, 1e5):
    - fault elements 0.05 (max r/a 4e4 to 4e6): 2.5e-5 at every L;
    - fault elements 0.0125 (max r/a 2e5 to 2e7): 3.7e-7, 3.8e-6, 5.2e-5.

    The cancellation is visible but small, even at r/a = 2e7.
  - **Correction (2026-11-17):** the earlier claim that the layered example "fails" at L = 1e4-1e5 (errors 8e-3 to 1)
    blamed far-field cancellation. It came mostly from the uncapped log-spaced elements: hundreds of H long, with the
    free surface and the interface only H apart (the proximity warning fired). Capping elements at ~5 H, as the
    examples do, avoids it. (A capped run out to 1e5 is too large for a dense solve, so the half-space test above
    was used to separate the two causes.)
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
- [x] **Triple junctions in connectivity files, and boundary conditions on connectivity rows** (2026-11-17).
  A 3-element chain cannot describe the hat of an element next to a triple junction (a fault breaking the free
  surface): every variant was wrong by up to 7e-2 in displacement, or singular.
  - **Rows:** columns `e1, e2, e3[, e4, ...]`, centre `e2`. `e4` is the third branch at a junction (-1 elsewhere);
    such a row is the one-ring hat of `e2` (`_ring_hat`, shared with `one_ring_hat_coefficients`). Force trapezoids
    stay 3 columns.
  - **BC columns** on a row (`BC_type_1, value_1, BC_type_2, value_2` or `BC_type, value`) apply at the centre of
    `e2`. The mesh file can then hold geometry and moduli only (`Mesh.from_csv` makes its BC columns optional, and
    reads an optional `label` column). Elements in rows of kind k are labelled k. Element BCs given as well must
    agree. Tips get no condition, so there is no silently ignored per-element value.
  - **Helpers:** `one_ring_connectivity`, `one_ring_support`, `Mesh.patch_connectivity`,
    `Mesh.connectivity_table(kind, geometry=...)` (writes a ready-to-save table with the mesh's BCs).
  - **Checks:** `HatConnectivityWarning` for a missing hat, a hat on a tip, or a hat that does not span the one-ring
    of its centre (e.g. a junction row without `e4`), with the suggested row. A repeated centre raises.
  - **Tests:** `tests/test_junction_connectivity.py`; explicit junction rows equal the general (quadratic) solve to
    1e-12 in both geometries. Examples: `planestrain/example_freesurface_fault.py`, and
    `antiplane/ConstructMesh.py` with `HeterogeneousElasticity_forces.py` (now slip hats, three files per mesh;
    same results as the per-element version to 1e-14).
- [x] **`mu` is optional (default 1); moduli are mesh data** (2026-11-17). Every element has `mu_minus`/`mu_plus`, and
  the reference modulus of the kernels cancels from the solution.
  - `solve`, `assemble`, `evaluate_fields`, `plot_fields`, and `get_kernels` (with a Mesh) take `mu=None` or 1. The
    reference is then 1, or the modulus a `uniform_background` records (`bg.mu`). An explicit `mu` that differs from
    `bg.mu` raises. `uniform_background` without `mu` needs `mu_medium`.
  - `evaluate_fields` / `plot_fields` give **true stresses by default**: `mu_field=None` uses
    `Mesh.modulus_at(x, y)`. `mu_field` can still be a callable or a number, or `"reference"` for the old unscaled
    output. A mesh without moduli gives reference stresses.
  - `Mesh.modulus_at`: the nearest element bounds the point's region (the segment to it crosses no other element),
    so its `mu_minus` / `mu_plus` on the point's side is the region's modulus. At a shared node, the element
    making the smallest angle with the point is used: it bounds the point's angular sector, with the point on the
    sector's side of its line. Only "h", "t" and "t_local" elements are used, because the moduli of "s" and "u"
    elements do not enter `solve`. A point on an element takes mu_minus.
  - Tests: `tests/test_reference_mu.py`: mu = 1 vs 3.7 in both geometries; layers; the corners of a square
    inclusion; untrusted fault moduli; background reference.
  - Not changed: exactly on a fault node, the kernels are singular. There, the evaluated stress differs between
    reference moduli (0.64 vs 0.70 for a value of 0.677 just off the node). Same cause as section 8.
- [ ] **Heterogeneous Green's functions: `get_kernels` for a whole mesh** (added 2026-11-17; likely the most used
  feature of the library). The user gives a mesh with `s` elements (the fault, where slip is the input) and the rest
  of the problem (`t` free surfaces, `u` clamped boundaries, `h` material interfaces), plus observation points. The
  function returns the displacement and stress at the observation points due to a unit one-ring hat on each `s` hat,
  with every other boundary condition made homogeneous: `t` and `u` values set to 0, `h` interfaces continuous.
  - **Method:** the matrix of the full solve is the same for every hat; only the right-hand side changes. Assemble
    and factor it once (as `solve(..., return_system=True)` does). Each `s` hat (one per slip component in plane
    strain) gives one right-hand side, and the non-`s` unknowns follow from one multi-RHS solve. The kernel is
    `G_obs,s + G_obs,other @ X`, built with the existing level-2 kernels.
  - **API:** shaped like `get_kernels(x_obs, y_obs, mesh, *, geometry, nu, ...)`. It returns the same field
    components (`ux, uy, sxx, ...` or `uz, sxz, syz`), each shaped `(n_obs, n_hats)`, plus the hat table (centre
    element, slip component), so that `fields = K @ slip_at_hat_centres`. The BC values given on `s` elements and
    on `t`/`u` elements are ignored, with a note in the docstring. Optionally, return the particular solution for
    the non-zero `t`/`u` values (for example a remote load) as a separate background term.
  - **Moduli come from the mesh only; no system `mu` argument.** Every element carries `mu_minus`/`mu_plus`, and
    the reference modulus of the kernels cancels. Checked 2026-11-17 on the layered antiplane mesh: `solve` with
    mu = 1 and mu = 3.7 gives the same `uz`, and the same true stresses (with `mu_field`), to 1e-14. So the
    reference is fixed internally at mu = 1.
    - **Stress at an observation point:** `Mesh.modulus_at(x, y)` (done, see below) gives the region's modulus;
      allow a `mu_obs` override, as `evaluate_fields` does with `mu_field`.
  - **Open points:**
    - a shortcut for the stress kernel at the `s` hat centres (self-interaction), which earthquake-cycle codes need.
    - storage for large meshes: `n_obs x n_hats` per component; maybe a `components=` argument.
    - keep the one-ring / connectivity-file hats (incl. `e4` junction rows), so the hat layout is the user's.
  - **Tests:**
    - column j equals `solve` + `evaluate_fields` with only hat j slipping;
    - `K @ s` equals the full solve for a smooth slip profile;
    - a homogeneous mesh (no `h`, no `t`) equals `get_kernels` with hat basis;
    - the layered antiplane example matches its image-series solution;
    - fault through a free surface (triple junction) in plane strain.

### T3. Tests and docs
- [x] Extend `tests/` with the T1 validation cases (`tests/test_validation.py`). Optional: a plane-strain Dundurs
  bimaterial dislocation solution.
- [x] Crossing cracks without remeshing (`tests/test_crossing_cracks.py`):
  - **Stable tangles:** random tangles with a tree-shaped crossing graph are stable (condition number 20 to 690),
    give hat == general, and have bounded stress at the crossings.
  - **Enclosed blocks:** tangles whose cracks cut out closed blocks (traction-free all round) slip about 20x the
    load, but both forms agree. That is the geometry, not the numerics.
  - **Node near another crack's collocation point:**
    - with `dr = 0`: converges down to d ~ 1e-10 L, loses precision near 1e-13 L, and blows up at 1e-16 L;
    - `dr = 0, d = 0` raises a clear `FloatingPointError`;
    - with the default `dr = 1e-9` the solve stays finite for all d.
    - Below d ~ dr L the offset puts the sampling point on the other side of crack B's line, and the solution jumps
      by ~40% to the limit reached from that side. The in-plane stress jumps across a slipping crack, so this is
      not round-off: it matches the `dr = 0` solution approached from the other side to 3e-7. (An earlier note
      blamed slip curvature; that was wrong.)
  - **Crack B passing exactly through A's collocation point P (pure remesh):** B's tips are fixed and its node next
    to P slides along B by s. The geometry is fixed, so the exact answer does not depend on s. Example panels (e)
    and (f) show the results; the tests are `test_crack_through_a_collocation_point_*`,
    `test_remeshing_*` and `test_collocation_point_on_another_crack_*`.
    - **Stable as s -> 0:** B always passes through P, so the offset sampling point stays on one side of B. The
      solution converges as s -> 0, stays finite down to s = 0, and shows no jump.
    - **Not invariant:** A's slip changes with s by 3% (8 elements per crack) or 0.8% (64), so it is
      discretization error that shrinks with refinement. The cause is that the true stress at P takes two values,
      one on each side of B, so A's traction condition at P is ambiguous. The `dr = 0` average and the one-sided
      default offset differ by ~3% at 8 elements.
    - **Erratic band:** when one of *B's* collocation points comes within ~0.1 element lengths of crack A, the
      answer is erratic (changes up to ~5x) and the condition number climbs from ~50 to 1e4. Exactly on A, where
      the collocation points of A and B coincide at P, it reaches 4e10. The band width in element lengths does not
      shrink with refinement.
- [x] **Warn when a collocation point lies on or close to another crack** (from the remesh test above).
  - **Function:** `bemcs.close_collocation_points(els, elements=None, tol=0.1)`. It returns the close pairs (`element`,
    `other`, `distance` in lengths of `element`), sorted by distance. The point-segment distances run vectorized
    in chunks: 0.4 s for 4000 elements, so no prefilter is needed yet.
  - **What is skipped:** the element itself and elements sharing a node with it, i.e. neighbours, kinks and
    junctions. Slip `"s"` rows are not checked because they do not sample the kernels.
  - **Covers both cases:** a node of one crack near another crack's collocation point is the same check, because
    the segment includes its end points.
  - **`Mesh.close_collocation_points(tol)`:** checks exactly the collocation points that `solve` uses.
  - **`assemble` / `solve(..., proximity_tol=0.1)`:** issue `bemcs.CollocationProximityWarning` listing the first
    pairs. `proximity_tol=None` silences it.
  - **No auto-fix.** The message points to moving a node or refining.
  - **Where it fires:** of the tests and examples, only the crossing-cracks cases warn. Random tangle seed 7 has a
    real pair at 0.078 L. Fault-in-box, split box, free surface and non-planar meshes are silent.
  - **Tests:** `test_proximity_check_*` in `tests/test_crossing_cracks.py`.
  - **Remedies still open:** move the collocation point along its element (needs a per-element collocation
    parameter in the assembly), or put a node at the crossing (turns it into a junction).
- [ ] Document heterogeneous elasticity (sections 2.1 and 2.2) in the README or wiki, plus a plane-strain tutorial.
- [ ] (Low priority) Update the GitHub wiki pages and tutorials for the revamped API. The existing tutorials use
  removed functions. Cover:
  - **Mesh CSV format:** `x1, z1, x2, z2, mu_minus, mu_plus`, plus BC columns. Plane strain:
    `BC_type_1, value_1, BC_type_2, value_2` (component 1/2 = x/y for `u`, `t`; t/n for `s`, `t_local`; `h` on
    both). Antiplane: `BC_type, value`. Shorthand: `BC_type, value[, value2]`.
  - **BC types:** `s`, `u`, `t`, `t_local`, `h`. Element orientation and the normal convention (tangent rotated
    +90 degrees). The -n / +n sides and `mu_minus` / `mu_plus`.
  - **Choosing moduli:** traction (`t`) elements take the modulus of the material they bound, matching the adjacent
    `h` side; the reference mu is arbitrary. Show the visual check with `bemcs.plot_mesh`: closed polygons of
    `t` / `h` elements should enclose uniform mu.
  - **Solving:** `bemcs.solve`, general (`basis={"slip": "quadratic"}`) vs one-ring hats (`{"slip": "hat"}`);
    force always trapezoids (default every other element; neighbours share one element, unlike hats); clamped fault
    tips (tip elements carry no BC); the collocation offset `dr` / `side`.
  - **Solution coefficients:** order (slip then force; source, then direction), `return_info=True` /
    `CoefficientLayout`, `split`, `element_coefficients`, `node_values`, `values_on_elements`. Quadratic coefficients
    are slip at s = -2w/3, 0, +2w/3.
  - **Fields and plots:** `evaluate_fields`, `plot_fields` (`derived=`, `mask=`, `mu_field=` for true stresses),
    `plot_mesh`.
  - **Lower-level API:** `get_kernels` / `get_slip_kernels` / `get_force_kernels`, `KernelInfo`, the kernel array
    layout, `traction_kernels`, `kernel_matrix`, `use_numba`.
  - **Known limitations:** singularity cancellation only for co-planar elements (kinks and junctions keep a log
    singularity); crack tangles that enclose blocks; collocation points on or near another crack (and nodes on
    another crack's collocation point); first-order
    accuracy of the trapezoid interface condition between collocation points.
- [x] Tidy up: `environment.yml` gains `mpmath` (the tests evaluate 50-digit references with it), `pkg_resources`
  is replaced, and library debug prints are removed.

- [x] **Speed (2026-11-16): numba by default, LU solve, cached transforms.**
  - **numba:** `use_numba=None` (the default everywhere) means numba when it is installed. `BEMCS_NUMBA=0` turns it
    off globally; `use_numba=False` still forces numpy per call. Cold compile of all four drivers is about 7 s,
    then they load from the on-disk cache in 0.14 s.
  - **Tests:** run with numba by default; `pytest --no-numba` runs the numpy path and skips the backend
    comparisons.
  - **Solver:** `solve(..., solver="auto")` uses LU for square systems, with a LAPACK condition estimate. It falls
    back to least squares (minimum norm) when the reciprocal condition is below 1e-12, e.g. for null spaces.
    `"lstsq"` and `"lu"` force one or the other.
  - **Caching:** `Mesh` caches its per-type subsets and basis transforms (`mesh.transform(kind, geometry=...)`).
  - **Measured on the layered example mesh (1181 elements):**
    - one solve: 6.0 s -> 1.1 s (least squares 3.3 s -> LU 0.06 s, same answer to 3e-11);
    - field on a 60,000-point grid: 26 s -> 4.8 s;
    - example: 93 s -> 25 s;
    - test suite: 97 s -> 23 s.

### T4. Possible extensions (later)
- [ ] Spatially varying `nu` (currently assumed uniform).
- [ ] Heterogeneous domains with topography/free surface, combining with existing non-planar fault tools.
- [ ] Numba acceleration for the kernels that do not have it yet: ~~plane-strain slip~~ (done, generated kernels);
  triangle forces remain (see below).
- [ ] **Triangle force sources (lowest priority).** Constant-force triangles, the area sources
  `bemcs.displacements_stresses_triangle_force_{antiplane,planestrain}` with `_nearfield` / `_farfield`. Both
  geometries are open:
  - Reference tests: point-force (Kelvin / Laplace) quadrature over the triangle, near and far field, as for the
    line forces.
  - Numba acceleration.

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
  - **Update (2026-11-16):** the element kernel formulas themselves now live in `bemAssembly.py`, as the generated
    `_quadratic_slip_f`, `_linear_force_g`, `_antiplane_slip_point` and `_antiplane_force_point` with their numba
    and numpy drivers.
  - **Removed from `bemcs.py` as superseded:** the original closed forms (`quadratic_kernel_farfield`,
    `quadratic_kernel_coincident`, the `displacements_stresses_*_no_rotation` functions,
    `f_slip_to_displacement_stress`) and unused leftovers (`phicoef`, `get_slip_slipgradient`,
    `get_quadratic_coefficients_for_linear_slip`). They are in git at 0ca988d.
  - **Moved to `tests/reference_kernels.py`:** the Crouch & Starfield constant-slip element (`constant_kernel`,
    `displacements_stresses_constant_no_rotation`, `rotate_displacement_stress`), still used as a reference.
  - **`bemcs.py` keeps:** geometry (`initialize_els`, `standardize_els_geometry`, `label_nodes`, `plot_els_geometry`,
    `inpolygon`, `discretized_line`), the quadratic basis functions, the Kelvin/Laplace point sources, the
    triangle-force kernels (area sources), and `get_strainenergy_from_stress`.
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
