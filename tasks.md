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

1. **Compliance uses the wrong Poisson ratio (bug in the notes and the code, confirmed).**
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
4. **`'s'` rows differ from the notes.** Notes say `I s = s_given`. Code evaluates slip as a displacement
   jump across `+/-1e-9` from both slip and force kernels (force kernels are continuous so it cancels, but
   it is wasteful). Check the hat-basis DOF really equals slip at the central element, and that the `'s'` row can
   be replaced by the identity.
5. **Test script calls `bemcs.plot_BEM_field`**, which is defined in `bemAssembly` and not exported from
   `bemcs/__init__.py`. Confirmed by running the script in the `bemcs` env: it solves fine (cond. number ~1.18) and then fails at the first plot call with `AttributeError`. Use `GF.plot_BEM_field`, or export it.
6. **Trapezoidal force basis was previously flagged as unreliable** (commit `8fd5a0f`: "weird issues"; later
   commits fixed stride and arctan bugs). It still needs a standalone check against the constant-force or
   line-force result.
7. **Duplicated functions** in `bemcs.py` and `bemAssembly.py` (audited, see section 6). The 3 slip/gradient
   helpers and the 2 antiplane kernels agree numerically. The plane-strain force kernel does not (item 9).
8. **Minor:** a `1-2nu` typo in the notes' last sentence, `__init__.py` uses deprecated `pkg_resources`, and
   `solve_planestrain_heterogeneous` has `print(np.linalg.cond(A))` on every call.
9. **`bemcs.get_displacement_stress_kernel_force_planestrain` (the `bemcs.py` version) is wrong for any
   non-horizontal element (confirmed).** It agrees with the numba version only for horizontal elements, and differs
   by up to ~0.16 (stresses) for a 30-degree element. The numba version in `bemAssembly` matches an independent
   reference (Gauss-Legendre integration of the Kelvin point source) to ~1e-9. Cause: `rotate_vector` applies
   `rot_mats_inv` (global->local) to displacements instead of `rot_mats`. The stress rotation also differs for
   general angles. Nothing in the repo calls the buggy version (the heterogeneous solver goes through
   `get_kernels_trapezoidalforce_planestrain`, which uses the correct numba kernel), but it is public API.
   Its docstring also lists the wrong return order (`sxx, sxy, syy, ux, uy`; it actually returns `ux, uy, sxx, syy, sxy`).
10. **numba is a hard dependency despite the `try/except`** at the top of `bemAssembly.py`: line 16
    `from numba import njit, prange` runs unconditionally. The plane-strain force numba kernel has no `cache=True`,
    so it costs **~140 s of JIT compilation** in every fresh Python process (it dominates the test-suite runtime).
11. **11 public functions in `bemcs.py` are not exported** from `bemcs/__init__.py`, including
    `kelvin_point_source_disp/_stress` and `get_displacement_stress_kernel_constant` (list in `tests/test_api.py::KNOWN_UNEXPORTED`).
12. **A third copy of kernels lives in `examples/antiplane/GF.py`** (it also defines
    `get_displacement_stress_kernel_force_planestrain`, among others).
13. `bemcs.py` lines 3009, 3036 and 3063 use `"$\sigma_{..}$"` in non-raw strings (invalid escape, `DeprecationWarning`,
    a `SyntaxWarning` on Python ≥ 3.12).

---

## 5. Task list

### T0. Environment
- [x] Use the `bemcs` conda env (numpy 2.3.4, numba 0.62.1; `import bemcs` works). Do not use the base env (broken numpy).

### TA. Code audit and API consolidation (do before T1)
- [x] Add a `pytest` suite skeleton (`tests/`, `pytest.ini`); run with
  `conda run -n bemcs python -m pytest`. Current status: 70 passed, 2 expected xfail, ~2.5 min (numba JIT).
- [x] Static API checks (`tests/test_api.py`): `__init__` exports resolve, no function defined twice in a module,
  no undefined global names, list of duplicates across modules, list of unexported functions.
- [x] Numerical equivalence of duplicated pairs (`tests/test_duplicates_consistent.py`).
- [x] Plane-strain force kernels vs an independent Kelvin reference (`tests/test_force_kernels_reference.py`),
  at 0, 90, 30 and -40 degrees.
- [ ] Agree on the package layout and the unified kernel API (see section 7), then implement it.
- [ ] Fix issue 9 (numpy plane-strain force kernel), or replace it with the unified implementation.
- [ ] Add `cache=True` to every `@njit` and make numba an optional, lazily imported dependency (issue 10).
- [ ] Add reference tests for the remaining kernels: slip (plane strain and antiplane, constant and quadratic),
  antiplane force, triangle forces, hat-slip and trapezoid-force patches.
- [ ] Resolve `KNOWN_UNEXPORTED` and `KNOWN_DUPLICATES` until both are empty. Then the strict xfail
  `test_no_redundancy_between_modules` will start passing; remove its marker.
- [x] Remove `examples/antiplane/GF.py` (issue 12). It was removed with the antiplane examples that used it.

### T1. Correctness of plane-strain heterogeneous solver
- [ ] Fix issue 4.1 (compliance) and re-run `testing_HeterogeneousMaterialsPlanestrain.py`.
- [ ] Resolve 4.2 and 4.3 (side of evaluation, definition and helper for `alpha`).
- [ ] Fix 4.5 so the test script runs end to end.
- [ ] Validate: (a) `alpha = 0` must reproduce the homogeneous hat-slip solution; (b) the antiplane
  limit; (c) a circular/layered inclusion with a known analytic solution, or an independent FEM result;
  (d) symmetry `mu1 <-> mu2` with sign-flipped `alpha`; (e) traction continuity across the interface.
- [ ] Add a `'t'`-row test (prescribed projected strain).
- [ ] Verify the trapezoidal-force kernels in isolation (issue 4.6).

### T2. Package integration
- [ ] Move `solve_planestrain_heterogeneous` into `bemcs/bemAssembly.py` (mirroring `solveAntiplaneBEM`).
- [ ] Build the mesh/connectivity interface (CSV: `x1,z1,x2,z2,BC_type,value`, `'h'` rows carry `alpha` or `(mu1, mu2)`).
- [ ] Support multiple patches, and patches that share nodes (junctions between slip and interface elements).
- [ ] Add smoothness/continuity rows for hat patches that overlap neighbours (currently one patch = one hat).
- [ ] Remove the duplicated functions (issue 4.7) and decide on the public API in `__init__.py`.

### T3. Tests and docs
- [ ] Extend `tests/` with the validation cases from T1.
- [ ] Document heterogeneous elasticity (sections 2.1 and 2.2) in the README or wiki, plus a plane-strain tutorial.
- [ ] Tidy up: update `environment.yml` if needed, replace `pkg_resources`, remove debug prints.

### T4. Possible extensions (later)
- [ ] Spatially varying `nu` (currently assumed uniform).
- [ ] Heterogeneous domains with topography/free surface, combining with existing non-planar fault tools.
- [ ] Numba acceleration for the kernels that do not have it yet (plane-strain slip, triangles).

---

## 6. Duplicate-function audit (bemcs.py vs bemAssembly.py)

| Name | `bemcs.py` | `bemAssembly.py` | Agree? |
| --- | --- | --- | --- |
| `slip_functions`, `slipgradient_functions` | `(x, a)` | `(x, a, basis="quadratic")` | yes |
| `get_matrices_slip_slip_gradient` | `(els, flag, reference)` | `+ basis="quadratic"`; only `flag="node"` | yes (`flag="node"`) |
| `get_displacement_stress_kernel_slip_antiplane` | numpy, takes `els` | numba, takes arrays | yes |
| `get_displacement_stress_kernel_force_antiplane` | numpy, takes `els` | numba, takes arrays | yes |
| `get_displacement_stress_kernel_force_planestrain` | numpy, takes `els` | numba, takes arrays | **no**: numpy version wrong (issue 9) |

---

## 7. Unified kernel API (agreed design)

Decisions (2026-11-15): keep two modules; remove the old kernel functions (no deprecated wrappers);
`basis` defaults per source; return displacements and stresses as N-D arrays plus an info object.

Work happens on branch `coderevamp` (from `main` after merging `het-planestrain`). All `examples/**/*.ipynb`
were converted to percent-format `.py` scripts (`# %%` cells, outputs dropped, display/autoreload magics
commented out), and `notebooks/`, `point_source/` and `examples/hrf/` are untracked and git-ignored (personal work).
`examples/antiplane/` keeps only `antiplane101.py` (+ `example_mesh.csv`), `ConstructMesh.py` and
`HeterogeneousElasticity_forces.py` (run `ConstructMesh.py` first to generate its two CSV inputs); both run cleanly.
When old functions are removed, the `examples/**/*.py` scripts that call them must be migrated to `get_kernels`.

### 7.1 Module rule

- **`bemcs/bemcs.py`: single-element physics and geometry.** It covers `els` construction and standardization,
  analytic element solutions (numpy), basis/shape functions, rotations and point sources. It **never imports
  `bemAssembly`**, so the dependency is one-way (enforced by a test).
- **`bemcs/bemAssembly.py`: everything that combines elements.** This covers `get_kernels` and its info object, numba
  backends, hat/trapezoid patch bases, smoothness and BC operators, system assembly, solvers and field plotting.
- `bemcs/__init__.py` re-exports the public API, including `bemcs.get_kernels`.

### 7.2 Signature

```python
U, S, info = bemcs.get_kernels(
    x_obs, y_obs, els,
    *,
    geometry,              # REQUIRED: "antiplane" | "planestrain"
    source,                # REQUIRED: "slip" | "force"
    basis=None,            # None -> default for source (below)
    connectivity=None,     # (n_patches, 3) int; required for patch bases, rejected otherwise
    mu=1.0,
    nu=0.25,               # planestrain only
    use_numba=False,
)
```

| geometry | source | basis (default first) | DOFs per element / patch | numba |
| --- | --- | --- | --- | --- |
| antiplane | slip | **quadratic** | 3 per element `[c0 c1 c2]` | yes |
| antiplane | force | **linear**, trapezoid | 2 per element `[b0 b1]`; 1 per patch | yes |
| planestrain | slip | **quadratic**, constant, hat | 6 per element `[s0 s1 s2 n0 n1 n2]`; 2; 2 per patch `[s n]` | no |
| planestrain | force | **linear**, trapezoid | 4 per element `[t.b0 t.b1 n.b0 n.b1]`; 2 per patch `[t n]` | yes |

Here `s/n` = shear/normal slip, `t/n` = force along/normal to the element (local frame), and `b0/b1` = the
linear ramps `(w+s)/2w` and `(w-s)/2w`.

Errors: a missing `geometry`/`source` raises `TypeError` (keyword-only, no default). An unknown value or an
unsupported combination raises `ValueError`. `use_numba=True` without a numba kernel raises `NotImplementedError`
(no silent fallback). A patch basis without `connectivity` raises `ValueError`.

### 7.3 Outputs

- `U`: displacements, shape `(N_obs, n_u, N_dof)`, where `n_u = 1` (`uz`) or `2` (`ux, uy`).
- `S`: stresses, shape `(N_obs, n_s, N_dof)`, where `n_s = 2` (`sxz, syz`) or `3` (`sxx, syy, sxy`).
- Columns are element-major (patch-major for patch bases), then direction, then basis node. So
  `U[:, k, :] @ coefs` is displacement component `k` for a coefficient vector `coefs`.
- `info` (a `KernelInfo` dataclass) holds `geometry, source, basis, mu, nu, backend`,
  `displacement_components`, `stress_components`, and a per-column DOF table (source index, direction,
  basis node, frame). `print(info)` gives a readable summary.

### 7.4 Implementation notes

- numpy and numba backends share one source: `use_numba=False` calls the jitted function's `.py_func`.
  On the 3 numba kernels this is bit-identical, and it costs 0.003 s instead of a 148.6 s first-call JIT for plane-strain force.
- The numpy-only `els`-based duplicates in `bemcs.py` (including the buggy plane-strain force kernel,
  issue 9) are deleted rather than fixed.
- Triangle (area) force sources are not `els`-based and stay as separate functions for now.
- Mixed labels (`source` = per-element array of `"s"`/`"f"`): requested, but the column ordering is undecided.
  Option (a): element-major, follows the mesh. Option (b): all slip DOFs, then all force DOFs; this matches `x = [s; f]` in the
  heterogeneous formulation. Until decided, callers concatenate two calls along the DOF axis.
- Stages: (1) add `get_kernels` + tests alongside the old functions, with tests comparing old and new;
  (2) migrate internal callers (`bemAssembly` solvers, `examples/*.py`); (3) delete the old functions and
  the comparison tests; empty `KNOWN_DUPLICATES`.
