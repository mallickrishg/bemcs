# Solving and using the solution

## `bemcs.solve`

```python
x = bemcs.solve(mesh, geometry="planestrain", nu=0.25)
```

- `geometry`: `"antiplane"` or `"planestrain"`.
- `mu`: reference shear modulus of the kernels (optional). It is arbitrary: the solution in true units does not
  depend on it. The default is 1, or the `mu` recorded by `background` (an explicit `mu` that differs from it raises
  an error). `assemble`, `evaluate_fields`, `plot_fields` and `get_kernels` (with a `Mesh`) take the same default.
- `nu`: uniform Poisson's ratio (plane strain).
- `side` and `dr`: when a condition has to be sampled off the element (see "Where conditions are sampled" below),
  the side it is sampled on, $-n$ (`side=-1`, default) or $+n$, and the distance in element lengths (default
  `1e-9`).
- `use_numba`: by default (`None`) the numba kernels are used when numba is installed; `False` forces plain numpy.
- `background`: a known field in the reference medium, for example a remote stress from
  `bemcs.uniform_background(stress, geometry=..., mu_medium=...)`, where `stress` is the true remote stress in the
  material of modulus `mu_medium`. Boundary and interface conditions then apply to the total field; pass the same `background` to
  `evaluate_fields` to get total fields.
- `return_system=True` also returns the linear system `(A, b, rows)`; `return_info=True` also returns a
  `CoefficientLayout` describing every entry of `x`.

By default (`solver="auto"`) a square system is solved by LU factorisation. If it is singular to working precision,
for example a domain with traction conditions only (a rigid-body null space), `solve` falls back to least squares and
returns the minimum-norm solution. `solver="lstsq"` always uses least squares; `solver="lu"` never falls back.

### General and reduced forms

The slip basis of the mesh picks the formulation:

- `basis={"slip": "quadratic"}` (general form): three coefficients per element and direction, plus continuity rows
  at every node.
- `basis={"slip": "hat"}` (reduced form): one one-ring hat per non-tip element and direction; continuity holds by
  construction.

They give the same solution. The reduced form is smaller and better conditioned.

### Which rows are written

- one boundary-condition row per direction at the centre of every non-tip slip element (general form) or hat
  element (reduced form);
- one interface row per direction at the centre of each trapezoid's middle element;
- in the general form, the continuity rows.

### Where conditions are sampled

Each condition row evaluates the kernels at its collocation point. Some quantities jump across an element:

- **Force elements:** displacement is continuous; traction jumps by the force density.
- **Slip elements:** displacement jumps by the slip. Traction on the element's own plane is continuous, but the
  stress tensor jumps (the stress along the element changes with the slip gradient), so traction on any other plane
  jumps too.

`solve` therefore samples exactly on the element, where the kernels take the continuous value, unless the row needs
a quantity that jumps there. In that case the point moves `dr` element lengths to side `side`. The moved cases are:

| Row | Where it lies | Sampled |
| --- | --- | --- |
| displacement (`"u"`, `"u_local"`) | on a slip element | on side `side` |
| traction or interface (`"t"`, `"t_local"`, `"h"`) | on a force element (its own or a coincident one) | on side `side` |
| traction (`"t"`, `"t_local"`) | on a slip element at an angle to its own element, e.g. where one crack passes through another's collocation point | on side `side` |
| traction on a slip element, e.g. a free surface | on its own element | on the element |
| displacement on a force element | on its own element | on the element |
| prescribed slip (`"s"`) | | no kernel is sampled |

Interface (`"h"`) rows are therefore always one-sided. This is intrinsic: the interface condition uses the modulus
on the sampled side, $\kappa = \mu_{\mathrm{side}}(1/\mu_- - 1/\mu_+)$ (see {doc}`heterogeneous`). With the
nodal (tent) force basis, nodes at a kink of the interface are sampled further off, at `kink_offset` element
lengths (default 0.5).

`evaluate_fields` does not move points: an observation point exactly on an element gets the two-sided average of
any quantity that jumps there.

### Proximity warning

If a collocation point lies within `proximity_tol` element lengths (default 0.1) of another element that is not a
neighbour, for example where one crack crosses another next to an element centre, `solve` issues a
`CollocationProximityWarning`. The condition at that point is ambiguous or poorly resolved. Move the nearby node,
or put a node at the crossing. `bemcs.close_collocation_points` lists the offending points.

## Using the solution

The coefficient vector `x` is ordered slip then force; within each, by source (element or patch, then basis node),
then direction ($t$, $n$ in plane strain). Use the helpers rather than indexing it directly:

```python
# Slip (or force) along the elements, at n_eval points per element
xe, ye, slip = mesh.values_on_elements(x, "slip", geometry="planestrain", frame="local")
# slip[:, 0] is shear (t), slip[:, 1] normal (n); with frame="global", x and y

# Displacements and stresses at arbitrary points
fields = bemcs.evaluate_fields(mesh, x, x_obs, y_obs, geometry="planestrain", nu=0.25)
fields["sxy"]

# Plot on a regular grid
fig, axes, grid = bemcs.plot_fields(
    mesh, x, geometry="planestrain", nu=0.25, xlim=(-2, 2), ylim=(-2, 2),
)
```

`evaluate_fields` and `plot_fields` return **true stresses** by default: each observation point takes the shear
modulus of the region it lies in, from `mesh.modulus_at(x, y)`. Displacements need no scaling. Choose otherwise with
`mu_field`:

- `None` (default): from the mesh. A mesh without moduli gives reference-medium stresses.
- a number, or a callable `mu_field(x, y)`: the true modulus at the points.
- `"reference"`: the unscaled stresses of the reference medium.

`Mesh.modulus_at(x, y)` finds the nearest interface (`"h"`) or traction (`"t"`, `"t_local"`) element and returns its
`mu_minus` or `mu_plus`, depending on which side of it the point lies. At a shared node it uses the element making the
smallest angle with the point, and a point exactly on an element gets `mu_minus`. The moduli of `"s"`, `"u"` and
`"u_local"` elements are ignored, since `solve` does not use them either. So the regions must be closed by `"h"`, `"t"`
and `"t_local"` elements, as `plot_mesh` checks.

## Lower-level API

`bemcs.get_kernels`, `get_slip_kernels` and `get_force_kernels` return displacement and stress kernels for any
basis at arbitrary observation points, with a `KernelInfo` describing the array layout. `traction_kernels` and
`kernel_matrix` build traction rows and flattened matrices from them. Use these to assemble custom systems, for
example with friction. See the {doc}`../api`.
