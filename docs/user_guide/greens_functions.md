# Green's functions

`bemcs.get_greens_functions` returns the displacement and stress at a set of observation points for unit slip on
each piece of a fault, with everything else in the mesh (free surfaces, clamped boundaries, material interfaces)
solved for. The fields of any slip distribution are then a matrix product. This is what geodetic slip inversions
and earthquake-cycle models need.

```python
gf = bemcs.get_greens_functions("mesh.csv", x_obs, y_obs, geometry="planestrain", nu=0.25)

gf.fields["uy"]               # (n_obs, n_sources): uy at every point for each unit source
u = gf(a)                     # all fields for source amplitudes a, one per source
u = gf(gf.amplitudes(slip))   # ... or for slip values at the sources' collocation points
```

## The mesh

The first argument is a mesh file (read with `Mesh.from_csv`) or a `Mesh`.

- **The fault** is every element with prescribed slip (`"s"`). Its slip values are ignored.
- **Every other condition is made homogeneous:** `"t"`, `"t_local"`, `"u"` and `"u_local"` values are set to 0,
  and `"h"` interfaces stay continuous.
- **Slip basis:** the sources are slip hats, so the mesh must use `basis={"slip": "hat"}`. That is the default
  when you pass a mesh file; `basis=` can change the force basis, for example `{"force": "nodal"}`.
- **Connectivity**, with a mesh file:
  - `None` (default): `bemcs` builds the patches (one-ring hats, including at triple junctions; force trapezoids
    with breaks at open ends, crossings and junctions);
  - one connectivity file, or a list of files: the slip or force kind of each file is inferred;
  - a dict `{"slip": path, "force": path}`.

  With a `Mesh`, its own connectivity is used.

The same mesh file works for both geometries: call `get_greens_functions` with `geometry="antiplane"` and again with
`geometry="planestrain"`.

## Sources

A source is one slip hat of the fault with unit amplitude, in one slip component:

- **antiplane:** one source per hat (slip in $z$);
- **plane strain:** two sources per hat, component 0 (shear, along the fault) and component 1 (normal, opening).

A hat is a smooth, non-negative bump over three fault elements (four next to a triple junction): 1 at the centre of
its middle element, falling to 0 with zero slope at its outer nodes (see {doc}`../theory/continuous_sources`). Fault
tip elements carry no hat of their own.

Neighbouring hats overlap (a hat is 1/6 at its neighbours' centres), so **amplitudes are not slip values**. To get the
fields of a slip distribution sampled at the sources' collocation points, convert it first with
`gf.amplitudes(slip)`.

## The result

`get_greens_functions` returns a `GreensFunctions`:

| Attribute or method | What it is |
| --- | --- |
| `fields[name]` | `(n_obs, n_sources)`: displacement and stress per unit source. Stresses are true stresses, using the mesh moduli at each point (`Mesh.modulus_at`); change this with `mu_obs` (a number, a callable, or `"reference"`). |
| `gf(a)` | All fields for amplitudes `a`, shaped like the observation points. |
| `gf.amplitudes(slip)` | Amplitudes whose slip at the sources' collocation points equals `slip`. |
| `gf.tractions(nx, ny, sx=None, sy=None, index=None)` | Traction kernels $\sigma \cdot n$ (true stresses) on planes with unit normal `(nx, ny)` per point: `(n, n_sources)` in antiplane; `(n, 2, n_sources)` in plane strain, global $(t_x, t_y)$, or local (shear, normal) when the unit tangent `(sx, sy)` is given. `index` picks the observation points, for example those on the fault. For amplitudes `a` the tractions are `K @ a`. |
| `gf.solution(a)` | The full BEM coefficient vector for amplitudes `a`, as `solve` returns it, e.g. for `mesh.values_on_elements`. |
| `sources` | Per source: `"element"` (its centre element), `"component"`, and `"x"`, `"y"` (its collocation point). |
| `slip_average` | `(n_fault, pe, n_sources)`: average slip of each source on each fault element, in the element's local frame (`pe` = 1 antiplane, 2 plane strain). Use it to compare with codes that use constant slip per element. |
| `slip_coefficients` | `(n_fault, 3, pe, n_sources)`: the quadratic slip coefficients of each source on each fault element. |
| `point_slip` | `(n_sources, n_sources)`: slip at each source's collocation point for unit amplitude of each source (1 on the diagonal, the hat overlap next to it). |
| `fault_elements` | Indices of the fault elements. |
| `matrix`, `rhs`, `coefficients`, `rows` | The BEM system $A X = B$ (one column per source), its solution and its row table (see `assemble`). |

`components=` limits the fields that are computed, to save memory on large problems.

**Stress-interaction kernels.** Put observation points on the fault (for example the element centres) and call
`gf.tractions` with the fault normals and tangents there; the result gives the shear and normal traction change on
the fault for any slip, as earthquake-cycle models need. `tractions` needs all stress components, so do not drop them
with `components=`.

## Cost

The BEM matrix does not depend on the source: it is assembled and factorised once, and all sources are solved
together, one right-hand side each. The kernels at the observation points are also evaluated once, for all
sources, with the same memory per point and unknown as `evaluate_fields` (see {doc}`solving`). The kernels use a unit reference modulus, which cancels, so
there is no `mu` argument.
