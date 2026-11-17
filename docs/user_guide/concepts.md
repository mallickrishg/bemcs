# Concepts

## Geometries

`bemcs` solves 2-D linear elasticity in two geometries, chosen with the `geometry=` argument:

- `"antiplane"`: one displacement component $u_z$ out of the plane; stresses $\sigma_{xz}$, $\sigma_{yz}$.
- `"planestrain"`: in-plane displacements $u_x$, $u_y$; stresses $\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{xy}$, with
  a uniform Poisson's ratio `nu`.

## Elements and source types

A mesh is a set of straight line elements. Each element is a source of one of two types:

**Slip elements** carry a displacement discontinuity (slip). Faults, cracks, and boundaries where displacement or
traction is prescribed are slip elements. Slip varies quadratically along each element.

**Force elements** carry a line force. They represent material interfaces: a jump in shear modulus is replaced by
an equivalent force on the interface, so the homogeneous-medium kernels can be used everywhere. Force varies
linearly along each element.

## Bases

Element coefficients are combined into the unknowns of the solve through a basis:

| Source | Basis | Unknowns | Notes |
| --- | --- | --- | --- |
| slip | `"quadratic"` (default) | 3 per element and direction | General form: continuity of slip and slip gradient is imposed by extra rows. |
| slip | `"hat"` | 1 per non-tip element and direction | Reduced form: one-ring hats satisfy continuity by construction. Same solution as `"quadratic"`. |
| force | `"trapezoid"` | 1 per 3-element patch and direction | The only force basis `solve` accepts. Patches default to every other element along each chain of force elements, restarting at fault crossings. |

A **one-ring hat** belongs to one element. It is non-zero on that element and its immediate neighbours (every
element that shares one of its end nodes), equals 1 at its own element centre, and goes to zero with zero slope at
the edge of that neighbourhood. At a triple junction the neighbourhood includes both other branches, so the hat
satisfies the junction conditions (slip conservation and the gradient conditions) by construction.

Both bases use the default patches unless you pass a connectivity; see {doc}`mesh_input`.

## Orientation and sign conventions

- Each element runs from node 1 `(x1, y1)` to node 2 `(x2, y2)`. Its unit tangent points from node 1 to node 2.
- The unit normal is the tangent rotated by +90 degrees.
- The **-n side** of an element is the side opposite its normal; the **+n side** is the side its normal points to.
  `mu_minus` and `mu_plus` are the shear moduli on those two sides.
- Local components are written $(t, n)$: along the tangent and along the normal. Global components are $(x, y)$.
- Reversing an element's orientation flips its tangent and normal, so local slip and traction values, and the
  roles of `mu_minus` and `mu_plus`, change sign or swap with it.

Use `bemcs.plot_mesh(mesh)` to draw the elements with their normals and the moduli on each side.

## Nodes: tips, overlaps and triple junctions

Where elements meet, `bemcs` imposes conditions on slip (handled automatically from the mesh topology):

- **Overlap node** (two elements): slip and slip gradient are continuous.
- **Open node** (fault tip): **clamped**, slip and slip gradient are zero. The element at a tip carries no boundary
  condition of its own; it only tapers the slip to zero. Extend a fault by one element beyond the region where
  you need to prescribe conditions.
- **Triple junction** (three elements): slip is conserved (the slip entering the junction equals the slip leaving
  it) and the slip gradients satisfy compatibility conditions. One branch must point into the junction and two out
  of it, or two in and one out. A junction where all three elements start, or all three end, is rejected with an
  error that names the elements; reverse one branch (and flip its boundary values) to fix it.
- Four or more elements at one node are not supported.
