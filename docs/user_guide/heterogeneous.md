# Heterogeneous materials

`bemcs` handles **piecewise-uniform** elastic media: regions of uniform shear modulus separated by sharp
interfaces, with a uniform Poisson's ratio. Each interface is meshed with force elements (`bc_type` `"h"`), and the
jump in modulus is replaced by an equivalent line force on the interface. The kernels stay those of a homogeneous
reference medium.

## Setting up an interface

1. Mesh the interface with elements whose `bc_type` is `"h"`.
2. Give each interface element the two moduli: `mu_minus` on its -n side and `mu_plus` on its +n side.
3. Give every traction boundary (for example a free surface) the modulus of the material it bounds, on both sides.
4. Check with `bemcs.plot_mesh(mesh)`: every closed polygon of `"t"` and `"h"` elements should enclose one uniform
   modulus.

The force is represented by trapezoids: linear force over 3-element patches, by default centred on every other
element along each chain of interface elements. Neighbouring trapezoids share exactly one element. Where a fault
crosses the interface the stride restarts and each side gets its own free force value; in plane strain, refine the
interface around the crossing (see {doc}`mesh_input`).

## What is solved

With uniform $\nu$, the true traction $t = \mu\, C \cdot n$ is continuous across an interface. In the reference
medium $\mu_0$ this requires an interface line force

$$
F = \mu_0\, t \left(\frac{1}{\mu_-} - \frac{1}{\mu_+}\right).
$$

Sampling the reference traction $(S_{\mathrm{ref}} \cdot n)$ on one side gives
$F = \kappa\,(S_{\mathrm{ref}} \cdot n)_{\mathrm{side}}$ with $\kappa = \mu_{\mathrm{side}}(1/\mu_- - 1/\mu_+)$
(`bemcs.interface_kappa`). Both sides give the same solution. See {doc}`../theory/heterogeneous` for the
derivation.

## Stresses in each material

The solve works in the reference medium, whose stresses are the true stresses divided by $\mu/\mu_0$.
`evaluate_fields` and `plot_fields` undo this by default: each point takes the modulus of the region it lies in
(`Mesh.modulus_at`, from the interface and traction elements), so the returned stresses are true stresses in every
material. Pass `mu_field="reference"` for the reference-medium stresses (see {doc}`solving`).

## Accuracy

The interface condition is exact at the collocation points and first order in element size between them.

## Faults that reach an interface

A node shared by a slip element and an interface element is treated as open for both: slip and force each taper to
zero there, and no continuity is imposed between them. Stress is singular where a fault crosses a material
contrast.
