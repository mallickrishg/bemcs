# Boundary elements with continuous sources

## Quadratic slip elements

Slip on each element is quadratic. The three coefficients of an element are the slip at $s = -2w/3$, $0$ and
$+2w/3$, where $w$ is the element half-length. The displacement and stress kernels of each shape function are
evaluated in closed form, expanded about the nearest element end point so that they stay accurate (to the
float64 limit) close to nodes.

## Why continuity matters

The stress kernel of a single element is singular at its end points: as $1/r$ when the end slip is non-zero, and
as $\log r$ when the end slip gradient is non-zero. When slip and slip gradient are continuous across a node shared
by two co-planar elements, these singular parts cancel and the stress stays bounded up to and at the node.

On a kinked mesh the geometry itself is only continuous ($C^0$): its direction jumps at every kink. Smooth slip
removes the singularity due to the slip field but not the one due to the geometry, so a weak logarithmic
singularity remains at kinks and triple junctions. Its strength grows with the angle between neighbouring
elements.

## Continuity conditions

- **Overlap nodes:** slip and slip gradient are continuous.
- **Open nodes (tips):** slip and slip gradient are zero (clamped).
- **Triple junctions:** slip is conserved (the slip vector entering the junction equals the sum leaving it) and the
  slip gradients satisfy compatibility conditions.

## General and reduced forms (slip)

Continuity can be imposed in two equivalent ways.

**General form.** Each element keeps its three coefficients per slip direction, and the continuity conditions are
added as extra rows of the linear system (Figure 1). The element shape functions (Figure 1a) are the quadratics
that equal 1 at one coefficient position ($s = -2w/3$, $0$ or $+2w/3$) and 0 at the other two; they sum to 1, so
equal coefficients give uniform slip. Along a fault, the pieces join with continuous slip and slip gradient, and
the slip and its gradient go to zero at the clamped tips (Figure 1b).

```{figure} ../figures/slip_general.png
:width: 100%

**Figure 1. Slip, general form.** (a) The three quadratic shape functions of one element (half-length $w$); each
equals 1 at its own coefficient position (coloured dots) and 0 at the other two (open circles). (b) A slip field on
a 5-element fault: each element (one colour each) carries its own three coefficients (dots), and the continuity
rows make slip and slip gradient continuous at every node and zero at the clamped tips (squares).
```

**Reduced form.** The continuity conditions are built into the basis instead. Each non-tip element gets one
**one-ring hat** per slip direction, found by solving a small square system on the element and its neighbours:

- support: the element and every element that shares one of its end nodes;
- continuity conditions at the nodes inside the support;
- zero slip and slip gradient at the edge of the support (and at fault tips);
- unit slip at the element's own centre.

On a straight run of elements the hat is a piecewise quadratic over three elements (Figure 2a): it equals 1 at its
own centre and 2/3 at its element's ends, and each element is covered by three overlapping hats (Figure 2b), as
for quadratic B-splines. Tip elements carry no hat: they only taper the slip of their neighbour's hat to zero. At a
triple junction the support of a junction-adjacent element includes both other branches, so the hat crosses the
junction and satisfies the junction conditions by construction (Figure 2c-e). The hats span exactly the slip
fields that the continuity conditions allow, so both forms give the same solution, with one unknown per non-tip
element and direction.

```{figure} ../figures/slip_reduced.png
:width: 100%

**Figure 2. Slip, reduced form (one-ring hats).** (a) The hats on a 7-element fault; the shaded tip elements carry
no hat. (b) The three hats that overlap one element (own hat and its two neighbours' hats). Together, all hats sum
to a constant (4/3). (c-e) Hats at a triple junction, main fault A-B with a splay C: unit shear slip at the centre
of the circled element, drawn as an offset along each element's normal (blue: shear, orange: normal slip). The hats
of the main-fault elements (c, d) pass straight through the junction. The splay hat (e) peaks at the junction and
puts equal and opposite slip on A and B, as slip conservation requires.
```

## Force: linear elements and trapezoids

Line forces follow the same pattern with one order lower. Each force element carries two linear ramps (Figure 3a),
the element basis. The unknowns of the solve are **trapezoids**: a trapezoid spans three elements, with unit force
on its middle element, falling linearly to zero over the two outer elements, and continuous force at the two
internal nodes (Figure 3b). Trapezoids are centred on every other element along a chain of force elements, so
neighbouring trapezoids share exactly one element (where hats share two). Unlike slip, force has no continuity
conditions across branches: where the force density can jump, the chain is broken and the element next to the break
gets a half-hat (a single linear ramp with a free value at the break). Breaks are placed at force chains ending on a
free surface, at fault crossings, at the ends of slip/force overlaps, and at **force junctions**, where three or
more force elements meet (for example a layer interface meeting a vertical contact); there every branch ends with
its own free value. The force kernels are the integrated Kelvin solution (plane strain) or its
antiplane counterpart.

```{figure} ../figures/force_bases.png
:width: 100%

**Figure 3. Force bases.** (a) The two linear ramps of one force element; they sum to uniform force. (b) Trapezoids
along a 9-element chain, centred on the shaded elements; neighbouring trapezoids overlap on one element.
```

## Antiplane and plane strain

Both geometries use the same bases, shape functions and continuity rules:

- **Antiplane** has one slip component (the jump in $u_z$) and one force component. An element has 3 slip (or 2
  force) coefficients, and there is one hat per non-tip element (one trapezoid per patch).
- **Plane strain** has two components, shear $t$ and normal $n$ in the element's local frame, each with the same
  shape functions. An element has 6 slip (or 4 force) coefficients, and there are two hats per non-tip element (two
  trapezoids per patch), one per direction. Each hat or trapezoid is normalised to unit shear or unit normal value
  in the local frame of its own (middle) element. Continuity across nodes is imposed on the global $(x, y)$ slip or
  force vector, so at a kink a shear hat also carries some normal slip on the neighbouring element.

The figures were generated by `docs/figures/make_basis_figures.py` from the package's own basis transforms
({func}`~bemcs.bemAssembly.one_ring_hat_coefficients`, {func}`~bemcs.bemAssembly.trapezoid_coefficients`).
