# Known limitations

**Singularities at kinks and junctions.** Singular stresses cancel exactly only where neighbouring elements are
co-planar. At kinks and triple junctions a weak logarithmic singularity remains, set by the angle between the
elements. A stress that grows as you approach such a node is this geometric singularity, not numerical noise.
Refining the mesh reduces the angles and therefore the singularity.

**Triple-junction orientation.** One branch must point into a junction and two out, or the reverse. Junctions
where all three branches start, or all three end, are rejected. Nodes shared by four or more elements are not
supported.

**Fault tips are clamped.** Slip and slip gradient go to zero at tips, so crack-tip singularities are not
represented, and tip elements carry no boundary condition.

**Collocation points near another crack.** When a collocation point lies within about 0.1 element lengths of an
element of another crack, the condition there is ambiguous (stress jumps across a slipping crack) or poorly
resolved, and the solution can change erratically with small changes to the mesh. `solve` warns about these cases.

**Crack tangles that enclose blocks.** If cracks cut out a closed, traction-free block, the block can move almost
freely and slips are very large. That is the geometry, not the numerics.

**Narrow wedges with opening allowed.** Two nearly parallel branches at a junction (angle below about 10 degrees)
have a near-null mode in which the thin wedge between them slides as a block. With opening (normal slip) allowed,
the system is then poorly conditioned whatever the basis.

**Interface accuracy.** The heterogeneous-material condition is exact at collocation points and first order in
element size between them.

**Material model.** Piecewise-uniform shear modulus with a uniform Poisson's ratio. Spatially varying Poisson's
ratio is not supported.
