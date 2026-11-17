# bemcs

**B**oundary **E**lement **M**ethod with **C**ontinuous **S**ources: boundary value problems in 2-D linear
elasticity (antiplane and plane strain), built from

- **quadratic slip elements** whose slip and slip gradient are continuous across nodes, so the singular parts of
  neighbouring elements cancel;
- **linear force elements**, combined into trapezoidal patches, that represent the equivalent body forces of
  material interfaces (heterogeneous, piecewise-uniform elasticity).

Developed for geophysical and tectonic modelling, with a focus on earthquake-cycle deformation.

- **New here?** Start with {doc}`getting_started`.
- **Setting up a model?** The {doc}`user_guide/index` explains elements, sign conventions, the mesh input format,
  boundary conditions and the solver.
- **Want the maths?** See {doc}`theory/index`.
- **Before trusting a result:** read {doc}`limitations`.

```{toctree}
:maxdepth: 2
:hidden:

getting_started
user_guide/index
theory/index
limitations
api
```
