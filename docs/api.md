# API reference

## Mesh and solver

```{eval-rst}
.. currentmodule:: bemcs

.. autosummary::
   :toctree: generated
   :nosignatures:

   Mesh
   solve
   assemble
   interface_kappa
   CoefficientLayout
   close_collocation_points
   CollocationProximityWarning
   TrapezoidCrossingWarning
   HatConnectivityWarning
   PinnedForceWarning
   one_ring_connectivity
   force_junctions
   uniform_background
```

### Mesh methods

```{eval-rst}
.. currentmodule:: bemcs

.. autosummary::
   :nosignatures:

   Mesh.from_csv
   Mesh.modulus_at
   Mesh.values_on_elements
   Mesh.patch_connectivity
   Mesh.connectivity_table
   Mesh.split_force_trapezoids
   Mesh.trapezoid_restart_points
```

## Fields and plotting

```{eval-rst}
.. currentmodule:: bemcs

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate_fields
   plot_fields
   plot_mesh
```

## Kernels

```{eval-rst}
.. currentmodule:: bemcs

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_kernels
   get_slip_kernels
   get_force_kernels
   KernelInfo
   traction_kernels
   kernel_matrix
```

## Geometry helpers

```{eval-rst}
.. currentmodule:: bemcs

.. autosummary::
   :toctree: generated
   :nosignatures:

   initialize_els
   standardize_els_geometry
   label_nodes
   plot_els_geometry
   discretized_line
```

## Point sources

```{eval-rst}
.. currentmodule:: bemcs

.. autosummary::
   :toctree: generated
   :nosignatures:

   kelvin_point_source_disp
   kelvin_point_source_stress
   laplacian_pointforce_disp
   laplacian_pointforce_sx
   laplacian_pointforce_sy
```

## Lower-level API (`bemcs.bemAssembly`)

These functions are not re-exported at the top level. They build the pieces that `solve` assembles: basis
transforms, element kernels and continuity operators. Use them to build custom systems (for example with
friction) or to inspect a basis. They may change more often than the functions above.

### Bases

```{eval-rst}
.. currentmodule:: bemcs.bemAssembly

.. autosummary::
   :toctree: generated
   :nosignatures:

   basis_transform
   one_ring_hat_coefficients
   hat_coefficients
   hat_elements
   one_ring_support
   hat_connectivity_issues
   trapezoid_coefficients
   trapezoid_connectivity
   patch_elements
```

### Element kernels

```{eval-rst}
.. currentmodule:: bemcs.bemAssembly

.. autosummary::
   :toctree: generated
   :nosignatures:

   element_kernels_slip
   element_kernels_force
```

### Operators

```{eval-rst}
.. currentmodule:: bemcs.bemAssembly

.. autosummary::
   :toctree: generated
   :nosignatures:

   continuity_rows
   slip_node_matrices
   force_node_matrix
   plane_strain_compliance
```

### Mesh helpers

```{eval-rst}
.. currentmodule:: bemcs.bemAssembly

.. autosummary::
   :toctree: generated
   :nosignatures:

   tip_elements
   fault_crossings
   node_gaps
   values_on_elements
```
