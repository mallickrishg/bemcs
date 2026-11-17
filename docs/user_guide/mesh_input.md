# Mesh input and boundary conditions

A model is a `bemcs.Mesh`: element geometry, a boundary condition on every element, and the shear modulus on each
side. Build it in Python, or read it from CSV files: a mesh file and, optionally, one connectivity file per
source type.

## Boundary condition types

| Type | Prescribes | Components | Allowed on |
| --- | --- | --- | --- |
| `"s"` | slip at the element centre | local $(t, n)$ | slip elements |
| `"u"` (alias `"u_global"`) | displacement | global $(x, y)$; antiplane $z$ | slip and force elements |
| `"u_local"` | displacement | local (shear, normal); antiplane $z$, same as `"u"` | slip and force elements |
| `"t"` (alias `"t_global"`) | traction | global $(x, y)$; antiplane $\sigma \cdot n$ | slip and force elements |
| `"t_local"` | traction | local (shear, normal) | slip and force elements |
| `"h"` | a material interface (no value) | both | force elements |

The aliases `"u_global"` and `"t_global"` are stored, and written back to files, as `"u"` and `"t"`.

In plane strain each element can have a different type for each of its two components; pass `bc_type` and
`bc_value` with shape `(n_els, 2)`. Column 0 is $x$ for global types and shear (along the element) for local types;
column 1 is $y$ for global types and normal for local types:

| Column | Global types (`"u"`, `"t"`) | Local types (`"s"`, `"u_local"`, `"t_local"`) | CSV columns |
| --- | --- | --- | --- |
| `bc_type[:, 0]`, `bc_value[:, 0]` | $x$ | shear $t$ | `BC_type_1`, `value_1` |
| `bc_type[:, 1]`, `bc_value[:, 1]` | $y$ | normal $n$ | `BC_type_2`, `value_2` |

The CSV column names count from 1, but the array columns count from 0: `BC_type_1` is column 0.

Local and global types can be mixed across the two components. For example, a **roller** (no shear traction, no
normal displacement) is

```python
bc_type[e] = ["t_local", "u_local"]   # column 0: shear traction; column 1: normal displacement
bc_value[e] = [0.0, 0.0]
```

`"h"` must be on both components.

Displacement conditions on slip elements are sampled on the $-n$ side (see {doc}`solving`). On the boundary of a
body, orient the elements so that their $-n$ side faces the inside of the body.

Tractions are **true tractions**: they use the modulus on the side where they are sampled (`mu_minus` by default).

## Shear moduli on each side

Every element needs `mu_minus` and `mu_plus`:

- **Interface (`"h"`) elements:** the two materials. They must differ.
- **Traction (`"t"`, `"t_local"`) elements, such as a free surface:** the modulus of the material that boundary
  belongs to, on both sides. It must match the adjacent interface side.
- **Slip and displacement (`"s"`, `"u"`, `"u_local"`) elements:** not used by the rows or by `Mesh.modulus_at`, but
  required; use the surrounding modulus.

The reference modulus `mu` of the kernels is arbitrary and optional (default 1). The moduli of the interface and
traction elements also tell `evaluate_fields` which material each observation point is in (`Mesh.modulus_at`). Check the
setup with `bemcs.plot_mesh`: every closed polygon of `"t"` and `"h"` elements should enclose one uniform modulus.

## Building a mesh in Python

```python
mesh = bemcs.Mesh(
    els,                         # standardized element geometry
    basis={"slip": "hat"},       # or "quadratic" (default); force is always "trapezoid"
    bc_type=bc_type,             # (n_els,) or (n_els, 2) strings
    bc_value=bc_value,           # (n_els,) or (n_els, 2) floats
    mu_minus=mu_minus,           # (n_els,)
    mu_plus=mu_plus,             # (n_els,)
)
```

Elements with `bc_type` `"h"` become force elements; all others are slip elements. To put a displacement or
traction condition on a force element instead (for example a boundary that carries a line force rather than slip),
pass `labels=` (`"s"` or `"f"` per element), give the mesh file a `label` column, or list the element in a force
connectivity file.

## CSV format

`bemcs.Mesh.from_csv(path)` reads one row per element. Columns:

- geometry: `x1, z1, x2, z2`;
- moduli: `mu_minus, mu_plus`;
- optional `label`: `s` (slip) or `f` (force);
- boundary conditions (optional when the connectivity files carry them), one of:
  - **plane strain, per component:** `BC_type_1, value_1, BC_type_2, value_2` (columns 0 and 1 of `bc_type` and
    `bc_value`, see above);
  - **antiplane:** `BC_type, value`;
  - **shorthand, same type on both components:** `BC_type, value[, value2]` (`value` is column 0, `value2`
    column 1).

Coordinates are divided by `length_scale=` (for example `1e3` to convert metres to kilometres).

```text
x1,z1,x2,z2,mu_minus,mu_plus,BC_type_1,value_1,BC_type_2,value_2
-1.0,0.0,-0.9,0.0,1.0,1.0,t_local,1.0,t_local,0.0
...
```

## Connectivity files

By default `Mesh` builds its patches itself: one one-ring hat per non-tip slip element (`basis={"slip": "hat"}`)
and force trapezoids centred on every other element of each force chain. To choose the patches yourself, pass
`connectivity`:

```python
mesh = bemcs.Mesh.from_csv(
    "mesh.csv",
    basis={"slip": "hat"},
    connectivity={"slip": "slip_connectivity.csv", "force": "force_connectivity.csv"},
)
```

Each value can be a CSV path, a pandas DataFrame with the same columns, or an int array (element indices only, no
boundary conditions). Give either kind, or both.

### Element columns

One row per patch. `e1, e2, e3[, e4, ...]` are 0-based element indices (rows of the mesh file); `e2` is the centre
of the patch.

- **Slip hats:** `e1` and `e3` are the neighbours of `e2` at its start and end node. When `e2` touches a triple
  junction (for example a fault that breaks the free surface), `e4` is the third branch. The row is the one-ring
  hat of `e2`.
- **Force trapezoids:** exactly three columns, `e1, e2, e3`, along the chain. More columns raise a `ValueError`.
- `-1` (or an empty cell) means "no element". It is used in `e4` on ordinary rows of a file that also has junction
  rows; `e1` to `e3` must be real elements. Extra columns that are `-1` on every row are dropped.

With a header row, columns are found by name (`e1`, `e2`, ... and the boundary-condition columns). Without one,
every column is an element index: 3 columns, or 4 or more when there are junction rows.

### Boundary conditions in the connectivity file

A connectivity file with a header can also carry the boundary conditions, with the same columns as a mesh file:
`BC_type_1, value_1, BC_type_2, value_2` (plane strain), or `BC_type, value[, value2]` (antiplane, or the same
type on both components). Each row's condition is imposed at the centre of its `e2`.

- Elements in slip rows become slip elements; elements in force rows become force elements.
- Elements that are the centre of no row (tips, trapezoid ends) get no condition row.
- The mesh file can then leave out its boundary-condition columns. Its `label` column is needed only for an
  element that is in no row at all, for example a force element outside every trapezoid (it carries no force).
  Without a label, `Mesh` raises a `ValueError` naming the element.
- If you also give per-element `bc_type` / `bc_value`, they must agree with the rows at the centres.
- An element that is the centre of two rows raises a `ValueError`.

```text
e1,e2,e3,e4,BC_type_1,value_1,BC_type_2,value_2
0,1,2,-1,s,1.0,s,0.0
1,2,3,-1,s,1.0,s,0.0
2,3,4,9,s,1.0,s,0.0
...
```

### Example: a triple junction

Each row is one hat, and its condition is imposed at the centre of its own `e2`. A triple junction therefore needs
no condition of its own: the junction conditions (slip conservation and gradient compatibility) are built into the
hats. Each element that touches the junction has its own row, with the third branch in `e4` and its own condition.
Branches with different condition types can meet at a junction.

In Figure 1 a main fault A-B, with a splay C, has 4 elements per branch. A has prescribed slip (`s`), B a local
traction (`t_local`) and C a global traction (`t`). The slip connectivity file is:

```text
e1,e2,e3,e4,BC_type_1,value_1,BC_type_2,value_2
0,1,2,-1,s,1.0,s,0.0
1,2,3,-1,s,1.0,s,0.0
2,3,4,8,s,1.0,s,0.0
3,4,5,8,t_local,0.0,t_local,0.0
4,5,6,-1,t_local,0.0,t_local,0.0
5,6,7,-1,t_local,0.0,t_local,0.0
3,8,9,4,t,0.0,t,0.0
8,9,10,-1,t,0.0,t,0.0
9,10,11,-1,t,0.0,t,0.0
```

- **Junction rows** (centres 3, 4 and 8): `e1` and `e3` are the neighbours at the centre's start and end node, and
  `e4` is the other branch at the junction (8 for A and B, 4 for C).
- **Ordinary rows:** `e4` is `-1`.
- **Tips** (0, 7 and 11) have no row and no condition. Their slip tapers to zero, set by the neighbouring hat.
  A branch of a single element, with one end at the junction and the other open, is a tip too: it gets no row.

```{figure} ../figures/junction_connectivity.png
:width: 100%

**Figure 1. Connectivity rows at a triple junction.** (a) The mesh, coloured by the boundary condition of each
element's row; tips (grey) have no row. The junction (diamond) has no row of its own. (b-d) The three rows that
touch the junction: the support of each hat in colour, its centre `e2` drawn thicker. Each row carries the
condition of its own centre element.
```

To write such a file, build the mesh with per-element `bc_type` and `bc_value` and use `connectivity_table` (below).
The figure and table come from `docs/figures/make_mesh_figures.py`.

### Starting from the default layout

Write the default patches of a mesh to a file and edit them:

```python
table = mesh.connectivity_table("slip", geometry="planestrain")   # bc=True adds the BC columns
table.to_csv("slip_connectivity.csv", index=False)
```

- `mesh.patch_connectivity(kind)`: the rows in use (given or default), `-1` padded.
- `bemcs.one_ring_connectivity(els)`: the default hat rows, one per non-tip element, 4 wide at junctions.
- `bemcs.bemAssembly.one_ring_support(els, e)`: the hat row of one element.
- `mesh.split_force_trapezoids()`: trapezoid rows with the stride restarted at fault crossings (see below).

### Checks

- **`HatConnectivityWarning`** (at `Mesh(...)`, slip basis `"hat"` with explicit connectivity): a non-tip slip
  element is the centre of no row (its hat is missing), a row is centred on a tip element, or a row's elements are
  not all the neighbours of its centre (for example a junction row without `e4`). The message gives the expected
  row. `bemcs.bemAssembly.hat_connectivity_issues(els, conn)` returns the same problems as a list.
- **`TrapezoidCrossingWarning`** (at `solve`): a force trapezoid spans a node where a fault crosses the force chain,
  or where a stretch of overlapping slip and force elements ends. The force density changes regime there, so a
  trapezoid across it misrepresents the force and leaves a spurious net force on the interface.
- **`PinnedForceWarning`** (at `solve`): an interior node of a force chain is covered by no trapezoid, so its force is
  pinned to zero. See "Break points" below.

### Break points: fault crossings, force junctions and open ends

The force density can jump at some nodes of a force chain, so the default trapezoid layout breaks the chain there
(`mesh.trapezoid_restart_points()`):

- **fault crossings**, where a fault crosses an interface (`mesh.fault_crossings`);
- **ends of slip/force overlaps**, where a stretch of coincident slip and force elements ends;
- **force junctions**, nodes where three or more force elements meet, for example a layer interface meeting a
  vertical contact (`bemcs.force_junctions(els, labels)`, `mesh.force_junctions`);
- **open force ends**, force chains ending on a free surface.

On each side of a break the stride is counted from the break, and the element next to it gets a half-hat
`[e, e, e]`, so each side (each branch, at a junction) takes its own free force value there. If you give your own force connectivity, start from
`mesh.split_force_trapezoids()` to get the same layout.

In plane strain, also refine the force elements around a crossing, for example 4 times finer within about one fault
length of it.

**Use an odd number of elements between break points.** Stride-2 trapezoids cover every interior node of a segment
with an odd number of elements. With an even number, the node next to one end is left uncovered and its force is
pinned to zero, which gave about 5 times larger errors in a layered-medium test. `solve` then issues a
`PinnedForceWarning`. Fix it with an odd count (refining by a factor 3 keeps an odd count odd), a force connectivity
that covers every interior node, or the nodal force basis.
