# Heterogeneous elasticity: equivalent body forces

## Equivalent body force

For an isotropic medium with uniform Poisson's ratio $\nu$ and spatially varying shear modulus $\mu(\mathbf{x})$,
plane-strain Hooke's law is

$$
\sigma_{ij} = 2\mu\,\varepsilon_{ij} + \frac{2\mu\nu}{1-2\nu}\,\delta_{ij}\,\varepsilon_{kk}.
$$

Writing equilibrium for this medium and moving the terms that contain $\nabla\mu$ to the right-hand side gives the
equations of a *homogeneous* medium loaded by an equivalent body force. For a piecewise-uniform medium,
$\nabla\mu$ is non-zero only on the interfaces, so the body force collapses to a line force on each interface, and
only the interfaces need to be meshed. In antiplane strain the same argument applies to the single displacement
component $u_z$.

## Sharp-interface condition

Let $C = 2\varepsilon + \frac{2\nu}{1-2\nu}\operatorname{tr}(\varepsilon) I$, so that $\sigma = \mu C$. The true
traction $t = \mu\, C \cdot n$ is continuous across an interface. In a reference medium of modulus $\mu_0$ the
interface must carry the line force

$$
F = \mu_0\, t \left(\frac{1}{\mu_-} - \frac{1}{\mu_+}\right),
$$

where $\mu_-$ and $\mu_+$ are the moduli on the $-n$ and $+n$ sides. The reference traction
$S_{\mathrm{ref}} \cdot n$ jumps across the interface, so it must be sampled on one side. Sampling on the side with
modulus $\mu_{\mathrm{side}}$ gives

$$
F = \kappa\, (S_{\mathrm{ref}} \cdot n)_{\mathrm{side}}, \qquad
\kappa = \mu_{\mathrm{side}}\left(\frac{1}{\mu_-} - \frac{1}{\mu_+}\right),
$$

that is $\kappa = 1 - \mu_-/\mu_+$ on the $-n$ side and $\kappa = \mu_+/\mu_- - 1$ on the $+n$ side
(`bemcs.interface_kappa`). Both choices give the same solution.

## Linear system

The unknowns are the slip coefficients $s$ and the force coefficients $f$. The rows are:

- **slip (`"s"`) rows:** slip at an element centre equals the prescribed value;
- **displacement (`"u"`) and traction (`"t"`) rows:** sampled just off the element; tractions are true
  tractions, $(\mu_{\mathrm{side}}/\mu_0)\, S \cdot n$;
- **interface (`"h"`) rows:** the force density at the collocation centre minus $\kappa$ times the sampled
  reference traction equals zero, with contributions from both slip and force sources;
- **continuity rows:** in the general form only.

## Validation

The test suite checks:

- no contrast gives no interface force;
- very stiff and very compliant inclusions reproduce rigid-boundary and free-surface solutions;
- true traction is continuous across interfaces at collocation points;
- the solution does not depend on the sampled side;
- reversing or mirroring the problem gives identical or mirrored fields;
- in antiplane, the exact bimaterial image solution, with second-order convergence.
