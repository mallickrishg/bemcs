# Viscoelasticity (linear Maxwell and Burgers) as a boundary-only problem

*Governing equations in rate form, loading and coupling to time steppers, implementation on the
heterogeneous-elastic solver (one-ring hat slip, trapezoidal force), tests, and open issues. Design
only: nothing here is implemented yet.*

---

## 0. Idea in one paragraph

For piecewise-uniform materials, the viscous strain of a linear Maxwell or Burgers body is divergence-free
in the interior of each region, once equilibrium is used, up to a pressure-gradient term in plane strain.
The viscous effect is therefore an extra source on **boundaries only**: interfaces where the rheology
jumps, and (in plane strain) every surface across which the pressure jumps, which includes faults inside
the region. The sources depend linearly on a handful of **boundary states** (tractions, pressures, and
their Kelvin-filtered versions). Their rates come from the existing elastic system matrix, which is
**unchanged**. The result is a small linear state-space model

$$
\dot{\mathbf y} = \mathbf A\mathbf y + \mathbf G\mathbf v + \mathbf g_r(t),
$$

driven by slip rates $\mathbf v$ and loading rates $\mathbf g_r$. It can be solved in closed form by
eigen-decomposition, or embedded in any ODE integrator together with a nonlinear fault law.

---

## 1. Rheology and notation

Total strain is $\varepsilon = \varepsilon^e + \varepsilon^M + \varepsilon^K$. The elastic part follows
Hooke's law with shear modulus $\mu$ and bulk modulus $K$. The Maxwell dashpot (viscosity $\eta_M$) and
the Kelvin element (modulus $\mu_K$, viscosity $\eta_K$) carry **deviatoric** strain only. With
$s = \sigma - pI$ the deviatoric stress and $p = \sigma_{kk}/3$ the mean stress:

$$
\dot\varepsilon^M = \frac{s}{2\eta_M},\qquad
\dot\varepsilon^K = \frac{s - s_K}{2\eta_K},\qquad
\dot s_K = \frac{s - s_K}{\tau_K},\quad s_K(0)=0,\quad \tau_K = \frac{\eta_K}{\mu_K},
$$

where $s_K = 2\mu_K\varepsilon^K$ is the **Kelvin-filtered deviatoric stress**. Hence

$$
2\dot\varepsilon^v = \frac{s}{\eta_M} + \frac{s - s_K}{\eta_K}.
$$

Special cases, all through one code path:

- **Maxwell:** no Kelvin element ($1/\eta_K = 0$, so $s_K$ never enters).
- **Zener (standard linear solid):** $1/\eta_M = 0$.
- **Elastic:** both off.

Rates used throughout (all have units of 1/time):

$$
\alpha_M = \frac{\mu}{\eta_M},\qquad \alpha_K = \frac{\mu}{\eta_K},\qquad \tau_K = \frac{\eta_K}{\mu_K}.
$$

Each side of an interface carries its own parameters (the $-n$ and $+n$ sides, as `mu_minus`, `mu_plus`):
$\mu_\pm,\ \eta_{M\pm},\ \mu_{K\pm},\ \eta_{K\pm}$. The reference medium of the kernels is $\mu_0$ with the
uniform Poisson's ratio $\nu$ of the elastic solver.

**Relaxation rates.** For a traction vector $\mathbf t$ and its filtered version $\mathbf t_K$ on a side, and
for the pressure $p$ and its filtered version $p_K$, define the stress rates

$$
\boldsymbol\rho = (\alpha_M+\alpha_K)\mathbf t - \alpha_K\mathbf t_K,
\qquad
\pi = (\alpha_M+\alpha_K)p - \alpha_Kp_K .
$$

They are the rates at which viscous flow removes deviatoric stress. For Maxwell, $\boldsymbol\rho = \alpha\mathbf t$
and $\pi = \alpha p$. The filtered states obey

$$
\dot{\mathbf t}_K = \frac{\mathbf t - \mathbf t_K}{\tau_K},\qquad
\dot p_K = \frac{p - p_K}{\tau_K}.
$$

**Laplace-domain modulus** (for the correspondence principle in the tests). For a step load the shear
response is that of an elastic body with

$$
\tilde\mu(s) = \left[\frac1\mu + \frac1{\eta_M s} + \frac1{\mu_K+\eta_K s}\right]^{-1},
$$

and with $K$ unchanged, $\tilde\nu(s)$ follows from $K$ and $\tilde\mu(s)$. For Maxwell this reduces to
$\mu s/(s+\alpha)$.

**States.** All states are the *total* current fields, including any ambient stress.

---

## 2. Antiplane strain

Antiplane uses the code's convention $\sigma = (\sigma_{xz},\sigma_{yz}) = \mu\nabla u$. There is no pressure
and no normal traction. The traction on an element is the scalar $t = \sigma\cdot\mathbf n$.

### 2.1 Equivalent force

Split $\nabla u = \sigma/\mu + \mathcal V$, where $\mathcal V$ is the viscous part, with

$$
\dot{\mathcal V} = \frac{\sigma}{\eta_M} + \frac{\sigma-\sigma_K}{\eta_K} = \frac{\boldsymbol\rho}{\mu},
\qquad \dot\sigma_K = \frac{\sigma-\sigma_K}{\tau_K}.
$$

The reference medium satisfies $\mu_0\nabla^2 u + f = 0$, so $f = -\nabla\cdot(\mu_0\sigma/\mu + \mu_0\mathcal V)$.
Inside a uniform region $\nabla\cdot\sigma = 0$. Since $\sigma_K$ is a filter of $\sigma$ with uniform
$\tau_K$, also $\nabla\cdot\sigma_K = 0$. So $\nabla\cdot\mathcal V = 0$, and the interior force vanishes.
Only the **interfaces** carry force, from the jump of the normal component:

$$
F = \mu_0t\left(\frac1{\mu_-}-\frac1{\mu_+}\right) + \mu_0\big(\mathcal V_n^- - \mathcal V_n^+\big).
$$

Traction $t$ is continuous across the interface, but $\mathcal V_n$ is not: the Kelvin state depends on the
side. So **each side of an interface carries its own Kelvin traction** $t_{K\pm}$.

### 2.2 Rate-form rows

Let $\dot S$ be the rate of the reference-medium stress sampled on side $\sigma\in\{-,+\}$ (from slip, from
force, and from loading). The true traction rate and the force rate are

$$
\dot t = \frac{\mu_\sigma}{\mu_0}(\dot S\cdot n)_\sigma - \rho_\sigma,
\qquad
\dot F = \kappa_\sigma(\dot S\cdot n)_\sigma + \frac{\mu_0}{\mu_\mp}\big(\rho_- - \rho_+\big),
$$

with $\kappa_\sigma$ = `interface_kappa` and the rate $\rho_\sigma$ built from the Kelvin state on side
$\sigma$. The interface (`'h'`) rows therefore keep today's left-hand side, and the viscous term moves to
the right-hand side:

$$
\dot F - \kappa_\sigma(\dot S\cdot n)_\sigma = \frac{\mu_0}{\mu_\mp}\big(\rho_- - \rho_+\big).
$$

The row is independent of the sampled side (the same algebra as for Maxwell).
Traction (`'t'`) rows on viscoelastic sides become

$$
\frac{\mu_\sigma}{\mu_0}(\dot S\cdot n)_\sigma = \dot t_{\mathrm{given}} + \rho_\sigma .
$$

A traction-free surface has $\dot t_{\mathrm{given}}=0$ and $t=0$, so it has no relaxation term. Slip
(`'s'`) rows are unchanged, with $\dot a = v$ on the right.

### 2.3 State equations

Let $\mathbf y = [\mathbf t;\ \mathbf t_K]$ be the stacked states, with the Kelvin tractions one per
viscoelastic side. Write $\mathbf r = \boldsymbol\Lambda\mathbf y$ for the relaxation rates above, and let

- $\mathbf M$: today's elastic system matrix, factored once;
- $\mathbf R$: maps relaxation rates to the right-hand sides of the rows above;
- $\mathbf C$: samples $(\mu_\sigma/\mu_0)\dot S\cdot n$ at the state points;
- $\mathbf S$: the selection that removes the relaxation from the sampling side;
- $\mathbf K$: the Kelvin filter, $(\mathbf t - \mathbf t_K)/\tau_K$.

The unknown rates are $\dot{\mathbf x} = \mathbf M^{-1}(\dot{\mathbf b} + \mathbf R\mathbf r)$, and so

$$
\dot{\mathbf y} = \big[(\boldsymbol\Gamma - \mathbf S)\boldsymbol\Lambda + \mathbf K\big]\mathbf y
  + \mathbf C\mathbf M^{-1}\dot{\mathbf b}, \qquad \boldsymbol\Gamma = \mathbf C\mathbf M^{-1}\mathbf R
$$

with $\dot{\mathbf b} = \mathbf b_v\mathbf v + \dot{\mathbf b}_r$ (slip rates, and loading). Call the bracket
$\mathbf A$ and the forcing $\mathbf G\mathbf v + \mathbf g_r$.

- **Maxwell limit:** $\mathbf y = \mathbf t$ and $\mathbf A = \boldsymbol\Gamma D_\alpha - D_\alpha$, the Maxwell matrix.
- **Zener:** one eigenvalue is zero, because the relaxed state is a finite stress rather than zero.
- **Initial state:** $\mathbf t(0)$ is the elastic response to the initial loading, and
  $\mathbf t_K(0)=0$. This is the problem of the original note, $\dot{\mathbf t} = \mathbf A\mathbf t$.

**A closed-form check: the Burgers inclusion.** For a circular inclusion in an elastic matrix
($\mu_m$), the interface operator acts on one angular mode as a scalar, with
$\beta = \mu_m/(\mu_i+\mu_m)$. The eigenvalues of $\mathbf A$ solve

$$
\lambda^2 + \left[\beta(\alpha_M+\alpha_K) + \frac1{\tau_K}\right]\lambda + \frac{\beta\alpha_M}{\tau_K} = 0,
$$

and these are exactly the zeros of $\mu_m + \tilde\mu_i(s)$ from the correspondence principle
For Maxwell it is the single root $\lambda = -\beta\alpha$. Burgers has two relaxation
times, and Zener ($\alpha_M=0$) has a zero root.

### 2.4 Faults embedded in a uniform region

The fault enters only through its slip rate. The viscosity does not jump across it, and traction is
continuous across it, so $\nabla\cdot\sigma = 0$ holds through the fault in the distributional sense. No
force is generated at the fault, for Maxwell or Burgers. Tractions at the fault points are outputs: they
are elements of $\mathbf y$ when the fault touches viscous material. A locked fault in a uniform Maxwell or
Burgers region therefore gives $\dot{\mathbf x}$ with no viscous source, and the surface displacement stays
at its coseismic value while stress relaxes everywhere (the correspondence-principle result, because
antiplane dislocation displacements do not depend on the shear modulus).

A fault that crosses an $\eta$-interface is split at the crossing (`fault_crossings`). A hat centre that
falls on the crossing is warned about, never moved.

---

## 3. Plane strain

### 3.1 Deviatoric structure

The viscous strain is deviatoric in three dimensions: $\varepsilon^v_{zz} = -(\varepsilon^v_{xx}+\varepsilon^v_{yy})\neq 0$.
Plane strain fixes the total $\varepsilon_{zz}=0$, so $\sigma_{zz}$ evolves, and $\sigma_{zz}$ never needs to be
tracked because $\sigma_{zz} = 3p - \sigma_{xx}-\sigma_{yy}$. The in-plane stress is

$$
\sigma_{ij} = 2\mu(\varepsilon_{ij}-\varepsilon^v_{ij}) + \lambda\theta\delta_{ij},\qquad i,j\in\{x,y\},
\qquad \theta = \varepsilon_{xx}+\varepsilon_{yy},
$$

and the mean stress is $p = K\theta$ exactly, since the viscous strain is trace-free. With uniform $\nu$ the
true pressure is the scaled reference pressure, $p = (\mu/\mu_0)p_{\mathrm{ref}}$.

Write $\mathcal V_{ij} = 2\varepsilon^v_{ij}$ for the in-plane block. Its rate is

$$
\dot{\mathcal V}_{ij} = \frac{\rho_{ij} - \pi\delta_{ij}}{\mu},
$$

with the relaxation rates of section 1, now with $\rho_{ij}$ built from the filtered stress
$\sigma_K = \text{filter}(\sigma)$ and $\pi$ from $p$ and $p_K = \text{filter}(p)$.

### 3.2 Equivalent force and harmonic pressure

The reference equation is $\nabla\cdot(\mathbf C_0:\varepsilon) + \mathbf f = 0$ with
$\mathbf C_0:\varepsilon = (\mu_0/\mu)\sigma + \mu_0\mathcal V$, so $\mathbf f = -\nabla\cdot[(\mu_0/\mu)\sigma + \mu_0\mathcal V]$.
In a uniform region, $\nabla\cdot\sigma = 0$ and $\nabla\cdot\sigma_K=0$, which leaves a **pressure gradient**
only:

$$
\dot f_i = \mu_0\partial_i\mathcal{P},\qquad
\mathcal{P} = \frac{\pi}{\mu} = \Big(\frac1{\eta_M}+\frac1{\eta_K}\Big)p - \frac{p_K}{\eta_K}.
$$

**The pressure is harmonic.** Take the divergence of the Navier equation. In a uniform region,

$$
(\lambda+2\mu)\nabla^2\theta = 2\mu\partial_i\partial_j\varepsilon^v_{ij}.
$$

The rate of the right side is a linear functional of the history of $-\nabla^2 p$, because
$\partial_i\partial_j\sigma_{ij}=0$ and $\partial_i\partial_j\sigma_{K,ij}=0$. For Maxwell,
$\nabla^2\dot\theta = -g_\nu\alpha\nabla^2\theta$, with $g_\nu = (1+\nu)/(3(1-\nu))$. Therefore
$\nabla^2 p$ stays zero if it is zero at the start. This holds for an elastic start with no body force,
and for any state produced by this model. The argument uses the rate equations, not the elastic Navier equation, which no longer holds
once viscous strain exists. **For a user-supplied initial state, the
assumption is that the interior pressure is harmonic** (document it; the solver cannot check it).

### 3.3 The distributional gradient: no extra normal force, a local pressure law

Extend $\mathcal{P}$ by zero outside viscous material. The force rate is then the full **distributional**
gradient $\mu_0\nabla\mathcal{P}_{\mathrm{dist}}$. It contains the volume term in each region, and a line
force $\mu_0[\mathcal{P}]\mathbf n$ on every surface where $\mathcal{P}$ jumps: interfaces, faults inside
the region, and free surfaces (where it jumps to zero). Writing the total force as

$$
\dot{\mathbf f}_{\mathrm{total}} = \dot F'\delta_I + \mu_0\nabla\mathcal{P}_{\mathrm{dist}},
$$

the line-force unknowns of the solver are $\dot F'$ (the elastic-contrast and shear parts), and the
pressure part is **not** a separate normal line force. A $G_{ik}n_k$ term that appears when Green's identity is applied region by region cancels
against the pressure part of the line force, as follows.

**Dilatation.** The dilatation of the reference medium obeys $(\lambda_0+2\mu_0)\nabla^2\dot\theta = -\nabla\cdot\dot{\mathbf f}$,
and since $\nabla\cdot(\mu_0\nabla\mathcal{P}_{\mathrm{dist}}) = \mu_0\nabla^2\mathcal{P}_{\mathrm{dist}}$,

$$
\dot\theta = \dot\theta^{(\text{slip},\dot F')} - \frac{\mu_0}{\lambda_0+2\mu_0}\mathcal{P}_{\mathrm{dist}} .
$$

The viscous pressure source acts on the dilatation **locally**. With $p = (\mu_\sigma/\mu_0)K_0\theta$, this gives

$$
\dot p_\sigma = \frac{\mu_\sigma}{\mu_0}\hat p_\sigma - g_\nu\pi_\sigma,\qquad
g_\nu = \frac{1+\nu}{3(1-\nu)}
$$

where $\hat p_\sigma$ is the reference pressure rate from slip and from $\dot F'$ **only**, computed with
the existing kernels, and $\pi_\sigma$ is the relaxation rate on that side. The pressure state therefore
needs no new kernels: it is the existing pressure kernel plus a local relaxation term.

**Displacement.** A pure-gradient force in the reference medium gives the irrotational field
$\dot{\mathbf u}^{\mathrm{vol}} = \nabla\Phi$, with $\nabla^2\Phi = -\frac{\mu_0}{\lambda_0+2\mu_0}\mathcal{P}_{\mathrm{dist}}$,
that is, a Newtonian (logarithmic) potential of $\mathcal{P}$. Because $\mathcal{P}$ is harmonic in each
region, Green's second identity turns it into integrals over the surfaces $\Gamma$ where $\mathcal{P}$ jumps:

$$
\dot u^{\mathrm{vol}}_i(\mathbf x) = \mu_0\sum_\Gamma\int_\Gamma
\Big([\mathcal{P}]\partial_{n'}H_i - H_i[\partial_{n'}\mathcal{P}]\Big) dS',
\qquad
H_i = \frac{1-2\nu}{8\pi\mu_0(1-\nu)}r_i\ln r,\quad r_i = x_i - x'_i,
$$

with $[\cdot]$ the jump from the $-n$ to the $+n$ side. A free surface is one-sided, with the outside
value of $\mathcal{P}$ equal to zero. The coefficient equals $\mu_0/(4\pi(\lambda_0+2\mu_0))$, matching the
potential form. The kernel obeys $\nabla'^2H_i = \partial G_{ik}/\partial x'_k$ in 2-D (using
$\nabla^2(x_i\ln r) = 2x_i/r^2$). Stresses follow from $\sigma_{ij} = 2\mu_0\partial_i\partial_j\Phi + \lambda_0\theta\delta_{ij}$,
which has $1/r$ and (for the double layer) hypersingular terms, handled like the existing slip kernels.

### 3.4 Interface and traction rows

Let $\mathbf t$ be the true traction vector (continuous across an interface), with $\boldsymbol\rho_\sigma$
and $\pi_\sigma$ the relaxation rates on side $\sigma$. The true traction rate is

$$
\dot{\mathbf t} = \frac{\mu_\sigma}{\mu_0}(\dot S\cdot n)_\sigma - \boldsymbol\rho_\sigma + \pi_\sigma\mathbf n .
$$

The interface rows for the line-force unknowns $\dot F'$ become

$$
\dot F' - \kappa_-(\dot S\cdot n)_- = \frac{\mu_0}{\mu_+}\big(\boldsymbol\rho_- - \boldsymbol\rho_+\big) + \mu_0\kappa_-\mathcal{P}_-\mathbf n,
$$

$$
\dot F' - \kappa_+(\dot S\cdot n)_+ = \frac{\mu_0}{\mu_-}\big(\boldsymbol\rho_- - \boldsymbol\rho_+\big) + \mu_0\kappa_+\mathcal{P}_+\mathbf n,
$$

which are equivalent (given the reference-traction jump $-(\dot F' + \mu_0[\mathcal{P}]\mathbf n)$ across the line force). Here $\dot S$ contains slip, $\dot F'$,
loading, **and the pressure layer** $\dot u^{\mathrm{vol}}$ of section 3.3. Traction rows become

$$
\frac{\mu_\sigma}{\mu_0}(\dot S\cdot n)_\sigma = \dot{\mathbf t}_{\mathrm{given}} + \boldsymbol\rho_\sigma - \pi_\sigma\mathbf n .
$$

### 3.5 State equations

At each collocation point that touches viscous material, the states are the total traction vector, and one
pressure, one Kelvin traction and one Kelvin pressure for each viscous side:

$$
\dot{\mathbf t} = \frac{\mu_\sigma}{\mu_0}(\dot S\cdot n)_\sigma - \boldsymbol\rho_\sigma + \pi_\sigma\mathbf n,\qquad
\dot p_\sigma = \frac{\mu_\sigma}{\mu_0}\hat p_\sigma - g_\nu\pi_\sigma,
$$

$$
\dot{\mathbf t}_{K\sigma} = \frac{\mathbf t - \mathbf t_{K\sigma}}{\tau_{K\sigma}},\qquad
\dot p_{K\sigma} = \frac{p_\sigma - p_{K\sigma}}{\tau_{K\sigma}} .
$$

The system closes as in section 2.3: $\dot{\mathbf y} = \mathbf A\mathbf y + \mathbf G\mathbf v + \mathbf g_r$, with
$\mathbf y$ containing $\mathbf t$, $p$, $\mathbf t_K$ and $p_K$. For Maxwell the Kelvin states drop out.
The matrix $\mathbf M$ is unchanged: the pressure layer enters only as additional columns of the
right-hand side.

### 3.6 Faults inside the viscoelastic region

A fault in a uniform region has no $\eta$ jump, but in plane strain it is **not** passive: the pressure
jumps across it, and the pressure layer acts on both faces. Two quantities change.

**(a) The pressure jump across the fault.** For a planar fault with tangential slip $s(\ell)$, local
analysis (traction continuous, $[u_t] = s$, $[\varepsilon_{tt}] = \partial_\ell s$, $[\varepsilon_{zz}]=0$) gives

$$
[p] = 2\mu g_\nu\big(\partial_\ell s + [\varepsilon^v_{nn}]\big),\qquad
2[\dot\varepsilon^v_{nn}] = -\frac{[p]}{\eta_M} - \frac{[p]-[p_K]}{\eta_K},
$$

since $s_{nn}$ is continuous only through the traction and $p$ jumps. In rate form, with $[\pi]$ the jump
of the relaxation rate and $g_\nu$ as above:

$$
[\dot p] = 2\mu g_\nu\partial_\ell v - g_\nu[\pi],\qquad
[\dot p_K] = \frac{[p] - [p_K]}{\tau_K}
$$

(the coefficient $2\mu g_\nu = 2\mu(1+\nu)/(3(1-\nu))$ is the elastic jump, and the decay rate for
Maxwell is $g_\nu\alpha = 3K\alpha/(3K+4\mu)$). This is the same
local law as section 3.3, evaluated on the two faces, so the BEM reproduces it automatically. It is also a
**test**. Three things follow.

- The jump is driven by the slip-rate **gradient** $\partial_\ell v$, so smooth slip matters: the one-ring
  hats have continuous gradient, and the jump is representable.
- After the slip stops, $[p]$ relaxes on a locked fault: Maxwell with one exponential, Burgers with two
  (the eigenvalues of the 2 by 2 matrix with rows $(-g_\nu(\alpha_M+\alpha_K),\ g_\nu\alpha_K)$ and
  $(1/\tau_K,\ -1/\tau_K)$).
- The jump sources a **normal line force** on the fault, $\mu_0[\mathcal{P}]\mathbf n$, which is the
  fault-face part of the pressure layer. Elastically the fault has no such force.

**(b) The fault-normal traction.** The normal component of the true traction relaxes toward the pressure:

$$
\dot t_n = \frac{\mu_\sigma}{\mu_0}(\dot S\cdot n)_{\sigma,n} - \rho_n + \pi_\sigma,
\qquad \dot t_t = \frac{\mu_\sigma}{\mu_0}(\dot S\cdot n)_{\sigma,t} - \rho_t,
$$

so the shear traction relaxes to zero but the normal traction relaxes toward the (side-dependent)
pressure. On the two faces the traction must agree, which requires

$$
\frac{\mu}{\mu_0}[\dot S_n] = -[\pi],
$$

and this is exactly what the line force $-\mu_0[\mathcal{P}]$ gives, since $\pi = \mu\mathcal{P}$ and $\mu$ is
continuous across the fault. So the formulation satisfies it identically, and sampling either face
gives the same $\dot t_n$ (the side-independence test). Consequences for fault laws:

- **The effective normal stress** $-t_n$ changes by viscous flow, in addition to the elastic stiffness
  response to slip, so a fault law that uses normal stress (Coulomb, rate-and-state with variable normal
  stress) must take $t_n$ from the state vector.
- Fault shear traction $t_t$ relaxes with $\rho_t$, and its stiffness to slip is the elastic one at
  $t=0^+$ and decreases afterwards.

**(c) Jump of the normal derivative.** The Green's-identity formula also needs $[\partial_n\mathcal{P}]$.
For a planar fault with purely tangential slip in a uniform region, $[\partial_n p]=0$ at all times
(Maxwell and Burgers). It is zero at $t=0^+$, because the elastic dislocation pressure is a pure
double layer, and its rate is homogeneous in $[\partial_n p]$ itself. With a normal (opening) slip
component, or a curved fault, $[\partial_n p]\ne 0$ and it comes from the Dirichlet-to-Neumann
(DtN) map of each region: harmonic $p$ in the region with Dirichlet data on **all** its boundary,
including both faces of every fault. Curvature also adds terms of order (curvature)$\times s$ to the jump
relations of (a).

**(d) Faults in antiplane** have none of this: no pressure, no normal traction (section 2.4).

### 3.7 New kernels (plane strain)

- **Pressure layer:** displacement-rate and stress-rate kernels for a unit $[\mathcal{P}]$ double layer
  and a unit $[\partial_n\mathcal{P}]$ single layer on a linear element, from $H_i$ and $\partial_nH_i$.
  They are elementary ($r_i\ln r$ and derivatives). Derive symbolically, then add at level 3.
- **DtN map** (or an equivalent single-layer representation of $\mathcal{P}$ in each region), needed only
  where $[\partial_n\mathcal{P}]\neq 0$.
- **Nothing new** for the pressure states themselves: $\hat p$ is the existing pressure kernel.

---

## 4. Loading and coupling to time steppers

The rate form is a standard linear state-space model, so any ODE integrator can embed it.

### 4.1 Inputs and outputs

- **State:** $\mathbf y$ (tractions, pressures and Kelvin states on boundaries, sections 2.3 and 3.5).
- **Inputs:** slip rates $\mathbf v$ on all `'s'` elements (the loaded fault, other faults, loading
  dislocations), and loading rates $\mathbf g_r(t)$.
- **Outputs:**
  - fault tractions: elements of $\mathbf y$ (shear, and in plane strain normal), including
    viscous relaxation and the elastic fault stiffness;
  - velocities at observation points: $\dot{\mathbf u}_{\mathrm{obs}} = \mathbf U_v\mathbf v + \mathbf U_y\mathbf y$;
    displacements by integrating the velocities.

The matrix $\mathbf G$ contains the **elastic fault-to-fault stiffness** (the same matrix as the fault
tractions of `get_greens_functions`) as its fault rows, plus the columns that load the interface states.
Faults interact with each other and with the interfaces through $\mathbf G$.

### 4.2 Loading

1. **Kinematic plate motion (recommended).** A deep or extended creeping dislocation with prescribed
   rate $v_{pl}$: `'s'` elements whose $v$ is prescribed. It is part of $\mathbf G\mathbf v$, gives
   bounded stresses, and is the standard backslip (Savage and Prescott) setup.
2. **Uniform far-field stress rate** $\dot\sigma_\infty$: adds the reference stress to the sampled
   reference traction (and in plane strain a pressure rate), so $\mathbf g_r$ is constant. A steady
   stress rate drives a Maxwell region to flow without bound, so this is mostly for validation (the
   inclusion test).
3. **Imposed boundary velocity** (`'u'` rows with time-dependent value): the same, with a
   time-dependent right-hand side.

The initial-value problem (relaxation after a coseismic step) is the special case $\mathbf v = 0$, $\mathbf g_r = 0$.

### 4.3 The user's time-stepping loop (for example rate-and-state with RK45)

```
state = [ slip a, fault-law state theta, y (viscoelastic boundary states) ]

tau_f  = fault rows of y                     # includes relaxation and elastic stiffness
v      = fault_law(tau_f, theta)             # rate-and-state, radiation damping, ...
da     = v
dtheta = state_law(v, theta)
dy     = A @ y + G @ v + g_r(time)           # two small dense matrix-vector products
```

- The system is boundary-only, so $\mathbf A$ and $\mathbf G$ are small dense matrices.
- The fault's elastic traction and its viscous relaxation come from one state vector. No separate
  stress bookkeeping, and no convolution with a time-dependent kernel: the state is a Markov embedding
  of the memory.
- **Initial condition.** $\mathbf y(0)$ must be the actual state, including the Kelvin states, not
  zero. For a single event it is the elastic response. For earthquake cycles it needs a spin-up
  (viscous memory lives in $\mathbf y$), or a periodic steady state. Saving and restoring $\mathbf y$
  restarts a run.
- **Stiffness.** The eigenvalues of $\mathbf A$ lie in $[-\alpha_{\max}, 0]$. They stiffen an explicit
  integrator only for low viscosity (an asthenospheric channel). Otherwise the fault dynamics set the step.
  For stiff cases use an implicit method (Radau, BDF) or an exponential integrator with the modal form.
- **Closed form.** With $\mathbf A = \mathbf T\Lambda\mathbf T^{-1}$ and prescribed piecewise-linear
  $\mathbf v(t)$, the solution is a sum of exponentials, for coseismic steps, constant creep and
  afterslip (Duhamel). The zero mode of Zener integrates to $t$, and the other modes to $(e^{\lambda t}-1)/\lambda$.
- **Reduction.** The response is a Prony series. Truncating to the slowest $m$ modes, or balanced
  truncation, shrinks the system if it is large.

### 4.4 Proposed interface

```python
ve = bemcs.ViscoelasticOperator(mesh, geometry=..., mu=..., nu=...)
ve.A, ve.G, ve.g_r, ve.state_slice["fault"]    # dense, for the user's own right-hand side
ve.rhs(y, v, time)                             # convenience dy/dt
ve.modes()                                     # eigen path for closed-form histories
ve.reduce(n_modes)                             # optional modal truncation
ve.initial_state(...)                          # elastic response to the initial loading
```

---

## 5. Implementation plan (one-ring hat slip, trapezoidal force)

**Unknowns and collocation stay as they are.** Slip rates sit on one-ring hats (one per non-tip slip
element and component). Force rates $\dot F'$ sit on trapezoids, with the `'h'` rows at the trapezoid
collocation points. The states live at **the same points**: tractions at every trapezoid `'h'` point and
every hat sample point that touches viscous material, with the pressure and Kelvin states per side. The
number of states is therefore proportional to the number of affected rows, and the relaxation map
$\mathbf R$ is a sparse selection with the weights above.

**Mesh input.** Per side, optional columns `eta_M_minus`, `eta_M_plus`, `mu_K_minus`, `mu_K_plus`,
`eta_K_minus`, `eta_K_plus` (default: elastic, `inf`), next to `mu_minus` and `mu_plus`. Maxwell leaves the
Kelvin columns off, and Zener sets `eta_M` to `inf`. Checks (warnings, never rewrites):

- inconsistent parameters on the two sides of a shared node;
- a hat centre on an $\eta$-crossing;
- in plane strain, a viscous region not closed by the mesh (the DtN map needs it);
- an interface element with $\mu_-=\mu_+$ but a rheology jump (labelled `'h'` even without a modulus jump).

**Code layout (proposal).**

```
bemcs/viscoelastic.py
    relaxation_system(mesh, *, geometry, mu, nu=None)   -> RelaxationSystem(M_lu, R, C, S, Lambda, K, state_info)
    ViscoelasticOperator.A, .G, .g_r, .modes(), .reduce(), .rhs(), .initial_state()
    ViscoelasticOperator.fields(x_obs, y_obs, t)        -> velocities / displacements / stresses
bemAssembly: pressure-layer kernels (H, dH/dn, stresses) for linear elements   (plane strain, stage 3)
bemcs.get_greens_functions(..., times=...)              -> time-dependent Green's functions as mode sums
```

`solve_ivp` (BDF) on the same state-space model is the cross-check for the eigen path, and the route to
nonlinear rheology later.

**Stages.**

1. Antiplane: interface rows, faults in viscous regions, Maxwell and Burgers (section 2). Tests 7.1,
   7.2 and 7.5.
2. Plane strain, viscous regions with no embedded faults: the pressure states, relaxation, and the
   pressure layer for interfaces (sections 3.1 to 3.5). Test 7.3 (inclusion, layered cases).
3. Plane strain with faults inside viscous regions: fault-face pressure layer, $[p]$ and $t_n$
   behaviour, DtN where needed (section 3.6). Tests 7.3 (fault in a full space and a half-space).
4. Time-dependent Green's functions, the `ViscoelasticOperator` interface, an RSF coupling
   example (setup, solve, plots only), and the linear-viscous fault.

---

## 6. Open questions

1. **Unbounded regions in plane strain.** For a Maxwell half-space, the decay of $p$ and $\mathcal{P}$ at
   infinity is needed for the local law of section 3.3 and the boundary integrals. A uniform ambient
   in-plane stress is not hydrostatic in plane strain (the deviatoric part is nonzero), so it relaxes; that
   is consistent but must be tested.
2. **DtN or single-layer representation** of $\mathcal{P}$ for $[\partial_n\mathcal{P}]$ when the slip has a
   normal component or the fault is curved. Check whether the local relation of section 3.6 (a) keeps its
   form with curvature corrections.
3. **Basis for the pressure data** on faces and interfaces. Pressure jumps are $\propto\partial_\ell v$
   on faults; the pressure-layer integrals need a piecewise-linear interpolation of $[\mathcal{P}]$.
   Decide whether to reuse trapezoids, or tents.
4. **Conditioning of the eigen-decomposition.** $\mathbf A$ is not symmetric. If $\mathbf T$ is
   ill-conditioned, fall back to `expm`, a Schur form, or the ODE solver. Check that the eigenvalues
   are real and non-positive (antiplane), and what happens in plane strain.
5. **Kelvin states per side** double the Kelvin storage at interfaces. Sides with equal Kelvin parameters
   can share one state.
6. Nonlinear (power-law) rheology breaks the boundary-only reduction (the effective $\eta$ varies in
   space), so it is out of scope.

---

## 7. Test plan

Tests go in `tests/test_viscoelastic*.py`, with reference solutions added to `tests/reference_solutions.py`.
They follow the elastic suite: tapered slip $\cos^2(\pi d/2D)$ and mesh-refinement convergence.

### 7.1 Antiplane, Maxwell, closed form

- **A1:** homogeneous Maxwell half-space, locked fault. Surface $u(t) = u(0^+)$, and stress
  $\propto e^{-\alpha t}$ everywhere.
- **A2:** elastic layer $H$ over a Maxwell half-space ($\mu_1=\mu_2$), fault $D<H$. Savage and Prescott:
  $u = B(d) + \sum_m P(m,\alpha t/2)[B(2mH+d)-B(2mH-d)]$.
- **A3:** the same, fault **through** the interface into the Maxwell half-space. $u =
  \sum_{m\ge0}\frac{(\alpha t/2)^m}{m!}e^{-\alpha t/2}B(d+2mH)$. This tests an embedded fault directly.
- **A4:** Maxwell circular inclusion in an elastic matrix under remote $\sigma_{yz}$. Interior stress
  $\frac{2\mu_i}{\mu_i+\mu_m}\tau_\infty e^{-\alpha_i\mu_m t/(\mu_i+\mu_m)}$, and the single eigenvalue of
  $\mathbf A$ is $-\alpha_i\mu_m/(\mu_i+\mu_m)$.
- **A5:** a fault next to a vertical contact with a Maxwell block, including a fault on the interface
  (`dist = 0`): `contact_displacement` with $k(t) = 1-(1-k_0)e^{-\mu_1\alpha_2 t/(\mu_1+\mu_2)}$.
- Notation: $B(d) = \arctan(d/x)/\pi$, $P(m,z)$ the regularized incomplete gamma function.
  Displacements are $u(t) = u(0) + \int_0^t \dot u  dt'$ from the closed-form mode integrals.

### 7.2 Antiplane, Burgers and Laplace-transform derived

Replace each viscous modulus by $\tilde\mu(s)$ of section 1 in an elastic closed form, divide by $s$ for a
step, and invert numerically (Talbot or de Hoog, `mpmath.invertlaplace`). The Maxwell cases A2 to A5 must
be reproduced first.

- **B1, Burgers inclusion:** the two eigenvalues of $\mathbf A$ equal the roots of the quadratic of
  section 2.3 (zeros of $\mu_m+\tilde\mu_i(s)$), and the interior stress is the corresponding
  two-exponential sum.
- **B2:** `layered_image_series` with $\mu_1\ne\mu_2$ and a Burgers half-space, for $D<H$ and $D>H$.
- **B3:** a Maxwell layer over a Burgers half-space with different rates (several relaxation times, a
  non-trivial $\mathbf A$).
- **B4, Zener limit:** $1/\eta_M=0$. One eigenvalue is zero, and the relaxed state equals the elastic
  solution with the relaxed modulus $\mu\mu_K/(\mu+\mu_K)$.
- **B5, Maxwell limit:** $\mu_K\to\infty$ or $1/\eta_K=0$ recovers section 7.1.

### 7.3 Plane strain

- **C1, fault in a Maxwell full space** (no interfaces, so the whole effect is the fault-face pressure
  layer). With $K$ fixed and $\tilde\nu(s)$ from $\tilde\mu(s)$, the dislocation field is the elastic one
  with $\tilde\nu(s)$; the response is
  $1-\frac{3\mu}{3K+4\mu}e^{-\lambda t}$ with $\lambda = \frac{3K\alpha}{3K+4\mu} = g_\nu\alpha$.
- **C1-B, the same in a Burgers full space** by numerical inversion (two exponentials).
- **C1-local, the fault-face laws of section 3.6:**
  - $[p]$ sampled across the fault follows the local jump ODE for tangential slip, for Maxwell (one
    exponential) and Burgers (two), including the $\partial_\ell v$ driving;
  - the traction components on the fault follow the correspondence solution: $t_t\to0$, and
    $t_n$ relaxes toward the pressure;
  - $[\partial_n p]=0$ for planar tangential slip at all times, and nonzero for a normal-slip or curved
    fault.
- **C2, dip-slip fault in a homogeneous Maxwell half-space:** half-space dislocation formulas (Segall
  2010, ch. 3) with $\tilde\nu(s)$ at fixed $K$, closed form if affine in $1/(1-\nu)$, otherwise by
  numerical inversion.
- **C3, Maxwell and Burgers circular inclusion under remote stress:** inversion of the plane-strain
  inclusion solution with $\tilde\mu_i(s)$ and $\tilde\nu_i(s)$. `circular_inclusion_constants` must first
  be generalized to $\nu_i\neq\nu_m$.
- **C4, elastic layer over a viscous half-space, dip slip** (stretch goal): compare with a published FEM or
  semi-analytic result.

### 7.4 Internal consistency (both geometries)

- **Elastic limits:** at $t=0^+$ the result equals `bemcs.solve`, and with all dashpots off nothing evolves.
- **Relaxed limits:** antiplane Maxwell $t\to\infty$ matches the elastic solve with the viscous modulus
  $\to0$ (a compliant region acts like a free surface); Zener matches B4.
- **Side independence:** `'h'` and `'t'` rows sampled on either side give the same $\mathbf t(t)$, including
  $\dot t_n$ on faults ($\frac{\mu}{\mu_0}[\dot S_n]=-[\pi]$ holds identically).
- **Integrators agree:** eigen path, `expm`, and `solve_ivp` (BDF) give the same $\mathbf y(t)$.
  Eigenvalues are real and non-positive in antiplane.
- **Rate form against the integrated form:** for a coseismic step, time integrals of the rate-form
  tractions reproduce the time-integrated ($W$, $P$) formulation.
- **Superposition:** constant-rate creep by Duhamel equals the time integral of coseismic step responses.
- **Convergence:** antiplane second order, matching the elastic suite.
- **Plane-strain unit tests:**
  - the pressure-layer displacement against direct quadrature with the existing constant-force
    triangles, for a prescribed harmonic $\mathcal{P}$ in a closed region;
  - the DtN map on harmonic polynomials;
  - the $H$ kernels against numerical quadrature of $r_i\ln r$;
  - the local law $\dot p = (\mu/\mu_0)\hat p - g_\nu\pi$ against a direct evaluation with volume
    quadrature.

### 7.5 Coupling tests

- **State-space against Duhamel:** a prescribed slip-rate history $\mathbf v(t)$ integrated with RK45 on
  $\dot{\mathbf y} = \mathbf A\mathbf y+\mathbf G\mathbf v+\mathbf g_r$ equals the closed-form
  eigen/Duhamel solution.
- **Interseismic loading:** a locked fault above a creeping extension in an elastic layer over a Maxwell
  half-space (Savage and Prescott, interseismic): kinematic loading by `'s'` elements with prescribed
  rate, against the analytic surface velocity.
- **Restart:** saving and restoring $\mathbf y$ (including the Kelvin states) continues a run identically.

---

## References

- Savage and Prescott (1978), *JGR* 83, 3369: elastic layer over a Maxwell half-space.
- Nur and Mavko (1974), *Science* 183, 204.
- Rybicki (1971), *BSSA* 61, 79: layered and contact image solutions (already used in
  `tests/reference_solutions.py`).
- Segall (2010), *Earthquake and Volcano Deformation*, ch. 3, 5 and 6.
- Barbot and Fialko (2010), *GJI* 182, 1124; Lambert and Barbot (2016), *GRL* 43, 10142: eigenstrain
  approaches that need volume meshes, for contrast.
