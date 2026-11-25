# Viscoelasticity (linear Maxwell) as a boundary-only problem

*Branch `viscoelasticity`. Working note: governing equations, implementation plan on top of the
heterogeneous-elastic solver (one-ring hat slip, trapezoidal force), and the test plan.*

Status: design. Nothing here is implemented yet. Symbolic checks of the closed forms quoted in
sections 3.4 and 6 were done with sympy (2026-11-25).

---

## 0. Idea in one paragraph

The heterogeneous-elastic solver already replaces a jump in $\mu$ across an interface by a line force
in a homogeneous reference medium ($\mu_0$, $\nu$), solved for together with slip
(`docs/theory/heterogeneous.md`). For a linear Maxwell body the viscous strain is one more source of
incompatibility. In **antiplane** strain, a uniform Maxwell region produces no source in its interior,
so the viscous effect is again a line force on interfaces where the fluidity $1/\eta$ jumps. That line
force is proportional to the **time-integrated traction** on the interface. In **plane strain**, the
deviatoric flow also produces a volume force $\propto\nabla P$ ($P$ = time-integrated pressure). $P$ is
harmonic in each uniform region, so Green's identities turn it into boundary terms. Either way, the
unknowns of the time problem live only on boundaries. The problem becomes a linear ODE
$\dot{\mathbf y} = \mathbf A\mathbf y + \mathbf g(t)$ for a few history variables per collocation
point. Each evaluation of the right-hand side is one solve with the **existing, unchanged** elastic
system matrix.

---

## 1. Constitutive law and notation

Maxwell body, deviatoric flow only:

$$
\dot\sigma_{ij} = C_{ijkl}\dot\varepsilon_{kl} - \frac{\mu}{\eta} \big(\sigma_{ij} - p \delta_{ij}\big),
\qquad p = \tfrac13\sigma_{kk},
$$

equivalently $\varepsilon = \varepsilon^e + \varepsilon^v$ with
$\dot\varepsilon^v = \operatorname{dev}\sigma/(2\eta)$.

- $\mu_\pm$, $\eta_\pm$: shear modulus and viscosity on the $-n$ and $+n$ side of an element (same convention as `mu_minus`, `mu_plus`).
- $\alpha = \mu/\eta$: inverse Maxwell time; $\eta=\infty$ (elastic) gives $\alpha=0$.
- $\mu_0$, $\nu$: reference medium of the kernels; $\nu$ is uniform (as in the elastic solver).
- $\mathbf{W}(\mathbf{x},t)$: time-integrated (true) traction.
- $P(\mathbf{x},t)$: time-integrated pressure (plane strain only).
- $\mathbf{D}$: time-integrated deviatoric stress, so that $2\mu_0\varepsilon^v = (\mu_0/\eta)\mathbf{D}$.

$$
\mathbf{W}(\mathbf{x},t) = \int_0^t \boldsymbol{\sigma}\cdot\mathbf{n} dt', \qquad
P(\mathbf{x},t) = \int_0^t p dt', \qquad
\mathbf{D} = \int_0^t \operatorname{dev}\boldsymbol{\sigma} dt'.
$$

Antiplane uses $\sigma = (\sigma_{xz},\sigma_{yz}) = \mu\nabla u_z$ (the code's convention), so
$\nabla\dot u = \dot\sigma/\mu + \sigma/\eta$ and the relaxation time is $\eta/\mu = 1/\alpha$.
Savage & Prescott (1978) write their results in $\tau = \mu t/(2\eta) = \alpha t/2$; section 6
confirms that this matches.

Viscous strain is deviatoric, so $\operatorname{tr}\varepsilon = \operatorname{tr}\varepsilon^e$. With
uniform $\nu$, the **true pressure is the scaled reference pressure exactly**:
$p = (\mu/\mu_0) p_{\mathrm{ref}}$. The **true traction gets a viscous correction**, derived below.

---

## 2. Antiplane

### 2.1 Equivalent force

Equation of the reference medium: $\mu_0\nabla^2u + f = 0$. Integrate $\nabla\dot u = \dot\sigma/\mu + \sigma/\eta$
in time from a stress-free viscous state ($\varepsilon^v(0)=0$):

$$
\mu_0\nabla u = \frac{\mu_0}{\mu}\sigma + \frac{\mu_0}{\eta} \boldsymbol{\mathcal W},
\qquad \boldsymbol{\mathcal W} = \int_0^t\sigma dt' .
$$

Take the divergence. $\nabla\cdot\sigma = 0$ holds everywhere, including across faults, because
traction is continuous. So $\nabla\cdot\boldsymbol{\mathcal W} = 0$ as well, and only the jumps of
$1/\mu$ and $1/\eta$ are left:

$$
f = \mu_0 t\left(\frac1{\mu_-}-\frac1{\mu_+}\right)\delta_I
  + \mu_0 W\left(\frac1{\eta_-}-\frac1{\eta_+}\right)\delta_I ,
\qquad t = \sigma\cdot n,\quad W = \boldsymbol{\mathcal W}\cdot n .
$$

The first term is the existing interface force. The second is new and needs only $W$ on the interface.

**Faults in a uniform Maxwell region produce no source.** $\eta$ does not jump across a fault, so
an embedded fault enters only through its slip, as in the elastic problem. Check: in a homogeneous
Maxwell half-space with a locked fault, $f\equiv0$. The surface displacement therefore stays at its
coseismic value, while stress relaxes everywhere as $e^{-\alpha t}$. This is the correspondence-principle
answer, because antiplane dislocation displacements do not depend on $\mu$.

### 2.2 True traction and the rows

The code samples the reference traction $(S_{\mathrm{ref}}\cdot n)_\sigma$ on side $\sigma\in\{-,+\}$.
The true traction there is

$$
t = \frac{\mu_\sigma}{\mu_0}(S_{\mathrm{ref}}\cdot n)_\sigma - \alpha_\sigma W .
$$

Substituting into the force above gives the interface (`'h'`) row. Its left-hand side is today's,
and $W$ appears as a new right-hand-side term:

$$
f - \kappa_\sigma (S_{\mathrm{ref}}\cdot n)_\sigma = c_\sigma W,
\qquad
\kappa_\sigma = \texttt{interface\_kappa},\quad
c_- = \frac{\mu_0}{\mu_+}(\alpha_- - \alpha_+),\quad
c_+ = \frac{\mu_0}{\mu_-}(\alpha_- - \alpha_+).
$$

The row is still independent of the sampled side. With $S_+ = S_- - f$, the $+$ row reduces to the
$-$ row: $\kappa_+/(1+\kappa_+) = \kappa_-$ and $c_+/(1+\kappa_+) = c_-$.

Traction (`'t'`) rows on elements whose sampled side is viscoelastic (a frictional or free fault
inside a Maxwell region, a loaded boundary) become

$$
\frac{\mu_\sigma}{\mu_0}(S_{\mathrm{ref}}\cdot n)_\sigma = t_{\mathrm{given}} + \alpha_\sigma W .
$$

A traction-free surface has $t\equiv0$, so $W\equiv0$ and the row is unchanged. Slip (`'s'`) and
displacement (`'u'`) rows are unchanged.

### 2.3 The ODE

Stack $W$ at every collocation point that touches viscoelastic material and whose traction is not
prescribed to be zero. In the system below, $\mathbf M$ is today's matrix, $\mathbf b(t)$ today's
right-hand side (slip, tractions, …), and $\mathbf B$ puts $c_\sigma W$ and $\alpha_\sigma W$ into
the rows above:

$$
\mathbf M\mathbf x = \mathbf b(t) + \mathbf B\mathbf W,
\qquad
\dot{\mathbf W} = \mathbf t = \mathbf C\mathbf x - \mathbf D_\alpha\mathbf W .
$$

Here $\mathbf C$ samples $(\mu_\sigma/\mu_0)S_{\mathrm{ref}}\cdot n$ at the state points and
$\mathbf D_\alpha = \operatorname{diag}(\alpha_\sigma)$. Hence

$$
\boxed{\ \dot{\mathbf W} = \underbrace{\big(\mathbf C\mathbf M^{-1}\mathbf B - \mathbf D_\alpha\big)}_{\mathbf A}\mathbf W
 + \mathbf C\mathbf M^{-1}\mathbf b(t),\qquad \mathbf W(0)=0\ }
$$

- **Coseismic step** $\mathbf b = \mathbf b_0 H(t)$: $\mathbf W(t) = \mathbf T \Lambda^{-1}(e^{\Lambda t}-1)\mathbf T^{-1}\mathbf C\mathbf M^{-1}\mathbf b_0$
  with $\mathbf A = \mathbf T\Lambda\mathbf T^{-1}$. Displacements and fault/interface tractions are
  then affine in $\mathbf W$.
- **Kinematic slip history** (afterslip, steady creep below a locking depth): Duhamel integral of the
  above. Constant rate is closed form; piecewise-linear histories are closed form per piece.
- **Fault governed by a fault law** (linear viscous $v=\tau/\eta_F$, rate-and-state): the fault slip
  joins the state vector. A linear law keeps the system linear.
- **Time-dependent Green's functions.** Applying the step solution to each slip hat gives
  $\mathbf G(t) = \mathbf G_{\mathrm{el}} + \sum_k \mathbf G_k(1-e^{\lambda_k t})$. This is a direct
  extension of `get_greens_functions` and is what cycle models and postseismic inversions need.

This is the matrix of the original note, $\mu\alpha_{,n}\mathcal G_{,n}\mathcal L_e^{-1} - \alpha I$,
written in the code's sharp-interface and reference-medium language. The inverse of the elastic
operator $\mathcal L_e$ is never formed: we factor $\mathbf M$ once and reuse the factorisation.

**Stress at interior points** of a Maxwell region relaxes pointwise:
$\dot\sigma = (\mu/\mu_0)\dot S_{\mathrm{ref}} - \alpha\sigma$, with $S_{\mathrm{ref}}(t)$ affine in
$\mathbf W(t)$. Every observation point therefore has a closed-form mode sum and needs no extra state.

### 2.4 Geometry notes

- A fault crossing an $\eta$-interface splits the interface at the crossing (`fault_crossings`), as in
  the elastic problem. A slip hat that straddles the crossing samples $\alpha$ on whichever side its
  collocation point lies. Warn if a hat centre falls on the crossing (warn, don't fix).
- With clamped tips (slip and slip gradient zero), tip stresses are bounded, so tip relaxation needs
  nothing special.

---

## 3. Plane strain

### 3.1 Equivalent force

The reference medium satisfies $\nabla\cdot(\mathbf C_0:\varepsilon) + \mathbf f = 0$. With uniform $\nu$,
$\mathbf C_0:\varepsilon = (\mu_0/\mu)\sigma + (\mu_0/\eta)\mathbf D$. Since $\nabla\cdot\sigma = 0$,
$\nabla\cdot\mathbf D = -\nabla P$. So

$$
\mathbf f =
\underbrace{\frac{\mu_0}{\eta}\nabla P}_{\text{volume, in each Maxwell region}}
+ \underbrace{\Big[\mu_0 \mathbf t\Big(\frac1{\mu_-}-\frac1{\mu_+}\Big)
+ \mu_0\Big(\frac{\mathbf W - P_-\mathbf n}{\eta_-} - \frac{\mathbf W - P_+\mathbf n}{\eta_+}\Big)\Big]\delta_I}_{\text{interface}} .
$$

The traction (and so $\mathbf W$) is continuous, while $p$ (and so $P$) jumps across interfaces
and faults, so $P_\pm$ are one-sided values. The true traction on side $\sigma$ is

$$
\mathbf t = \frac{\mu_\sigma}{\mu_0}(S_{\mathrm{ref}}\cdot n)_\sigma - \alpha_\sigma(\mathbf W - P_\sigma\mathbf n),
$$

and the 'h' and 't' rows change as in antiplane (same algebra, now with 2-vectors).

### 3.2 $P$ is harmonic in each uniform region

The appendix argument of the original note uses the elastic Navier equation, which no longer holds
once viscous strain builds up. A corrected argument:

1. Take the divergence of the rate equations: $(\lambda+2\mu)\nabla^2\dot\theta = -\frac{\mu}{\eta}\nabla^2 p$.
2. Use $p = K\theta$: $\nabla^2\dot p = -\frac{\mu K}{\eta(\lambda+2\mu)}\nabla^2p$.
3. So $\nabla^2p$ decays exponentially from its initial value. It is zero for all time if the initial
   state is elastic and has no body force, and then $\nabla^2P = 0$ as well.

This holds in $V_r\setminus F$ for each uniform Maxwell region $V_r$. Across a fault $F$ inside
$V_r$, $P$ jumps.

### 3.3 Volume term to boundary terms

Let $r = \mathbf x - \mathbf x'$ and take the 2-D Kelvin solution $G_{ik}$ of the reference medium.
Define

$$
H_i(\mathbf x,\mathbf x') = \frac{1-2\nu}{8\pi\mu_0(1-\nu)}  r_i\ln r,
\qquad \nabla'^2 H_i = \frac{\partial G_{ik}}{\partial x'_k}
$$

(checked symbolically, using $\nabla^2(x_i\ln r) = 2x_i/r^2$; this is the 2-D analogue of the note's
$r_{,i}$ kernel). Integrate by parts, apply Green's second identity, and use $\nabla'^2P=0$. Then

$$
u_i^{\mathrm{vol}}(\mathbf x) = \sum_r \frac{\mu_0}{\eta_r}\oint_{\partial V_r}
\Big[ P \big(G_{ik}n'_k - \partial_{n'}H_i\big) + H_i \partial_{n'}P \Big]  dS' ,
$$

with $n'$ the outward normal of $V_r$. The boundary $\partial V_r$ includes the interfaces, free
surfaces, and **both faces of every fault inside $V_r$**. On a fault the two faces combine to
$-\int_F\big([P](G_{ik}n_k-\partial_nH_i) + H_i[\partial_nP]\big)dS$. The fault therefore *is* a
viscoelastic source in plane strain (unlike antiplane), with strength $\propto [P]$.

- The $P G_{ik}n'_k$ term is a normal line force: the **existing trapezoid force kernels**.
- $H_i$ and $\partial_nH_i$ integrated against linear elements are new kernels. They are elementary
  ($r_i\ln r$ and its derivative on a segment), so closed forms are cheap to derive and add to level 3.
- $\partial_nP$ comes from $P$ through a Laplace Dirichlet-to-Neumann (DtN) map on $\partial V_r$.
  The Laplace kernels already exist: the **antiplane force kernel is the Laplace single layer** and the
  **antiplane slip kernel is the Laplace double layer** (it jumps by the "slip"). Alternative that
  avoids inverting the DtN map: represent $P = \mathcal S_r[\omega_r]$ (a single layer on
  $\partial V_r$), make $\omega_r$ the state, and get
  $\partial_nP = \pm\tfrac12\omega + \mathcal K'\omega$ directly.

### 3.4 Fault jump conditions (a built-in check)

For tangential slip $s(\ell)$ on a planar fault, traction continuity and $[\mathbf u]=s \mathbf e_t$ give
the elastic jumps (at $t=0^+$):

$$
[\sigma_{tt}] = \frac{2\mu}{1-\nu}\partial_\ell s,\quad
[\sigma_{zz}] = \frac{2\lambda\mu}{\lambda+2\mu}\partial_\ell s,\quad
[p] = \frac{2\mu(1+\nu)}{3(1-\nu)}\partial_\ell s,\quad [\partial_n p] = 0 .
$$

The pressure of a dislocation is a pure Laplace double layer; curved faults add curvature terms. After
$t=0$ the jumps relax, even on a locked fault. With $b=[\dot\varepsilon_{nn}]$ and $v$ the slip rate:

$$
\tfrac{d}{dt}[\sigma_{tt}] = (\lambda+2\mu)\partial_\ell v + \lambda b - \tfrac{\mu}{\eta}([\sigma_{tt}]-[p]),\quad
\tfrac{d}{dt}[\sigma_{zz}] = \lambda(\partial_\ell v + b) - \tfrac{\mu}{\eta}([\sigma_{zz}]-[p]),\quad
b = -\frac{\lambda\partial_\ell v + \frac{\mu}{\eta}[p]}{\lambda+2\mu},
$$

and $[p]\to0$ as deviatoric stress dies out. The representation above produces these jumps
automatically. The fault states are just $P_\pm$ sampled on the two faces. This local ODE is therefore
a **test**, not an extra equation.

### 3.5 States and ODE

The plane-strain states, sampled at collocation points on $\partial V_r$:

- $\mathbf W$ (2 components): on interfaces and on faults and boundaries with nonzero traction;
- $P_\sigma$ (or $\omega_r$): one per side that touches Maxwell material, including free surfaces.

They evolve as $\dot{\mathbf W} = \mathbf t$ and $\dot P_\sigma = (\mu_\sigma/\mu_0)p_{\mathrm{ref},\sigma}$.
Both right-hand sides are linear in $\mathbf x$, so the system closes exactly as in section 2.3:
$\dot{\mathbf y} = \mathbf A\mathbf y + \mathbf C\mathbf M^{-1}\mathbf b(t)$. The volume terms add
columns to $\mathbf B$; the matrix $\mathbf M$ is unchanged.

$\sigma_{zz}$ never has to be tracked: it is $3p - \sigma_{xx} - \sigma_{yy}$.

---

## 4. Implementation plan (one-ring hat slip, trapezoidal force)

**Unknowns and collocation stay as they are.** Slip amplitudes sit on one-ring hats (one per non-tip
slip element and component). Force amplitudes sit on trapezoids (one per 3-element patch and
component), with the 'h' rows at the trapezoid collocation points. The states use **the same points**:
$W$ at every trapezoid 'h' point and every hat 't' point that touches Maxwell material. The number of
states therefore equals the number of affected rows, and $\mathbf B$ is a sparse selection matrix with
weights $c_\sigma$ and $\alpha_\sigma$. The trapezoid amplitude interpolates the viscous line force
between collocation points exactly as it does the elastic one.

**Mesh input.** Add the optional columns `eta_minus` and `eta_plus` (default `inf`) to the mesh CSV
and `Mesh(...)`, next to `mu_minus` and `mu_plus`. Checks (warnings, never rewrites):

- inconsistent $\eta$ on the two sides of a shared node;
- a hat centre on an $\eta$-crossing;
- in plane strain, a Maxwell region whose boundary is not closed by the mesh (DtN needs it).

**Code layout (proposal).**

```
bemcs/viscoelastic.py
    relaxation_system(mesh, *, geometry, mu, nu=None)  -> RelaxationSystem(M_lu, B, C, D_alpha, state_info)
    RelaxationSystem.modes()                           -> eigenvalues/vectors of A (cached)
    RelaxationSystem.step_response(b0, t)              -> W(t), x(t)   (closed form)
    RelaxationSystem.kinematic(b_of_t, t)              -> Duhamel, piecewise-linear histories
    RelaxationSystem.fields(x_obs, y_obs, t)           -> displacements, true stresses (pointwise relaxation)
bemcs.get_greens_functions(..., times=...)             -> time-dependent GFs as mode sums
bemAssembly: element kernels H, dH/dn for linear elements (plane strain, stage 2)
```

`RelaxationSystem.solve_ode(...)` wraps `scipy.integrate.solve_ivp` (BDF). It is the cross-check
for the eigen path, and the route to later nonlinear (power-law) rheology.

**Stages.**

1. Antiplane: interfaces, faults inside Maxwell regions, and fault 't' rows with $W$. Tests 6.1, 6.2
   and 6.4.
2. Plane strain, Maxwell regions without faults inside them: $P$ states, H kernels, DtN. Tests 6.3
   (C3) and 6.4.
3. Plane strain with faults inside Maxwell regions: fault-face $P_\pm$. Tests C1 and C2.
4. Time-dependent Green's functions, kinematic histories, and a linear-viscous fault law.

---

## 5. Open questions

1. **Unbounded regions in plane strain.** For a Maxwell half-space, the 2-D $G\sim\ln r$ against
   $P\sim$ dipole decay should leave no contribution from infinity, but this needs checking. The 2-D
   Laplace single layer on an open (truncated) boundary also has the usual log and total-charge issue.
2. **Eigen-decomposition conditioning.** $\mathbf A$ is not symmetric. If $\mathbf T$ is
   ill-conditioned, fall back to `expm`, a Schur form, or the ODE solver. Antiplane eigenvalues are
   expected to be real and negative; check this.
3. Where $\eta$ jumps but $\mu$ does not, 'h' rows exist only because of $W$. Their left-hand side
   has $\kappa_\sigma = 0$, so the row is simply $f = c_\sigma W$. That is fine, but these elements
   must be labelled 'h' even when $\mu_-=\mu_+$; check that `Mesh` accepts this.
4. Nonlinear (power-law) rheology breaks the interface-only reduction (the effective $\eta$ varies in
   space), so it is out of scope.

---

## 6. Test plan

The tests go in `tests/test_viscoelastic*.py`, with reference solutions added to
`tests/reference_solutions.py`. They follow the elastic suite's style: tapered slip
$\cos^2(\pi d/2D)$ and mesh-refinement convergence rates.

### 6.1 Antiplane, closed-form

| id | setup | reference |
| --- | --- | --- |
| A1 | homogeneous Maxwell half-space, locked fault | surface $u(t) = u(0^+)$; stress $\propto e^{-\alpha t}$ everywhere |
| A2 | elastic layer $H$ over Maxwell half-space, $\mu_1=\mu_2$, fault $D<H$ | Savage & Prescott: $u = B(d) + \sum_m P(m,\alpha t/2) [B(2mH+d)-B(2mH-d)]$ |
| A3 | same, fault **through** the interface into the Maxwell half-space ($d>H$ part) | $u = \sum_{m\ge0} \dfrac{(\alpha t/2)^m}{m!}e^{-\alpha t/2} B(d+2mH)$ |
| A4 | Maxwell circular inclusion ($\mu_i$, $\alpha_i$) in an elastic matrix under remote $\sigma_{yz}=\tau_\infty$ | uniform interior stress $\dfrac{2\mu_i}{\mu_i+\mu_m}\tau_\infty e^{-\alpha_i\mu_m t/(\mu_i+\mu_m)}$; the slowest eigenvalue of $\mathbf A$ is $-\alpha_i\mu_m/(\mu_i+\mu_m)$ |
| A5 | fault in an elastic block next to a vertical contact with a Maxwell block (and `dist = 0`: fault **on** the elastic/Maxwell interface) | `contact_displacement` with $k(t) = 1-(1-k_0) e^{-\mu_1\alpha_2 t/(\mu_1+\mu_2)}$ |

- $B(d) = \arctan(d/x)/\pi$, as in `layered_image_series`, and
  $P(m,z) = 1-e^{-z}\sum_{j<m}z^j/j!$ (regularised incomplete gamma).
- **A2 and A3** are the correspondence principle applied to Rybicki's series with
  $k(s) = \alpha/(2s+\alpha)$. They are inverted in closed form (sympy-checked). Their long-time limits
  are physical checks: in A2 the image sum tends to its fully relaxed value; in A3 the deep part tends
  to a rigid-block offset $\operatorname{sign}(x)/2$ per unit slip.
- **A3 tests the embedded-fault case directly.**

### 6.2 Antiplane, Laplace-transform derived

Replace each Maxwell modulus by $\mu s/(s+\alpha)$ in an elastic closed form, divide by $s$ for a
step, and invert numerically (Talbot or de Hoog, `mpmath.invertlaplace`). At the parameters where A2
to A5 apply, the numerical inversion must reproduce them first.

- **B1:** `layered_image_series` with $\mu_1\ne\mu_2$ and a Maxwell half-space, for $D<H$ and $D>H$.
- **B2:** a Maxwell layer over an elastic half-space, with $\mu_1(s)$.
- **B3:** a Maxwell layer over a Maxwell half-space with two different $\alpha$. This gives two
  relaxation spectra and a non-trivial $\mathbf A$.

### 6.3 Plane strain

| id | setup | reference |
| --- | --- | --- |
| C1 | edge dislocation or tapered fault in a **Maxwell full space** (no interfaces: the whole effect is fault-face $[P]$) | closed form. All $\nu$ dependence is through $g=1/(2(1-\nu))$, and $g(t) = 1-\dfrac{3\mu}{3K+4\mu}e^{-\lambda t}$ with $\lambda=\dfrac{3K\alpha}{3K+4\mu}$ |
| C2 | buried or surface-breaking dip-slip fault in a homogeneous Maxwell half-space | half-space dislocation formulas (Segall 2010 ch. 3) with $\nu(s)$ at fixed $K$. Closed form like C1 if they are affine in $1/(1-\nu)$, otherwise numerical inversion |
| C3 | Maxwell circular inclusion under remote stress | Laplace inversion of the plane-strain inclusion solution with $\mu_i(s)$ and $\nu_i(s)$. `circular_inclusion_constants` must first be generalised to $\nu_i\ne\nu_m$ |
| C4 | elastic layer over Maxwell half-space, dip slip (stretch goal) | no simple closed form; compare to a published FEM or semi-analytic result |

- **C1 and C2** test the fault-face terms and the jump relaxation of section 3.4. Sample $[P]$ across
  the fault and compare with the local jump ODE.
- **C3** checks interface $P_\pm$ and the H and DtN kernels without faults.

### 6.4 Internal consistency (both geometries)

- **Elastic limits:** at $t=0^+$ the result equals today's `bemcs.solve`; with $\eta\to\infty$ nothing
  evolves.
- **Relaxed limit (antiplane):** $t\to\infty$ matches the elastic solve with $\mu_{\mathrm{Maxwell}}\to0$
  (a compliant region acts like a free surface; the existing compliant-inclusion test is the template).
- **Side independence:** 'h' and 't' rows sampled on either side give the same $\mathbf W(t)$.
- **Integrators agree:** eigen path, `expm`, and `solve_ivp` (BDF) give the same $\mathbf W(t)$.
  $\operatorname{Re}\lambda_k<0$ for every mode, and antiplane modes are real.
- **Superposition:** constant-rate creep via Duhamel equals the time integral of coseismic step
  responses.
- **Convergence:** under mesh refinement, the antiplane rate matches the elastic suite (second order).
- **Unit tests of the plane-strain pieces:**
  - the Green's-identity conversion against direct volume quadrature with the existing constant-force
    triangles, for a prescribed harmonic $P$ in a closed region;
  - the DtN map on harmonic polynomials;
  - the H kernels against numerical quadrature of $r_i\ln r$.

---

## References

- Savage & Prescott (1978), *JGR* 83, 3369: elastic layer over a Maxwell half-space.
- Nur & Mavko (1974), *Science* 183, 204.
- Rybicki (1971), *BSSA* 61, 79: layered and contact image solutions (already used in
  `tests/reference_solutions.py`).
- Segall (2010), *Earthquake and Volcano Deformation*, ch. 3, 5 and 6.
- Barbot & Fialko (2010), *GJI* 182, 1124; Lambert & Barbot (2016), *GRL* 43, 10142: eigenstrain
  approaches that need volume meshes, for contrast.
