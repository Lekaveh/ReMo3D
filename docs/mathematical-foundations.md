# Mathematical Foundations

## 2.1 Governing PDE

ReMo3D solves the stationary current-flow equation for electric potential
`u(x)` in a heterogeneous, isotropic conductivity field `sigma(x)`:

$$
\begin{aligned}
-\nabla \cdot (\sigma \nabla u) &= \sum_k I_k \, \delta(\mathbf{x} - \mathbf{x}_k) && \text{in } \Omega \\
u &= 0 && \text{on } \Gamma_D \\
\sigma \nabla u \cdot \mathbf{n} &= 0 && \text{on } \Gamma_N
\end{aligned}
$$

```text
-div(sigma grad u) = sum_k I_k delta(x - x_k)   in Omega
u = 0                                            on Gamma_D
sigma grad u . n = 0                             on Gamma_N
```

with:

- `Omega`: computational domain
- `Gamma_D`: outer circular or spherical boundary
- `Gamma_N`: remaining boundaries, including symmetry boundaries
- `I_k`: source strengths from `source_terms`
- `x_k`: current-electrode locations on the borehole axis

The code uses conductivity rather than resistivity:

$$
\sigma = \frac{1}{\rho}
$$

```text
sigma = 1 / rho
```

where `rho` is the resistivity supplied in the model files.

## 2.2 2D Axisymmetric Formulation

When `dip == 0`, the geometry is treated as rotationally symmetric around the
borehole axis. The 3D PDE is reduced to a 2D meridional cross-section with:

- `x`: radial distance from the axis
- `y`: depth along the axis

Under the assumption of no azimuthal dependence,

$$
dV = r \, dr \, d\theta \, dz
$$

```text
dV = r dr dtheta dz
```

so integrating over `theta in [0, 2pi]` gives the weak form:

$$
\int_{\Omega_{2D}} 2 \pi r \, \sigma \, \nabla u \cdot \nabla v \, dr \, dz
= \sum_k I_k \, v(0, z_k)
$$

```text
Integral_Omega2D 2 pi r sigma grad u . grad v dr dz = sum_k I_k v(0, z_k)
```

That is exactly what appears in the solver:

```python
a += 2*np.pi*ngs.grad(u)*ngs.grad(v)*ngs.x*sigma*ngs.dx
```

`ngs.x` is the radial coordinate `r`, and the `2*pi` factor is the integrated
azimuthal measure.

## 2.3 Full 3D Formulation

When `dip != 0`, cylindrical symmetry is lost and the code switches to a 3D
half-space mesh. The weak form becomes:

$$
\int_{\Omega_{3D}} \sigma \, \nabla u \cdot \nabla v \, dV = \sum_k I_k \, v(\mathbf{x}_k)
$$

```text
Integral_Omega3D sigma grad u . grad v dV = sum_k I_k v(x_k)
```

implemented as:

```python
a += ngs.grad(u)*ngs.grad(v)*sigma*ngs.dx
```

## 2.4 Point Source Model

`AddPointSource` injects a discrete Dirac-type source into the linear form by:

1. finding the mesh point for the requested source location
2. locating the containing finite element
3. evaluating the local basis functions there
4. adding `fac * phi_j(x_k)` to each element DOF

In code:

```python
ei = ngs.ElementId(ngs.VOL, mp.nr)
fel = spc.GetFE(ei)
dnums = spc.GetDofNrs(ei)
shape = fel.CalcShape(*mp.pnt)
for d, s in zip(dnums, shape):
    f.vec[d] += fac*s
```

Mathematically this approximates the load functional:

$$
\ell(v) = \sum_k I_k \, v(\mathbf{x}_k)
$$

```text
ell(v) = sum_k I_k v(x_k)
```

Physically, each point source is a current injection or extraction electrode.

## 2.5 Boundary Conditions

### Dirichlet boundary

The outer boundary of the local domain is fixed to zero potential:

$$
u = 0 \quad \text{on } \Gamma_D
$$

```text
u = 0 on Gamma_D
```

This approximates the vanishing-potential condition at infinity by placing the
boundary far from the electrodes and conductivity contrasts.

### Neumann boundary

All remaining boundaries are natural boundaries in the weak form:

$$
\sigma \nabla u \cdot \mathbf{n} = 0 \quad \text{on } \Gamma_N
$$

```text
sigma grad u . n = 0 on Gamma_N
```

This is used for:

- the non-Dirichlet 2D boundaries
- the symmetry boundary of the 3D half-space model
- any mesh entity not tagged as `dirichlet_boundary`

Internal material interfaces are not explicit BCs. They are handled through the
piecewise-constant conductivity field `sigma`.

## 2.6 Geometric Factors

The code uses the point-source potential in a homogeneous medium:

$$
V(r) = \frac{\rho}{4 \pi r}
$$

```text
V(r) = rho / (4 pi r)
```

### Single current electrode, two potential electrodes

If the active current electrode is `A`, then:

$$
V(M) - V(N) = \frac{\rho}{4 \pi} \left(\frac{1}{AM} - \frac{1}{AN}\right)
$$

```text
V(M) - V(N) = rho / (4 pi) * (1/AM - 1/AN)
```

so:

$$
\rho = \frac{4 \pi \, AM \, AN}{AN - AM} \, (V(M) - V(N))
$$

```text
rho = 4 pi AM AN / (AN - AM) * (V(M) - V(N))
```

This is the formula used when `B` is missing. The `A`/`B`-symmetric case is:

$$
\rho = \frac{4 \pi \, BM \, BN}{BN - BM} \, (V(M) - V(N))
$$

```text
rho = 4 pi BM BN / (BN - BM) * (V(M) - V(N))
```

### Two current electrodes, one potential electrode

For a measurement at `N` with current electrodes `A` and `B`:

$$
V(N) = \frac{\rho}{4 \pi} \left(\frac{1}{AN} - \frac{1}{BN}\right)
$$

```text
V(N) = rho / (4 pi) * (1/AN - 1/BN)
```

so:

$$
\rho = \frac{4 \pi \, AN \, BN}{AN - BN} \, V(N)
$$

```text
rho = 4 pi AN BN / (AN - BN) * V(N)
```

Likewise, for a single measurement electrode `M`:

$$
\rho = \frac{4 \pi \, AM \, BM}{BM - AM} \, V(M)
$$

```text
rho = 4 pi AM BM / (BM - AM) * V(M)
```

### Mapping to `_set_tool_parameters`

The code branches by the missing electrode:

- missing `A`: `4*pi*BM*BN/(BN-BM)`
- missing `B`: `4*pi*AM*AN/(AN-AM)`
- missing `M`: `4*pi*AN*BN/(AN-BN)`
- missing `N`: `4*pi*AM*BM/(BM-AM)`

`abs(...)` is applied because the final quantity of interest is the magnitude of
apparent resistivity.

## 2.7 Apparent Resistivity Formula

For every finished solve, the worker evaluates the potential at the measuring
locations and multiplies by the geometric factor:

$$
\begin{aligned}
\rho_a &= K \, |u(M) - u(N)| && \text{for two potential electrodes} \\
\rho_a &= K \, |u(M)| && \text{for one potential electrode}
\end{aligned}
$$

```text
rho_a = K * |u(M) - u(N)|     for two potential electrodes
rho_a = K * |u(M)|            for one potential electrode
```

The implementation mirrors that directly.

### Why the 3D worker divides by two

The 3D mesh is only a half-sphere with a symmetry boundary. The worker divides
the measured response by two because the solve is performed on the mirrored
half-space model while the geometric-factor formulas are written for the full
space expression used by the codebase.

## 2.8 Static Condensation

When `condense=True`, NGSolve eliminates element-internal unknowns before the
global conjugate-gradient solve.

With a block partition:

$$
\begin{bmatrix}
A_{EE} & A_{EI} \\
A_{IE} & A_{II}
\end{bmatrix}
\begin{bmatrix}
u_E \\
u_I
\end{bmatrix}
=
\begin{bmatrix}
f_E \\
f_I
\end{bmatrix}
$$

```text
[A_EE  A_EI] [u_E] = [f_E]
[A_IE  A_II] [u_I]   [f_I]
```

the reduced system is the Schur complement in the exposed unknowns:

$$
\begin{aligned}
S u_E &= f_E - A_{EI} A_{II}^{-1} f_I \\
S &= A_{EE} - A_{EI} A_{II}^{-1} A_{IE}
\end{aligned}
$$

```text
S u_E = f_E - A_EI A_II^-1 f_I
S = A_EE - A_EI A_II^-1 A_IE
```

The code reconstructs the eliminated contributions with:

- `harmonic_extension_trans`
- `harmonic_extension`
- `inner_solve`

using:

```python
f.vec.data += a.harmonic_extension_trans * f.vec
gfu.vec.data = inv * f.vec
gfu.vec.data += a.harmonic_extension * gfu.vec
gfu.vec.data += a.inner_solve * f.vec
```

This ordering is centralized in `ngsolve_functions._condensed_solve` and reused
by both CPU and GPU solver paths. The Task 0 benchmark harness compares this
condensed sequence against `condense=False` baselines before later performance
optimizations are accepted.

## 2.9 Preconditioners

The solve path is:

```python
c = ngs.Preconditioner(a, preconditioner)
inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
```

Allowed values are:

- `"local"`: cheaper, local preconditioning
- `"multigrid"`: hierarchy-based preconditioning and the default in this repo

For the elliptic problems solved here, `"multigrid"` is the more performance-
oriented default for moderate and large meshes.

## 2.10 Simulation Domain Geometry

The outer domain is intentionally artificial:

- 2D path: circular cross-section
- 3D path: half-sphere

The purpose is to truncate an unbounded resistivity problem into a finite domain
where zero Dirichlet data at the outer boundary remain a good approximation.

### Why radius matters

- larger `domain_radius`: boundary farther away, better approximation, higher
  meshing and solve cost
- smaller `domain_radius`: faster but more sensitive to boundary effects

The master warns when any electrode lies beyond `0.75 * domain_radius` and
aborts when any electrode is outside the domain at all.
## See Also
- [solver.md](solver.md#62-solvebvp-cpu-version): where the documented weak forms are assembled in code.
- [model-api.md](model-api.md#34-_set_tool_parameters): how geometric factors and depth shifts are derived from tool strings.
- [mesh-generation.md](mesh-generation.md#47-mesh-size-control-strategy): geometry and refinement choices that affect the PDE solve.
