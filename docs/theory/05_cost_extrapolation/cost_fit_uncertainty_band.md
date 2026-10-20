# Uncertainty band of the shifted power law cost fit

This note explains how the ±1σ error band drawn around the shifted power
law curve on the "Computational cost vs bare mass" plots is computed, why
the fit parameters' individual errors are not enough to compute it, and
how the pipeline keeps the information needed to do it correctly.

Numbers quoted as examples come from the Zolotarev, Wilson kernel, n = 4,
PCAC branch of the `Zolotarev_several_config_varying_n_adjusted_dMin` data
set.

Relevant code:

| Stage | File | What it does |
| --- | --- | --- |
| Analysis | `core/src/analysis/cost_extrapolation/_cost_extrapolation_core.py`, `fit_and_extrapolate_cost` | Fits the cost, extrapolates it, exports parameters **and their correlations** |
| Plotting | `core/src/analysis/cost_extrapolation/_cost_extrapolation_visualization_core.py`, `create_cost_fit_plot`, `_build_cost_fit_params` | Rebuilds the correlated parameters and draws the band |
| Styling | `core/src/analysis/cost_extrapolation/_cost_extrapolation_visualization_config.py` | Band transparency, line styles, extrapolation margin |

---

## 1. The fit model

For each parameter group, the average computational cost per spinor per
configuration is fitted as a function of the bare mass $m$ with

$$
C(m) = \frac{a}{m - b} + c ,
$$

using `lsqfit.nonlinear_fit` with broad Gaussian priors on $a$, $b$, $c$.
(The plot legend writes the denominator as $am + |b|$ because $b$ comes
out negative, e.g. $b \approx -0.0074$.)

The fit returns the parameters as `gvar` objects, i.e. as a mean vector
$\bar{\mathbf p} = (\bar a, \bar b, \bar c)$ together with the **full
3 × 3 covariance matrix** $\Sigma$ of the posterior. This covariance
contains the prior information as well as the data.

Example (Wilson, n = 4):

| Parameter | Mean | σ |
| --- | --- | --- |
| $a$ | 4.150 | 0.936 |
| $b$ | −0.00744 | 0.00349 |
| $c$ | 4.56 | 11.83 |

---

## 2. What the band is

The band shows, at every bare mass $m$ along the curve, the 1σ
uncertainty of the fitted function value $C(m)$ that comes from the
uncertainty of the fit parameters:

$$
C(m) \pm \sigma_C(m).
$$

It is drawn from slightly left of the target (derived) bare mass up to
slightly beyond the largest bare mass used in the fit. The curve is
dashed in the extrapolation region (left of the fitting range) and solid
inside it, while the band is continuous over the whole span.

### Linear error propagation

To first order in the parameter fluctuations,

$$
\sigma_C^2(m) = \mathbf g(m)^{\mathsf T}\, \Sigma\, \mathbf g(m)
= \sum_{i,j} g_i(m)\, g_j(m)\, \sigma_i \sigma_j \rho_{ij},
$$

where $\rho_{ij}$ are the parameter correlation coefficients
($\rho_{ii} = 1$) and $\mathbf g$ is the gradient of $C$ with respect to
the parameters:

$$
g_a = \frac{\partial C}{\partial a} = \frac{1}{m-b}, \qquad
g_b = \frac{\partial C}{\partial b} = \frac{a}{(m-b)^2}, \qquad
g_c = \frac{\partial C}{\partial c} = 1 .
$$

In the code this formula is never written out by hand: `gvar` carries
the covariance and performs exactly this first-order propagation
automatically when the function is evaluated on correlated `gvar`
parameters (see Section 5).

---

## 3. The problem: the parameters are strongly correlated

The formula above has two kinds of terms:

- **diagonal terms** $g_i^2 \sigma_i^2$, which only need the individual
  parameter errors;
- **off-diagonal terms** $2\, g_i g_j \sigma_i \sigma_j \rho_{ij}$, which
  need the correlations.

Originally the results CSV stored only the means and the individual
errors $\sigma_a, \sigma_b, \sigma_c$. The correlations were lost when
the fit result was written out, so the band could not be reconstructed
correctly at the plotting stage.

The correlations are far from negligible. For Wilson, n = 4:

| Pair | $\rho$ |
| --- | --- |
| $a$–$b$ | −0.972 |
| $a$–$c$ | −0.981 |
| $b$–$c$ | +0.926 |

Similar values (|ρ| ≈ 0.89–0.98) hold for all four groups.

**Why they are so large.** The three parameters can trade off against
each other while the curve stays close to the data. A larger amplitude
$a$ raises the whole curve, which the fit compensates by lowering the
offset $c$ (hence $\rho_{ac} < 0$) and by moving the pole $b$ further
away from the data (hence $\rho_{ab} < 0$). Many combinations of
$(a, b, c)$ describe the data almost equally well, but only along these
correlated directions. The individual errors are large because each
parameter alone is poorly determined; the *curve* is much better
determined than any single parameter.

### What ignoring the correlations would do

Using only the diagonal terms (setting $\rho_{ij} = 0$ for $i \neq j$)
overestimates the band badly:

| Bare mass $m$ | $C(m)$ | $\sigma_C$ with correlations | $\sigma_C$ ignoring correlations |
| --- | --- | --- | --- |
| 0.00589 (target) | 316.0 | **27.3** | 108.3 |
| 0.010 | 242.6 | **10.8** | 72.8 |
| 0.020 | 155.8 | **5.2** | 40.9 |
| 0.040 | 92.0 | **2.6** | 23.9 |
| 0.070 | 58.2 | **2.7** | 17.1 |

At $m = 0.07$, for example, $\sigma_c = 11.8$ alone (from $g_c = 1$)
would give a band of ±17, several times larger than the error bar of the
data point the curve was fitted to at that mass, which is clearly
unphysical. The off-diagonal
terms are negative and cancel most of the diagonal contributions.

---

## 4. The fix: export the correlations

### Analysis stage

In `fit_and_extrapolate_cost`, after the fit, the correlation matrix of
the three parameters is evaluated with `gvar`:

```python
param_corr = gv.evalcorr([fit.p["a"], fit.p["b"], fit.p["c"]])
```

and its three independent off-diagonal entries are written to the
results CSV alongside the existing means and errors:

| Column | Meaning |
| --- | --- |
| `cost_fit_param_corr_ab` | $\rho_{ab}$ |
| `cost_fit_param_corr_ac` | $\rho_{ac}$ |
| `cost_fit_param_corr_bc` | $\rho_{bc}$ |

Correlations are stored rather than covariances because they are
dimensionless and bounded in $[-1, 1]$. Together with the errors that
are already exported, they determine the covariance completely:

$$
\Sigma_{ij} = \sigma_i \sigma_j \rho_{ij}.
$$

This applies to both the PCAC and pion branches, since both call
`fit_and_extrapolate_cost`.

### Plotting stage

`_build_cost_fit_params` reads the three means, three errors and three
correlations from the results row, rebuilds the full matrices and
creates correlated `gvar` parameters:

```python
corr = np.eye(3)
corr[0, 1] = corr[1, 0] = rho_ab
corr[0, 2] = corr[2, 0] = rho_ac
corr[1, 2] = corr[2, 1] = rho_bc

params = gv.gvar(means, np.outer(errors, errors) * corr)
```

`create_cost_fit_plot` then evaluates the model on an array of bare
masses using these parameters,

```python
y_band = params[0] / (x_fit - params[1]) + params[2]
```

and fills between `gv.mean(y_band) - gv.sdev(y_band)` and
`gv.mean(y_band) + gv.sdev(y_band)`. Because the parameters are
correlated `gvar`s, each `gv.sdev` value is exactly the linear
propagation $\sqrt{\mathbf g^{\mathsf T} \Sigma \mathbf g}$ of
Section 2, including the off-diagonal terms.

### Numerical precision of the round trip

The CSV stores values with 6 decimal places. This is safe: the
eigenvalues of the Wilson n = 4 correlation matrix are
2.92, 0.074 and 0.0058. Even the smallest is several thousand times
larger than the rounding (~10⁻⁶), so the rebuilt matrix stays positive
definite and the band is unaffected by the rounding.

### Older CSV files

If a results CSV was produced before this change (no correlation
columns), `_build_cost_fit_params` returns `None`. The band is then
skipped and a warning is logged; everything else on the plot is drawn as
before. A band built without correlations is never drawn, because it
would be wrong (Section 3).

---

## 5. Consistency check against the extrapolated point

The green diamond and its error bar come from the analysis stage, not
from the plotting code. There, the fit function is evaluated at the
derived bare mass $m_\star$ as a `gvar` that carries its own uncertainty
from the first (mass-fit) step:

$$
C(m_\star) \quad\text{with}\quad m_\star = 0.005889 \pm 0.000144 .
$$

So its error contains **two** independent contributions:

1. the fit uncertainty at fixed $m_\star$, which is what the red band
   shows: $\sigma_{\text{fit}} = 27.29$;
2. the uncertainty of the target bare mass itself, propagated through the
   slope of the curve:
   $\left|\dfrac{\partial C}{\partial m}\right| \sigma_{m_\star}
   = \dfrac{a}{(m_\star - b)^2}\,\sigma_{m_\star} \approx 3.37$.

These add in quadrature:

$$
\sqrt{27.29^2 + 3.37^2} = 27.50 ,
$$

which reproduces the exported `Extrapolated_cost` of 316.0 ± 27.5. This
confirms that the correlations rebuilt from the CSV give the same
covariance the fit produced. It also explains why, at the target mass,
the red band is very slightly narrower than the diamond's error bar:
the band does not include the bare-mass uncertainty, which is shown
separately by the orange vertical band.

The quadrature sum assumes that the derived bare mass is uncorrelated
with the cost fit parameters. This is how the pipeline treats them: the
derived bare mass comes from the PCAC/pion mass fit, while the cost fit
uses timing data, and it is passed between stages as a mean and error
only.

---

## 6. Limitations and caveats

**First-order (linear) propagation.** Both the band and the diamond's
error use linearised error propagation. The model is nonlinear in $b$,
and at the target mass the relative uncertainty of the distance to the
pole is not small, $\sigma_b / (m_\star - \bar b) \approx 0.26$. A
Monte Carlo check (sampling the parameters from the multivariate
Gaussian with the exported covariance, 400 000 samples) gives at the
target mass a median of 314.9 with an asymmetric 68 % interval of
−23.5 / +33.1, i.e. a half-width of 28.3, compared with the linear
±27.3. Inside the fitting range the two methods agree (at $m = 0.01$:
half-width 11.0 vs 10.8; at $m = 0.07$: 2.7 vs 2.7). So the linear band
is accurate where there is data and slightly underestimates, and
symmetrises, the uncertainty in the extrapolation region. This is the
usual behaviour for an extrapolation towards a pole and is worth
stating when presenting the result.

**Pointwise, not simultaneous.** The band is a ±1σ interval for $C(m)$
at each $m$ separately. It is not a region that contains the entire true
curve with 68 % probability. Neighbouring points of the band are also
strongly correlated with each other, so the band should not be read as
independent errors at each mass.

**Priors.** The covariance is that of the Bayesian posterior and
therefore includes the (broad) priors on $a$, $b$, $c$. With priors this
wide the data dominate, but the band is formally conditional on them.

**Statistical only.** The band reflects the statistical uncertainty of
the shifted power law fit. It does not account for the choice of model;
the generalized power law fit on the same plot is a diagnostic for that
systematic, and it is not included in the band.
