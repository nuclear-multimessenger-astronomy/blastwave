# Radiation Models

blastwave implements six synchrotron radiation models, selected via the `model` parameter. All models compute the specific intensity \(I_\nu\) at each point on the blast wave surface, which is then integrated over the equal arrival time surface (EATS) or forward-mapped grid to produce observer-frame flux densities.

## Analytic synchrotron (`sync`)

The default model. Implements the Sari, Piran & Narayan (1998) piecewise power-law synchrotron spectrum.

**Characteristic quantities:**

Minimum electron Lorentz factor:

\[
\gamma_m = \frac{p-2}{p-1} \, \varepsilon_e \, \frac{m_p}{m_e} \, (\Gamma - 1)
\]

Cooling Lorentz factor:

\[
\gamma_c = \frac{6\pi \, m_e \, c}{\sigma_T \, B^2 \, t}
\]

Magnetic field from post-shock energy density:

\[
B = \sqrt{8\pi \, \varepsilon_B \, e_\mathrm{density}}
\]

where \(e_\mathrm{density}\) is the post-shock internal energy density.

Characteristic frequencies \(\nu_m\) and \(\nu_c\):

\[
\nu_{m,c} = \frac{3e}{4\pi \, m_e \, c} \, B \, \gamma_{m,c}^2
\]

Peak spectral power:

\[
P_{\nu,\mathrm{max}} = \frac{\sqrt{3} \, e^3 \, B \, n_\mathrm{blast}}{m_e \, c^2}
\]

**Spectral segments:**

The spectrum is divided at \(\nu_m\) and \(\nu_c\) into power-law segments. For slow cooling (\(\gamma_m < \gamma_c\)):

| Frequency range | Slope |
|----------------|-------|
| \(\nu < \nu_m\) | \(\nu^{1/3}\) |
| \(\nu_m < \nu < \nu_c\) | \(\nu^{-(p-1)/2}\) |
| \(\nu > \nu_c\) | \(\nu^{-p/2}\) |

Fast cooling (\(\gamma_c < \gamma_m\)) swaps the roles of \(\nu_m\) and \(\nu_c\) with an intermediate \(\nu^{-1/2}\) segment.

```python
flux = blastwave.FluxDensity_tophat(t, nu, P, model="sync")
```

## Deep Newtonian phase (`sync_dnp`)

Identical to `sync` but handles the trans-relativistic regime where \(\gamma_m\) would drop below unity. When \(\gamma_m < 1\), clamps \(\gamma_m = 1\) and adjusts the spectral normalization. Use this for late-time afterglows or mildly relativistic blast waves.

```python
flux = blastwave.FluxDensity_tophat(t, nu, P, model="sync_dnp")
```

## Synchrotron self-absorption (`sync_ssa`)

Extends the `sync` model with self-absorption at low frequencies, essential for radio light curves.

**Approach:** Computes the self-absorption Lorentz factor \(\gamma_a\) by finding where the optically thin intensity equals the Rayleigh-Jeans blackbody limit:

\[
I_\mathrm{thin}(\nu_a) = I_\mathrm{BB}(\nu_a) = \frac{2 k_B T_\mathrm{eff}}{c^2} \nu^2
\]

where the effective temperature \(T_\mathrm{eff} \propto (\gamma_\mathrm{eff} - 1) \, m_e c^2 / 3\) corresponds to the electrons radiating at frequency \(\nu\).

The final intensity is:

\[
I_\nu = \min(I_\mathrm{thin}, I_\mathrm{BB})
\]

This produces the characteristic self-absorbed radio spectrum: \(I_\nu \propto \nu^2\) below \(\nu_a\), transitioning to the optically thin power-law above.

The code handles six spectral regimes determined by the ordering of \(\gamma_a\), \(\gamma_m\), \(\gamma_c\), covering all slow/fast cooling configurations.

```python
flux = blastwave.FluxDensity_tophat(t, nu, P, model="sync_ssa")
```

## Synchrotron self-Compton (`sync_ssc`)

Adds inverse Compton (IC) scattering to the synchrotron model.

**Thomson Y parameter:**

\[
Y = \frac{1}{2}\left(\sqrt{1 + 4b} - 1\right), \quad b = \frac{\eta_e \, \varepsilon_e}{\varepsilon_B}
\]

where \(\eta_e\) is the radiative efficiency (depends on the cooling regime). The Y parameter modifies the cooling break:

\[
\gamma_c = \frac{6\pi \, m_e \, c}{\sigma_T \, B^2 \, t \, (1 + Y)}
\]

The solver iterates (~100 steps) until \(\gamma_c\) and \(Y\) converge self-consistently.

**Klein-Nishina corrections** (`ssc_kn=1` in `P`): When the scattered photon energy approaches \(m_e c^2\) in the electron rest frame, the Thomson cross-section must be replaced by the Klein-Nishina cross-section. The code computes piecewise power-law segments for \(Y(\gamma)\) with breaks at the KN transition Lorentz factors.

**IC flux contribution:** Approximated as \(Y_T \times\) synchrotron evaluated at the upscattered frequency \(\nu_\mathrm{IC} = \nu / (\frac{4}{3} \gamma_m^2)\).

```python
flux = blastwave.FluxDensity_tophat(t, nu, P, model="sync_ssc")

# With Klein-Nishina corrections
P["ssc_kn"] = 1
flux = blastwave.FluxDensity_tophat(t, nu, P, model="sync_ssc")
```

## Thermal synchrotron (`sync_thermal`)

Implements the Margalit & Quataert (2021, MQ21) thermal + non-thermal electron model. This is important for mildly relativistic shocks where a significant fraction of the post-shock electron energy remains in a thermal (Maxwellian) distribution rather than being accelerated into a power-law tail.

**Dimensionless electron temperature:**

\[
\Theta = \varepsilon_T \cdot \frac{9 \mu \, m_p}{32 \mu_e \, m_e} \cdot \beta^2
\]

where \(\varepsilon_T\) is the electron thermalization efficiency, solved via a quadratic to account for relativistic corrections.

**Thermal emissivity** (MQ21 Eq. 10): Uses the Mahadevan et al. (1996) fitting function \(I'(x)\) for the thermal synchrotron kernel, with a fast-cooling correction factor.

**Non-thermal (power-law) emissivity** (MQ21 Eq. 14): Standard power-law electrons with modified minimum Lorentz factor \(\gamma_m = 1 + a(\Theta) \cdot \Theta\) that depends on temperature, and energy fraction \(\delta = \varepsilon_e / \varepsilon_T\).

**Self-absorption:** Both thermal and power-law absorption coefficients are computed:

\[
\tau = (\alpha_\mathrm{th} + \alpha_\mathrm{pl}) \cdot dr
\]

\[
I_\nu = \frac{j_\mathrm{th} + j_\mathrm{pl}}{\alpha_\mathrm{th} + \alpha_\mathrm{pl}} \left(1 - e^{-\tau}\right)
\]

**Parameters:**

| Key | Default | Description |
|-----|---------|-------------|
| `eps_T` | 1.0 | Electron thermalization efficiency |
| `delta` | `eps_e/eps_T` | Power-law energy fraction |
| `full_volume` | 0 | Set to 1.0 for FM25 full-volume post-shock |
| `k` | 0.0 | CSM density power-law index (for full-volume mode) |

```python
P = {**P, "eps_T": 1.0, "delta": 0.1}
flux = jet.FluxDensity(t, nu, P, model="sync_thermal")
```

### Full-volume post-shock (Ferguson & Margalit 2025)

By default, `sync_thermal` uses a thin-shell approximation. The `full_volume` flag activates the FM25 extension where emission comes from the entire post-shock volume:

- Post-shock fluid Lorentz factor \(\Gamma_\mathrm{fluid}\) from Rankine-Hugoniot jump conditions
- Volume-integrated shell thickness \(dr = R \cdot (1 - \xi_\mathrm{shell})\)
- Downstream density and energy density rescaled for uniform post-shock conditions

This is important for trans-relativistic shocks (\(\beta\Gamma \sim 0.1\)--10) where the thin-shell approximation underpredicts optically thin flux by up to ~1 dex.

```python
P = {**P, "eps_T": 1.0, "full_volume": 1.0, "k": 0.0}
flux = jet.FluxDensity(t, nu, P, model="sync_thermal")
```

## Numeric electron distribution (`numeric`)

Uses a Chang-Cooper implicit finite-difference scheme to evolve the full electron energy distribution \(N(\gamma)\) and compute synchrotron emissivity + self-absorption from the numerical solution, rather than analytic approximations.

See [Numeric Model](../api/numeric.md) for details.

```python
flux = blastwave.FluxDensity_tophat(t, nu, P, model="numeric")
```

## Model comparison

| Model | SSA | IC | Thermal | Numeric | Speed | Use case |
|-------|-----|----|---------|---------|----|----------|
| `sync` | | | | | Fastest | Standard GRB X-ray/optical |
| `sync_dnp` | | | | | Fast | Late-time / trans-relativistic |
| `sync_ssa` | Yes | | | | Fast | Radio light curves |
| `sync_ssc` | | Yes | | | Moderate | High-energy / IC-dominated |
| `sync_thermal` | Yes | | Yes | | Moderate | NS mergers, FBOTs, mildly relativistic |
| `numeric` | Yes | | | Yes | Slow | Non-standard distributions, pair production |

## References

- Sari, R., Piran, T., & Narayan, R. (1998). *ApJL*, 497, L17.
- Wijers, R. A. M. J. & Galama, T. J. (1999). *ApJ*, 523, 177.
- Margalit, B. & Quataert, E. (2021). *ApJ*. [arXiv:2111.00012](https://arxiv.org/abs/2111.00012)
- Ferguson, R. & Margalit, B. (2025). [arXiv:2509.16313](https://arxiv.org/abs/2509.16313) | [GitHub](https://github.com/RossFerguson1/synchrotron_shock_model)
- Dermer, C. D. (2009). *High Energy Radiation from Black Holes.* Princeton.
