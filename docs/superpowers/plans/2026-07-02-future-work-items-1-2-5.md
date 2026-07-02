# Future work items 1, 2, 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new Bayesian analyses to `notebooks/02_bayesian_hierarchical.ipynb` — fake vs. real groundhogs, Phil agreement rate, and geographic clustering — matching the notebook's existing PyMC/arviz style, and update the README/Future Work section to point at them.

**Architecture:** Three new sections appended after the current Future Work section (cell 33) in `notebooks/02_bayesian_hierarchical.ipynb`, each following the pattern already established in sections 1–2: build a small `pm.Model`, sample with `pm.sample(2000, tune=2000, chains=4, target_accept=0.95, random_seed=RANDOM_SEED)`, check convergence with `az.summary`, report results as posterior differences/probabilities rather than point estimates.

**Tech Stack:** PyMC, arviz, pandas, numpy, scipy.special (already imported in cell 1 as `sp`, aliased `pm`/`az`/`pd`/`np`).

## Global Constraints

- Match existing notebook conventions: `RANDOM_SEED = 42` (from cell 1), same sampler kwargs (`2000, tune=2000, chains=4, target_accept=0.95`), r-hat < 1.01 and zero divergences required before reporting results (per notebook's existing convergence-check pattern in cells 4–6, 19–20).
- `gdf` (built in cell 3) already has a `type` column with the literal string `'Groundhog'` for the 38 real groundhogs — confirmed against `data/combined_data.json`'s `isGroundhog` field (perfect 1:1 correspondence, `isGroundhog==1` iff `type=='Groundhog'`). Use `gdf['type'] == 'Groundhog'`, do not re-derive from `isGroundhog`.
- `geo` (built in cell 30) has `slug`, `lat`, `lon`, `shortname`, already deduped to the 81/93 groundhogs with usable coordinates.
- `pred_df` (built in cell 17) has one row per (year, groundhog) with columns `year`, `shadow` (0/1), `slug`, `shortname`, `decade`. Phil's slug is `'punxsutawney-phil'`.
- Every new code cell must run via `jupyter nbconvert --to notebook --execute --inplace notebooks/02_bayesian_hierarchical.ipynb` before commit (repo root, with `notebooks/data` symlinked to `../data` per `.github/workflows/publish.yml` — see Task 4 for the exact commands).
- Outputs get stripped before commit (`jupyter nbconvert --clear-output --inplace`), per this repo's existing convention (see prior commit "Add Quarto site + Pages deploy").

---

### Task 1: Fake vs. real groundhogs — two-group hierarchical model

**Files:**
- Modify: `notebooks/02_bayesian_hierarchical.ipynb` (append cells after cell 33, the "## Future Work" markdown cell)

**Interfaces:**
- Consumes: `gdf` (DataFrame with `type`, `n`, `shadows`, `label` columns, from cell 3), `pm`, `az`, `np`, `pd`, `plt`, `RANDOM_SEED` (all from cell 1)
- Produces: `idata_real` (InferenceData with posterior vars `mu_real`, `mu_fake`, `p_real`, `p_fake` — logit-space group means and their invlogit'd probabilities), `gdf['is_real']` (bool column added to `gdf` for reuse in later tasks if needed)

- [ ] **Step 1: Add markdown header cell**

```markdown
## 4. Fake vs. real groundhogs

38 of the 93 registry entries are actual groundhogs (`type == 'Groundhog'`);
the rest are taxidermied groundhogs, people in costumes, plush toys, an
alligator, a lobster, and similar. Does shadow rate differ between the real
groundhogs and everything else, or is it noise around 50% either way?
```

- [ ] **Step 2: Add model cell**

```python
gdf['is_real'] = (gdf['type'] == 'Groundhog').astype(int)
print(gdf.groupby('is_real')['n'].agg(['count', 'sum']))

group_idx = gdf['is_real'].values  # 0 = not real, 1 = real
coords_real = {"group": ["fake", "real"]}

with pm.Model(coords=coords_real) as real_model:
    mu_group = pm.Normal("mu_group", mu=0.0, sigma=1.5, dims="group")
    sigma_i = pm.HalfNormal("sigma_i", sigma=1.0)

    z_i = pm.Normal("z_i", mu=0.0, sigma=1.0, shape=len(gdf))
    theta_i = mu_group[group_idx] + z_i * sigma_i
    p_i = pm.math.invlogit(theta_i)

    y = pm.Binomial("y", n=gdf['n'].values, p=p_i, observed=gdf['shadows'].values)

    p_group = pm.Deterministic("p_group", pm.math.invlogit(mu_group), dims="group")

    idata_real = pm.sample(
        2000, tune=2000, chains=4, target_accept=0.95,
        random_seed=RANDOM_SEED, progressbar=True,
    )
```

- [ ] **Step 3: Add convergence-check cell**

```python
print(az.summary(idata_real, var_names=["mu_group", "sigma_i"]))
n_div = int(idata_real.sample_stats.diverging.sum())
print(f"divergences: {n_div}")
```

- [ ] **Step 4: Add results cell**

```python
p_fake_draws = idata_real.posterior['p_group'].sel(group="fake").values.flatten()
p_real_draws = idata_real.posterior['p_group'].sel(group="real").values.flatten()
diff = p_real_draws - p_fake_draws

print(f"P(real) posterior mean: {p_real_draws.mean():.3f} (94% HDI {np.percentile(p_real_draws,3):.3f}-{np.percentile(p_real_draws,97):.3f})")
print(f"P(fake) posterior mean: {p_fake_draws.mean():.3f} (94% HDI {np.percentile(p_fake_draws,3):.3f}-{np.percentile(p_fake_draws,97):.3f})")
print(f"Difference (real - fake): {diff.mean():+.3f} (94% HDI {np.percentile(diff,3):+.3f} to {np.percentile(diff,97):+.3f})")
print(f"P(real group mean > fake group mean): {(diff > 0).mean():.3f}")
```

- [ ] **Step 5: Add markdown takeaway cell (fill in after seeing real numbers)**

```markdown
[Write 1-2 sentences stating the actual P(real > fake) value and whether the
94% HDI on the difference excludes zero. Follow the notebook's existing
voice — see cell 15's takeaway for tone.]
```

- [ ] **Step 6: Execute and verify**

Run from repo root:
```bash
ln -sf ../data notebooks/data
jupyter nbconvert --to notebook --execute --inplace notebooks/02_bayesian_hierarchical.ipynb
```
Expected: exits 0, no `CellExecutionError`. Open the notebook and confirm the convergence-check cell (Step 3) printed r-hat ≤ 1.01 for `mu_group` and `sigma_i`, and `divergences: 0`. Fill in Step 5's markdown with the actual numbers before moving on.

- [ ] **Step 7: Strip outputs and commit**

```bash
jupyter nbconvert --clear-output --inplace notebooks/02_bayesian_hierarchical.ipynb
rm -f notebooks/data
git add notebooks/02_bayesian_hierarchical.ipynb
git commit -m "Add fake vs. real groundhog model (future work item 1)"
```

---

### Task 2: Phil agreement rate

**Files:**
- Modify: `notebooks/02_bayesian_hierarchical.ipynb` (append after Task 1's cells)

**Interfaces:**
- Consumes: `pred_df` (from cell 17), `gdf` (from cell 3, for naive `shadow_rate` per slug), `pm`, `az`, `sp` (scipy.special), `np`, `pd`, `RANDOM_SEED`
- Produces: `idata_agreement` (InferenceData with `mu_eff`, `sigma_eff`, `effect` (per-groundhog, dims `"groundhog"`)), `agree_df` (DataFrame: `slug`, `shortname`, `n_shared`, `n_agree`, `p_phil`, `p_g`, `expected_rate`, `observed_rate`)

- [ ] **Step 1: Add markdown header cell**

```markdown
## 5. Follow the leader: agreement with Punxsutawney Phil

Per-groundhog agreement rate with Phil, for groundhogs with 5+ years of
shared reporting history. The baseline isn't a flat 50% — two groundhogs
with their own independent shadow rates `p_phil` and `p_g` agree by chance at
rate `p_phil*p_g + (1-p_phil)*(1-p_g)`. Model the observed agreement rate
against that baseline directly, rather than eyeballing a 33%-71% raw range.
```

- [ ] **Step 2: Add data-prep cell**

```python
phil_by_year = pred_df[pred_df['slug'] == 'punxsutawney-phil'].set_index('year')['shadow']
p_phil = gdf.loc[gdf['slug'] == 'punxsutawney-phil', 'shadow_rate'].iloc[0]

agree_rows = []
for slug, sub in pred_df[pred_df['slug'] != 'punxsutawney-phil'].groupby('slug'):
    g_by_year = sub.set_index('year')['shadow']
    shared_years = g_by_year.index.intersection(phil_by_year.index)
    if len(shared_years) < 5:
        continue
    g_vals = g_by_year.loc[shared_years]
    phil_vals = phil_by_year.loc[shared_years]
    n_shared = len(shared_years)
    n_agree = int((g_vals.values == phil_vals.values).sum())
    p_g_row = gdf.loc[gdf['slug'] == slug, 'shadow_rate']
    if p_g_row.empty:
        continue
    p_g = p_g_row.iloc[0]
    expected_rate = p_phil * p_g + (1 - p_phil) * (1 - p_g)
    agree_rows.append({
        'slug': slug,
        'shortname': sub['shortname'].iloc[0],
        'n_shared': n_shared,
        'n_agree': n_agree,
        'p_phil': p_phil,
        'p_g': p_g,
        'expected_rate': expected_rate,
        'observed_rate': n_agree / n_shared,
    })

agree_df = pd.DataFrame(agree_rows)
print(f"{len(agree_df)} groundhogs with 5+ shared years with Phil")
print(agree_df[['shortname', 'n_shared', 'observed_rate', 'expected_rate']].sort_values('observed_rate', ascending=False).to_string(index=False))
```

- [ ] **Step 3: Add model cell**

```python
coords_agree = {"groundhog": agree_df['slug'].values}
logit_expected = sp.logit(agree_df['expected_rate'].values)

with pm.Model(coords=coords_agree) as agreement_model:
    mu_eff = pm.Normal("mu_eff", 0.0, 1.0)
    sigma_eff = pm.HalfNormal("sigma_eff", 1.0)

    z_eff = pm.Normal("z_eff", 0.0, 1.0, dims="groundhog")
    effect = pm.Deterministic("effect", mu_eff + z_eff * sigma_eff, dims="groundhog")

    theta = logit_expected + effect
    p_agree = pm.Deterministic("p_agree", pm.math.invlogit(theta), dims="groundhog")

    y = pm.Binomial("y", n=agree_df['n_shared'].values, p=p_agree, observed=agree_df['n_agree'].values, dims="groundhog")

    idata_agreement = pm.sample(
        2000, tune=2000, chains=4, target_accept=0.95,
        random_seed=RANDOM_SEED, progressbar=True,
    )
```

- [ ] **Step 4: Add convergence-check cell**

```python
print(az.summary(idata_agreement, var_names=["mu_eff", "sigma_eff"]))
n_div = int(idata_agreement.sample_stats.diverging.sum())
print(f"divergences: {n_div}")
```

- [ ] **Step 5: Add results cell**

```python
effect_post = idata_agreement.posterior['effect']
agree_df['effect_mean'] = effect_post.mean(dim=['chain', 'draw']).values
agree_df['effect_lo'] = effect_post.quantile(0.03, dim=['chain', 'draw']).values
agree_df['effect_hi'] = effect_post.quantile(0.97, dim=['chain', 'draw']).values
agree_df['credibly_above_baseline'] = agree_df['effect_lo'] > 0
agree_df['credibly_below_baseline'] = agree_df['effect_hi'] < 0

print(agree_df[['shortname', 'n_shared', 'observed_rate', 'expected_rate', 'effect_mean', 'effect_lo', 'effect_hi']].sort_values('effect_mean', ascending=False).to_string(index=False))
print(f"\n{agree_df['credibly_above_baseline'].sum()} groundhogs credibly agree with Phil more than chance")
print(f"{agree_df['credibly_below_baseline'].sum()} groundhogs credibly agree with Phil less than chance")
```

- [ ] **Step 6: Add markdown takeaway cell (fill in after seeing real numbers)**

```markdown
[Write 1-2 sentences: does any groundhog show a 94% HDI on `effect` that
excludes zero? Name it if so. State the population-level `mu_eff` finding —
is there a systemic Phil-copycat tendency, or is `mu_eff`'s HDI straddling
zero?]
```

- [ ] **Step 7: Execute and verify**

```bash
ln -sf ../data notebooks/data
jupyter nbconvert --to notebook --execute --inplace notebooks/02_bayesian_hierarchical.ipynb
```
Expected: exits 0. Confirm r-hat ≤ 1.01 for `mu_eff`, `sigma_eff`, 0 divergences. Fill in Step 6's markdown.

- [ ] **Step 8: Strip outputs and commit**

```bash
jupyter nbconvert --clear-output --inplace notebooks/02_bayesian_hierarchical.ipynb
rm -f notebooks/data
git add notebooks/02_bayesian_hierarchical.ipynb
git commit -m "Add Phil agreement-rate model (future work item 2)"
```

---

### Task 3: Geographic clustering (linear model + GP discussion)

**Files:**
- Modify: `notebooks/02_bayesian_hierarchical.ipynb` (append after Task 2's cells)

**Interfaces:**
- Consumes: `geo` (from cell 30: `slug`, `lat`, `lon`, `shortname`), `gdf` (for `post_mean`, computed in cell 12), `pm`, `az`, `np`, `pd`, `RANDOM_SEED`
- Produces: `idata_geo` (InferenceData with `intercept`, `beta_lat`, `beta_lon`, `sigma`)

- [ ] **Step 1: Add markdown header cell**

```markdown
## 6. Geographic clustering

Using the 81 groundhogs with usable coordinates: is there a linear spatial
gradient in shadow rate (using each groundhog's partially-pooled posterior
mean from section 1 as the outcome)? Expected signal is weak, both because
the registry is heavily clustered around the mid-Atlantic and because a
linear term can only pick up a gradient, not clustering. Treated as a first
pass — see the note at the end of this section on why a full Gaussian
process is the natural next step.
```

- [ ] **Step 2: Add data-prep + model cell**

```python
geo_model_df = geo.merge(gdf[['slug', 'post_mean', 'n']], on='slug', how='inner').dropna(subset=['post_mean'])
print(f"{len(geo_model_df)} groundhogs with coordinates + posterior mean shadow rate")

lat_z = (geo_model_df['lat'] - geo_model_df['lat'].mean()) / geo_model_df['lat'].std()
lon_z = (geo_model_df['lon'] - geo_model_df['lon'].mean()) / geo_model_df['lon'].std()

with pm.Model() as geo_model:
    intercept = pm.Normal("intercept", 0.5, 0.5)
    beta_lat = pm.Normal("beta_lat", 0.0, 0.2)
    beta_lon = pm.Normal("beta_lon", 0.0, 0.2)
    sigma = pm.HalfNormal("sigma", 0.2)

    mu = intercept + beta_lat * lat_z.values + beta_lon * lon_z.values
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=geo_model_df['post_mean'].values)

    idata_geo = pm.sample(
        2000, tune=2000, chains=4, target_accept=0.95,
        random_seed=RANDOM_SEED, progressbar=True,
    )
```

- [ ] **Step 3: Add convergence-check cell**

```python
print(az.summary(idata_geo, var_names=["intercept", "beta_lat", "beta_lon", "sigma"]))
n_div = int(idata_geo.sample_stats.diverging.sum())
print(f"divergences: {n_div}")
```

- [ ] **Step 4: Add results cell**

```python
beta_lat_draws = idata_geo.posterior['beta_lat'].values.flatten()
beta_lon_draws = idata_geo.posterior['beta_lon'].values.flatten()

print(f"beta_lat: mean={beta_lat_draws.mean():+.3f}, 94% HDI [{np.percentile(beta_lat_draws,3):+.3f}, {np.percentile(beta_lat_draws,97):+.3f}]")
print(f"beta_lon: mean={beta_lon_draws.mean():+.3f}, 94% HDI [{np.percentile(beta_lon_draws,3):+.3f}, {np.percentile(beta_lon_draws,97):+.3f}]")
print(f"P(beta_lat > 0): {(beta_lat_draws > 0).mean():.3f}")
print(f"P(beta_lon > 0): {(beta_lon_draws > 0).mean():.3f}")
```

- [ ] **Step 5: Add markdown takeaway + GP discussion cell (fill in numbers after seeing results)**

```markdown
[1-2 sentences: do either 94% HDI exclude zero? State the actual values.]

**Why a full Gaussian process would be the stronger follow-up:** this linear
model can only detect a monotonic gradient across latitude or longitude
separately. It can't detect clustering that isn't a straight-line trend — for
example two separate regional hotspots with similar shadow rates but average
coordinates in between, or a spatial pattern that runs diagonally rather
than along the lat/lon axes. A GP with a 2D spatial kernel (e.g. Matérn) over
`(lat, lon)` would model spatial covariance directly: nearby groundhogs
correlated regardless of the direction of the pattern, with the kernel's
lengthscale itself estimated from the data rather than assumed linear. Worth
running if this or a later pass turns up any spatial signal worth pinning
down more precisely.
```

- [ ] **Step 6: Execute and verify**

```bash
ln -sf ../data notebooks/data
jupyter nbconvert --to notebook --execute --inplace notebooks/02_bayesian_hierarchical.ipynb
```
Expected: exits 0. Confirm r-hat ≤ 1.01, 0 divergences. Fill in Step 5's markdown numbers.

- [ ] **Step 7: Strip outputs and commit**

```bash
jupyter nbconvert --clear-output --inplace notebooks/02_bayesian_hierarchical.ipynb
rm -f notebooks/data
git add notebooks/02_bayesian_hierarchical.ipynb
git commit -m "Add geographic clustering model + GP discussion (future work item 5)"
```

---

### Task 4: Update README and Future Work section, deploy

**Files:**
- Modify: `README.md`
- Modify: `notebooks/02_bayesian_hierarchical.ipynb` (cell 33, the "## Future Work" markdown cell — mark items 1, 2, 5 done)

**Interfaces:**
- Consumes: nothing new (this task is pure documentation)
- Produces: nothing consumed by other tasks (this is the final task)

- [ ] **Step 1: Update cell 33 in the notebook**

Read the current cell 33 content, then replace the numbered list so items 1, 2, and 5 are marked done with a pointer to their new section, e.g.:

```markdown
## Future Work

Things worth doing next, already checked against the real data so these aren't
just vague ideas:

1. ~~**Fake vs. real groundhogs**~~ — done, see section 4 above.
2. ~~**Follow the leader**~~ — done, see section 5 above.
3. **Streakiness**: year-to-year autocorrelation per groundhog, for the 25
   with 20+ recorded calls. Do any of them run in actual streaks (several
   shadow years in a row, then several no-shadow years), or is every call
   independent of the last.
4. **Career-length effect**: groundhogs that started reporting in 2023-2025
   versus century-old veterans like Phil (since 1887) and Octoraro Orphie
   (since 1926). Does shadow rate or reporting reliability change with how
   long a groundhog has been on the job.
5. ~~**Geographic clustering**~~ — done (linear pass), see section 6 above.
   Full Gaussian process is the natural next step if a later look turns up
   spatial signal worth pinning down.

One thing that got ruled out: checking predictions against actual NOAA
spring-onset data. The API's own embedded temperature field only has usable
values for 13 of 1761 predictions, too sparse to build any kind of accuracy
score on.
```

Use the Read tool on the notebook JSON to find cell 33's exact current markdown source before editing (the strikethrough text above is illustrative — match the existing prose style exactly, don't just copy this verbatim).

- [ ] **Step 2: Update README's notebook description**

In `README.md`, find this line (in the "What's here" section):

```
- `notebooks/02_bayesian_hierarchical.ipynb`: the real analysis. Two hierarchical PyMC models (per-groundhog partial pooling, and a crossed groundhog x decade model to check whether the apparent post-1980 trend survives once you control for who was reporting), a posterior-based ability ranking, and a geographic map of the groundhog registry. Both models converge cleanly: r-hat ~1.0, zero divergences.
```

Replace with:

```
- `notebooks/02_bayesian_hierarchical.ipynb`: the real analysis. Two hierarchical PyMC models (per-groundhog partial pooling, and a crossed groundhog x decade model to check whether the apparent post-1980 trend survives once you control for who was reporting), a posterior-based ability ranking, a geographic map of the groundhog registry, plus three follow-up models: fake vs. real groundhogs, agreement rate with Punxsutawney Phil, and a first pass at geographic clustering. All models converge cleanly: r-hat ~1.0, zero divergences.
```

- [ ] **Step 3: Update README's Future Work section**

Find:

```
## Future work

See the Future Work section at the end of `notebooks/02_bayesian_hierarchical.ipynb` for what's next: fake vs. real groundhog comparisons, Phil-agreement rates, per-groundhog streakiness, career-length effects, and geographic clustering. Checking predictions against actual NOAA spring-onset data was investigated and dropped: the API's own temperature field only has usable values for 13 of 1761 predictions, too sparse to build anything on.
```

Replace with:

```
## Future work

Fake vs. real groundhog comparisons, Phil-agreement rates, and a first pass
at geographic clustering are done — see sections 4-6 of
`notebooks/02_bayesian_hierarchical.ipynb`. Still open: per-groundhog
streakiness and career-length effects, see the Future Work section at the
end of that notebook. Checking predictions against actual NOAA spring-onset
data was investigated and dropped: the API's own temperature field only has
usable values for 13 of 1761 predictions, too sparse to build anything on.
```

- [ ] **Step 4: Verify full notebook re-executes clean end to end**

```bash
cd /Users/tk/Documents/Personal/groundhog_day_2026_pred
rm -rf data notebooks/data
python3 fetch_data.py
ln -s ../data notebooks/data
jupyter nbconvert --to notebook --execute --inplace notebooks/01_shadow_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_bayesian_hierarchical.ipynb
```
Expected: both exit 0, no `CellExecutionError`. This re-fetches fresh data and runs both notebooks in the same order CI does, so it's the final check that nothing in the new cells broke the full pipeline.

- [ ] **Step 5: Strip outputs, clean up, commit, push**

```bash
jupyter nbconvert --clear-output --inplace notebooks/01_shadow_eda.ipynb notebooks/02_bayesian_hierarchical.ipynb
rm -f notebooks/data
git add README.md notebooks/01_shadow_eda.ipynb notebooks/02_bayesian_hierarchical.ipynb
git status --short   # confirm 01 shows no diff (or only if it changed - it shouldn't)
git commit -m "Update README and Future Work section for items 1, 2, 5"
git push origin main
```

- [ ] **Step 6: Verify CI deploy succeeds**

```bash
sleep 30
gh run list --repo korentomas/groundhog_day_2026_pred --workflow=publish.yml --limit 1
```
Expected: run for the new commit shows `in_progress` then eventually `completed success`. If it fails, run `gh run view <run-id> --repo korentomas/groundhog_day_2026_pred --log-failed` and fix — PyMC sampling in CI takes several minutes with 3 new models added, expect this run to take longer than prior runs (~5-8 min instead of ~3-4 min).

---

## Self-Review Notes

- **Spec coverage:** item 1 (Task 1), item 2 (Task 2), item 5 linear + GP discussion (Task 3), README/notebook guidance (Task 4) — all three spec sections covered.
- **Type consistency:** `gdf['is_real']` (Task 1) is not consumed by Tasks 2/3, no cross-task dependency risk. `agree_df` (Task 2) and `geo_model_df` (Task 3) are each self-contained, built fresh from `gdf`/`geo`/`pred_df` rather than from each other's outputs — no ordering dependency beyond "cell 1 and section-1/3 cells must have already run", which is guaranteed by appending sequentially in the same notebook.
- **No placeholders:** all code cells are complete and runnable as written. The two "markdown takeaway" steps (Task 1 Step 5, Task 2 Step 6, Task 3 Step 5) intentionally require filling in actual numbers after execution — this mirrors the existing notebook's pattern (e.g. cell 15, cell 23) where takeaway prose quotes real computed values, which can't be known before running the model. This is real analysis work, not a missing spec.
