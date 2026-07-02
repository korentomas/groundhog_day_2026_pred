# Future work items 1, 2, 5

Implements the first three items from the Future Work list in
`notebooks/02_bayesian_hierarchical.ipynb`: fake vs. real groundhogs,
Phil agreement, and geographic clustering. Items 3 (streakiness) and 4
(career-length) are out of scope.

## Placement

Three new sections appended to `notebooks/02_bayesian_hierarchical.ipynb`,
after the current Future Work section. Same style as the rest of the
notebook: PyMC models, arviz diagnostics (r-hat, divergences), inline plots.

## 1. Fake vs. real groundhogs

- Check the `type` field on the 88 registry entries to build the real/not-real
  split (README already states 37 real / 51 not; confirm against the data
  rather than trusting the README number blindly).
- Two-group hierarchical model (real vs. not-real) with partial pooling,
  matching the structure of the existing per-groundhog model in section 1.
- Report: posterior difference in group-mean shadow rate, P(real group mean >
  fake group mean).

## 2. Follow the leader (Phil agreement)

- Groundhogs with 5+ years of shared reporting history with Phil.
- Per-groundhog expected agreement rate under independence is NOT 50/50 — it's
  `p_phil * p_g + (1 - p_phil) * (1 - p_g)`, using each groundhog's own
  marginal shadow rate.
- Hierarchical binomial model on observed agreement counts, partial pooling
  across groundhogs, compared against the per-groundhog independence
  baseline.
- Report: which groundhogs (if any) have credibly higher-than-independence
  agreement with Phil.

## 3. Geographic clustering

- Linear-in-lat/lon Bayesian regression on per-groundhog posterior mean
  shadow rate (from section 1's model). Tests for a spatial gradient.
- Given the weak expected signal (noted in the existing Future Work text) and
  the compute cost of a full Gaussian process, this is the first pass rather
  than a full spatial-autocorrelation model.
- Close with a markdown cell arguing why a full GP would be the stronger
  follow-up if this shows any signal — a linear term can't capture
  non-monotonic spatial clustering (e.g. two separate hotspots), a GP would.

## Out of scope

- Items 3 (streakiness) and 4 (career-length effect) from the Future Work
  list.
- Full Gaussian process implementation for item 5 — discussed, not built.

## README / site guidance

- Update the "What's here" description of `02_bayesian_hierarchical.ipynb`
  to mention the three new sections.
- Update the Future Work section to mark items 1, 2, 5 as done (link to the
  relevant notebook section) and keep 3, 4 as open.
- No new site pages needed — Quarto renders the notebook as-is, new sections
  show up automatically on next deploy.
