# Similarity Visualisation Revert to Evolving Bars (April 2026)

## What changed

The trajectory stage now renders per-algorithm similarity progression as an evolving bar chart (GIF) again, instead of a layer-by-epoch heatmap.

Concretely, for each algorithm (`ga`, `ssd`, `salun`, `certified`, `scrub`) and each reference (`retrained`, `original`), we now save:

- a static summary line figure (unchanged), and
- an evolving bar chart GIF where each frame is one unlearning step.

This restores the previous notebook behavior where metric-level changes are visually tracked step-by-step.

## Why this was done

This revert is for interpretability and presentation consistency in the dissertation:

1. The research question at this stage is metric evolution over unlearning steps, not spatial structure across layers.
2. A bar chart per step makes cross-metric ranking and magnitude differences immediately visible for non-expert readers.
3. The GIF form preserves temporal ordering (step 0 to final step) without requiring many separate static figures.

In short: heatmaps were compact, but evolving bars are easier to explain in a thesis narrative when you want to discuss "which similarity metrics drop or recover first" as unlearning progresses.

## How the evolving bars are computed

At each unlearning step:

1. We compute similarity rows for all tracked layers and all selected metrics.
2. Lower-is-better metrics are oriented to "higher = more similar" using the same normalization logic as before.
3. We take the mean value across layers for each metric.
4. We render one bar chart frame from these per-metric means.

This means the bar chart tracks model-level similarity trends while still being derived from all layer-level calculations.

## Implementation details

Files changed:

- `src/gradient_ascent/trajectories.py`
  - replaced heatmap renderer with `save_similarity_evolving_bar_plot(...)` (GIF output via `PillowWriter`).
- `src/gradient_ascent/experiments.py`
  - `SimilarityArtifact` now stores `evolving_bar_plot_path` instead of `heatmap_plot_path`.
  - trajectory export now writes `..._evolving_bars.gif` and logs it to wandb.
- `src/gradient_ascent/notebook_helpers.py`
  - notebook display path now shows `evolving_bar_plot_path`.

## Expected outputs after this change

For the five unlearning baselines, you should now get:

- 10 similarity summary figures (`5 algorithms x 2 references`), and
- 10 evolving similarity bar GIFs (`5 algorithms x 2 references`).

## References

1. Cleveland, W. S., and McGill, R. “Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods.” *Journal of the American Statistical Association* 79(387), 1984.  
   [https://doi.org/10.1080/01621459.1984.10478080](https://doi.org/10.1080/01621459.1984.10478080)
2. Heer, J., and Robertson, G. “Animated Transitions in Statistical Data Graphics.” *IEEE TVCG* 13(6), 2007.  
   [https://doi.org/10.1109/TVCG.2007.70539](https://doi.org/10.1109/TVCG.2007.70539)
3. Tufte, E. R. *The Visual Display of Quantitative Information* (2nd ed.), Graphics Press, 2001.  
   (General principles for choosing simpler encodings for comparative quantitative reading.)
