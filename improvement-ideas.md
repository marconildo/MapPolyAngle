# Improvement Ideas

## Goal

Improve the terrain-aware polygon autosplitter for Wingtra fixed-wing mapping in steep terrain so that:

- for a given flight time budget, coverage quality is better
- for a given coverage target, flight time is lower
- the one-shot `Auto split` result is closer to what an expert user would manually build

## Current Tuning Todo

- [x] Add a soft tolerance around the mean line-length gate so near-miss regions can survive when they are otherwise strong.
- [x] Increase large-root recursion depth from 2 to 3.
- [x] Strengthen anti-overcut pressure with higher region-count and inter-region transition penalties.
- [x] Retain a broader coarse-to-fine frontier by keeping more split candidates and more per-bucket frontier states.
- [x] Improve backend debug artifacts so returned plans include gate and line-length diagnostics for tuning.
- [ ] Revisit exact cut-boundary reweighting so exact-geometry reevaluation does not reuse whole-cell weights on sliced boundary cells.
- [ ] Decide on the frontend frontier explorer UI: expose backend-returned accepted solutions on a tradeoff slider instead of silently picking one.

This note separates:

- near-term improvements to the current Python backend
- limitations of the current formulation
- bigger next-iteration ideas

## Current Problem Shape

The backend is trying to solve a real tradeoff:

- fewer polygons means fewer turns and better battery efficiency
- more polygons means each polygon can align better to local contour direction and reduce line lift / range / density penalties

The current implementation is already acceptable, but it still tends to:

- take too many cuts in some branches
- miss some useful lower-cut alternatives
- rely on a surrogate cost that is not always aligned with the frontend exact evaluator
- behave differently in one-shot solve versus manual repeated child re-splitting

## Main Issues In The Current Backend

### 1. One-shot solve is not solving the same problem as manual re-splitting

Manual child re-splitting rebuilds the DEM/grid/features for the child polygon. The current one-shot recursive solve does not. It keeps solving inside one root grid context.

Implication:

- manual re-splitting can discover cleaner child structure because the child gets a finer effective representation
- one-shot solve is structurally disadvantaged and can underperform the manual workflow even if the search logic were perfect

### 2. Depth is too shallow for large root polygons

Large requests currently use a shallow recursion depth. That limits the number of leaves reachable in one solve and makes the result strongly dependent on the first few split choices.

Implication:

- the solver may never even represent the better plan that a user later discovers by manually splitting one of the children

### 3. “Exact geometry” reevaluation is still approximate

The solver reconstructs exact cut polygons, but region scoring still reuses the original cell set and original per-cell static inputs. Boundary cells cut by the exact polygon are not reweighted by clipped area.

Implication:

- plan costs after exact cut reconstruction are still biased
- boundary-sensitive quality differences between similar candidate plans can be scored incorrectly

### 4. Tradeoff is not a real backend control variable

The request schema accepts a `tradeoff`, but the current solve path mostly uses it as output labeling, not as a true input to search, pruning, or final selection.

Implication:

- there is no true backend mode for “prefer fewer cuts” versus “prefer better quality”
- the system returns a filtered frontier, then the frontend exact scorer and user workflow do the real tradeoff selection

### 5. The backend surrogate is not tight enough against the frontend exact evaluator

The frontend exact evaluator is closer to the real metric because it uses actual flown strips / poses against terrain rasters. The backend surrogate uses terrain-derived proxies.

Observed effect in saved runs:

- the backend often prefers finer splits than the final exact-ranked or exported result
- later child requests are especially vulnerable to this

This suggests the current surrogate overvalues the quality gain from extra splitting in some cases.

### 6. Region-count and mission-time penalties are too weak

The current plan objective adds only a small explicit penalty per added region, and inter-region transition time is simplified.

Implication:

- the solver can keep choosing extra cuts for relatively small surrogate quality gains
- those gains do not always survive the frontend exact reranker

### 7. Candidate generation and pruning are immediate-gain biased

The split generator ranks cuts using immediate 2-way improvement and returns only a small number of split candidates. The frontier is also aggressively thinned.

Implication:

- a split that is mediocre immediately but enables a much better second split can be pruned too early
- this is a likely reason the backend sometimes misses efficient multi-step decompositions

### 8. Practical gates are doing useful work, but also shape the solution strongly

Current plan and region gating relies heavily on:

- minimum child fraction
- scaled mean line length
- capture efficiency
- compactness / convexity
- non-largest fraction / smallest-region fraction

These gates prevent obviously bad specks, but they also create a narrow acceptable corridor. Relaxed fallback then reintroduces some plans through a separate mechanism.

Implication:

- the solver behavior can become discontinuous
- “almost acceptable” plans may be dropped, then a different relaxed fallback candidate is returned instead

## Best Near-Term Improvements

These are the first changes worth trying before redesigning the algorithm.

### 1. Make recursive child solves regrid locally

Highest-value change.

Idea:

- after a split is selected for a child subproblem, rebuild DEM/grid/features for that child polygon and solve recursively on that child context
- if full regridding at every branch is too expensive, do it selectively:
  - root solve stays coarse
  - only promising child branches get local regridding
  - or only branches above a size / complexity threshold get regridded

Expected benefit:

- makes one-shot solve much closer to the current manual workflow
- likely improves both cut placement and bearing estimation on child regions

### 2. Increase depth adaptively instead of using a hard small/large split

Current depth policy is too blunt.

Idea:

- set depth from region complexity, not just cell count
- possible signals:
  - spread of preferred bearings
  - break-strength variance
  - expected quality improvement between best 1-region and best 2-region solution
  - number of distinct heading modes

Expected benefit:

- spend more search only when the region is truly multimodal
- avoid under-solving difficult mountain faces

### 3. Strengthen the cost of extra cuts

Current split-count pressure is probably too weak.

Try:

- increase explicit region-count penalty
- increase inter-region transition penalty
- make turn/transit penalty depend more strongly on short fragmented regions
- penalize plans where the quality gain over a simpler plan is small relative to added region count

Expected benefit:

- fewer gratuitous cuts
- backend frontier aligns better with what the exact reranker later prefers

### 4. Calibrate the surrogate against exact preview data

This is the most practical tuning path.

Idea:

- use saved solve options plus frontend exact reranker outputs as a calibration dataset
- fit better weights for:
  - hole fraction
  - low-density fraction
  - p10 deficit
  - mean deficit
  - line-lift penalties
  - mission-time penalties
  - region-count penalties

Expected benefit:

- backend ordering of options gets closer to the true exact ranking
- fewer cases where backend prefers 4 regions and exact later picks 3, or backend prefers 3 and exact picks 2

### 5. Fix exact-polygon reevaluation to use clipped cell weights

Important correctness improvement.

Idea:

- during exact geometry reevaluation, compute each region’s area weights from exact polygon ∩ cell area
- use those clipped weights in the objective instead of the original whole-cell weights
- optionally also resample representative terrain height for clipped boundary fragments if needed

Expected benefit:

- more trustworthy scoring differences between nearby candidate plans
- less bias at straight cut boundaries

### 6. Let the backend really optimize for a requested tradeoff

Idea:

- use `tradeoff` to shape solve-time scoring and/or final returned solution set
- examples:
  - low tradeoff: stronger penalty on region count and mission time
  - high tradeoff: weaker cut penalty, stronger density / GSD penalty

Expected benefit:

- more predictable behavior
- user intent is represented inside the solve instead of only at the UI layer

### 7. Keep more “strategically different” candidates in the frontier

The current frontier is small and bucketed mainly by region count.

Idea:

- retain more diversity by heading pattern / split topology, not just score ordering
- explicitly keep:
  - best low-region-count candidate
  - best low-time candidate
  - best low-quality-cost candidate
  - best boundary-aligned candidate
  - best balanced-area candidate

Expected benefit:

- better chance that the frontend exact reranker sees the true winner

### 8. Improve root split ranking to reward future optionality

Current ranking is based on immediate improvement.

Idea:

- add a “future split potential” term
- simple proxy:
  - if each child still has strong multimodality in preferred bearing, that split is more valuable
  - if a split isolates a coherent face plus a still-complex residual, that is a good root action

Expected benefit:

- better multi-step decompositions
- fewer cases where a decent first cut gets discarded because its immediate score is modest

## Instrumentation Improvements

These should be added even before major tuning, because they make the solver debuggable.

### 1. Save surrogate and exact scores side-by-side

For every returned option, save:

- backend surrogate cost
- backend mission time
- region count
- exact frontend preview score
- exact hole / low / q10 metrics

This creates a direct calibration dataset.

### 2. Save pruned candidate summaries

For top rejected candidates, log:

- split direction
- threshold
- immediate quality gain
- time delta
- why it was pruned

This is essential for understanding whether the solver is missing good low-cut alternatives.

### 3. Track manual follow-up splits as an oracle signal

If a user rejects the one-shot result and manually splits a child, record:

- which child was split
- what alternatives had originally been available
- whether the new final tree resembles an originally rejected backend candidate

This is extremely valuable product feedback for tuning.

## Limitations Of The Current Formulation

These are not just implementation bugs; they are structural.

### 1. Straight cuts are restrictive

Real terrain boundaries are often curved or piecewise linear. Straight cuts are convenient, but they force awkward region shapes.

### 2. Pure top-down splitting cannot undo bad earlier choices

Once a cut is made, the solver does not merge or reshape regions later.

### 3. Region geometry and flight direction are optimized in a limited alternating way

The current loop effectively does:

- propose region partition
- pick best heading per region among a small candidate set

That is much weaker than jointly optimizing region shape and heading.

### 4. The backend objective still simplifies the real Wingtra flight physics

Important real effects are only partially represented:

- exact turn geometry
- stabilization penalties
- min-clearance mode effects
- turn-extension effects
- true lidar beam footprint / return geometry

### 5. The solve treats coverage as mostly additive by region

But in practice:

- overlap between neighboring regions can be beneficial
- a nested smaller area inside a larger one can be useful
- some intentional redundancy is worth time in high-relief areas

The current partition framing does not express that very naturally.

## Next Iteration Ideas

These are larger algorithm changes after current-backend tuning.

### 1. Split-merge local search

Instead of only recursive splitting:

- start from a coarse partition
- split, merge, and reassign regions iteratively
- allow the solver to recover from a bad early cut

### 2. Curved or polyline cut boundaries

Allow region boundaries to follow terrain breaks rather than only straight half-plane cuts.

Possible approaches:

- graph cut on cells with boundary regularization
- shortest path over a ridge/valley break-strength field
- piecewise linear boundaries with a complexity penalty

### 3. Hierarchical coarse-to-fine solve

Do:

- coarse root partitioning first
- then exact local refinement of promising boundaries
- then optional child-level re-optimization

This is probably the best compromise between runtime and quality.

### 4. Joint optimization of region boundaries and headings

Current method picks heading from a small candidate set after defining region membership. A stronger approach would optimize both together.

Possible formulations:

- region labeling with per-label heading variable
- alternating optimization:
  - assign cells to regions
  - optimize region headings
  - refine region boundaries

### 5. Frontend exact evaluator as a teacher model

Use the current exact preview path as supervision to train or fit a better backend surrogate.

This could be:

- manual weight fitting
- Bayesian optimization of surrogate weights
- a learned regression or ranking model over region metrics

### 6. Optional non-partition overlays

Not every improvement needs a clean partition.

Useful future idea:

- allow a small additional “patch” polygon inside or overlapping a larger polygon when that is more efficient than fully partitioning the whole area

This matches real operator behavior in difficult terrain.

## Recommended Order Of Work

### Phase 1: Improve the current backend

1. Add stronger debug outputs for candidate ranking and exact-vs-surrogate comparison.
2. Reweight extra-region and mission-time penalties.
3. Fix exact reevaluation to use clipped cell weights.
4. Add true backend tradeoff control.
5. Prototype local regridding for recursive child solves.

### Phase 2: Reassess after tuning

Measure on saved cases:

- number of final regions
- exact lidar / camera quality metrics
- exact mission time
- agreement between backend surrogate ranking and frontend exact ranking

### Phase 3: Start the next iteration

If the tuned backend still leaves clear performance on the table:

1. try split-merge search
2. test curved or piecewise-linear boundaries
3. consider a learned surrogate calibrated from exact preview data

## Bottom Line

The current backend is not fundamentally wrong. The main issue is that it is solving an approximation of the real operator workflow and an approximation of the real evaluation metric at the same time.

The best immediate path is:

- make one-shot solve more like manual recursive splitting
- make backend scoring more faithful to the exact evaluator
- make the region-count / mission-time penalty stronger and user-controllable

That should already reduce unnecessary cuts and improve the quality-per-flight-time tradeoff without a full redesign.
