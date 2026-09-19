# Project map

Every idea in this project, grouped. Status tags: **DONE** (measured, holds up),
**FAILED** (measured, did not work -- kept so it is not retried), **OPEN**
(not yet tested), **RUNNING**.

The single most useful line in this file: *seven training objectives and three
sharpness terms were tried and none beat plain `l1+mse`.* The lever is not the
loss.

---

## 1. Model and architecture

### 1.1 Spatial-conditioning U-Net (the working model)
- **DONE** 1D→2D lift: mean / product / |difference| per track, directed
  `fwd_i·rev_j` motif planes, distance channel.
- **DONE** Scores **0.656** median per-region Pearson against AkitaV1's 0.556
  on Akita's own published test target, winning 68% of 222 regions.
- **FAILED** Conditioning ablation refuted the mechanistic story: `tile` mode
  (0.9384) barely below `full` (0.9406), so the explicit pairing is not what
  makes it work.

### 1.2 Windowless pairwise model (`~/src/hic-pairwise`)
Predict `M[i,j]` directly from signals at `i` and `j`. No windows, so no reach
limit and trans contacts become expressible.
- **DONE** Works: 0.554 stratified vs a 0.012 distance-only floor.
- **DONE** Endpoints alone score 0.287 -- context is essential, confirming the
  insulation argument.
- **OPEN** The whole motivation is untested: Akita's cooler stores only a
  0.92 Mb band, so every pair tested sits inside what a window already sees.
  Needs a matched long-range target -- the Galaxy hg19 matrices will provide it.
- **OPEN** Saturates by ~3000 steps, which points at the features, not capacity.

### 1.3 Alternative backbones
- **OPEN** `dilated`, `axial`, `pixelmlp` implemented, never run at 2 kb.
  Note: dilated is Akita's trunk family and the user does not want to copy it.
- **OPEN** Reducing U-Net downsampling depth (ours to choose, not borrowed).

### 1.4 Throughput
- **DONE** bf16: 1.34×, accuracy-neutral (0.9542 vs 0.9536, inside seed spread).
- **DONE** `torch.compile`: +12-34%, accuracy-neutral at matched step.
- **DONE** DDP across both GPUs. Net **2.8×** on the 2 kb run (29 → 82 win/s).
- **FAILED** channels_last (slower), FP8 (no conv path on Ada).
- **FAILED** Conditioning-tensor optimisation: 4.8× faster to build but only
  1.6% of a step. The U-Net's convolutions dominate.

---

## 2. Training objectives

### 2.1 What won
- **DONE** `l1+mse+scc` in COUNT space: **+0.0033** SCC at 25 kb, every one of
  three seeds beating every baseline seed.

### 2.2 What failed (do not retry without a new reason)
- **FAILED** `scc` alone (−0.034): correlation is scale-invariant, so nothing
  pins magnitudes and Hann-blended windows disagree on scale.
- **FAILED** `dcorr` in obs/exp space (−0.021): the decay is already divided
  out, so stratifying again adds nothing.
- **FAILED** `ssim`, `grad`, `hf` (sharpness terms) -- see 4.3.
- **FAILED** `insul`, `donut` (structure terms) -- `donut` did not even raise
  APA (1.445 vs baseline 1.453), its one design goal.

### 2.3 Conclusion
The loss is not where the headroom is. Look at inputs, architecture, or accept
the reproducibility ceiling (section 5).

---

## 3. Targets and normalisation

- **DONE** Use **Akita's published train/test matrices verbatim**. Every attempt
  to normalise locally made things worse; this single change took the score from
  0.410 to 0.656.
- **FAILED** Own normaliser, and the repo's `make_akita_normalised_matrix.py` on
  the whole genome -- its own diagnostic reports "STILL BIASED" (distance-only
  baseline 0.49 vs 0.24 on Akita's target).
- **DONE, hazard** A whole-genome obs/exp target keeps residual distance
  dependence; a per-span one does not. Akita normalises within each ~1 Mb span.
- **OPEN** Per-window obs/exp normalisation at cache time, matching Akita.

---

## 4. Evaluation -- the area with the most traps

### 4.1 What the metrics actually measure
- **DONE** Pixel Pearson **rewards blur**: a blurred copy of the truth scores
  0.937 Pearson, 0.991 insulation, 0.824 boundary F1.
- **DONE** SCC ≈ Pearson on obs/exp data (r = 0.9976 across regions). SCC is
  defined for raw counts; on obs/exp its stratification is redundant and its
  variance weighting favours the noisy far strata.
- **DONE** Every new metric must be run against a **distance-only** and a
  **region-swap** baseline. This caught three bad numbers in one day.

### 4.2 Feature-level score (`hicmultiome/bioscore.py`)
- **DONE** Insulation correlation, boundary F1, loop F1, APA, loops-invented,
  reported separately. Our margin over Akita is **wider on biology than on
  pixels**: boundary F1 0.582 vs 0.388, loop F1 0.553 vs 0.465.
- **DONE** Loop calling must run on coarsened bins; at native 2 kb the calls
  carry no CTCF signal at all (anchor AUROC 0.50).

### 4.3 Sharpness
- **DONE** We are blurry: fine-scale energy 0.057 of the truth, vs Hi-cGAN 0.239
  (Akita 0.041).
- **DONE** `+hf` genuinely sharpens (APA 1.633 vs 1.453, closest to the measured
  2.43) but **scatters loops** (precision 0.435 vs 0.847). Sharpness and
  placement are in tension.
- **OPEN** Raise APA toward 2.43 *without* losing precision. Nothing tried yet
  achieves this.
- **NOTE** Much of the truth's high-frequency content is measurement noise and
  is not predictable in principle.

### 4.4 Chromatin-state stratification
- **DONE** All three models score worse in signal-poor regions; the effect is
  universal, not a defect of ours.
- **DONE, caveat** Our correlations are the steepest (RAD21 0.624 vs Akita
  0.375), so stratifying **flatters us** -- it removes exactly where our
  track-dependence hurts and Akita's sequence-only design does not.
- **RULE** Region classes must come from input tracks, never from scores.

---

## 5. The reproducibility ceiling (the reframing result)

- **DONE** Replicate agreement is strongly resolution-dependent: SCC ~0.95 at
  40 kb, **0.39 at 2 kb, 0.26 at 1 kb**. Two experiments on the same cells barely
  agree at the resolution we work at.
- **DONE** Our 0.656 at 2048 bp sits against a biological-replicate Pearson of
  **0.626 at 2 kb**. Not symmetric (we are scored against a deep merged target)
  but it means the gap to 1.0 is mostly irreproducible noise.
- **DONE** This explains section 2.3: the objectives were not failing, there was
  little left to win.
- **RUNNING** Ceiling vs **read depth**, from GM13977's 16 replicates, pooling
  disjoint subsets at 1/2/4/8 replicates. Separates sparse sampling from
  genuinely irreproducible signal.
- **HAZARD** A ceiling must be computed through the *same* pipeline as the
  scored target, smoothing included. Getting this wrong produced an impossible
  "138% of achievable".

---

## 6. Can prediction beat measurement?

- **DONE** Down-sampling: our prediction recovers TAD boundaries from the deep
  map better than a real experiment does **below ~3-5% of depth** (boundary F1
  0.650 vs 0.629 at 3%). On pixel correlation the smoothed shallow map still
  wins.
- **FAILED** The stronger claim, at loop level: loops we predict that the Hi-C
  map lacks are supported by promoter capture Hi-C at **45.5%**, against a
  distance-matched control rate of 50.4% (OR 0.82, p = 0.76). They are not
  missed biology.
- **DONE, control** Loops we predict that the map *does* contain are supported at
  68.1%, matching measured loops at 68.5% -- the pipeline is sound.
- **OPEN** Repeat the capture test on all 413 regions (25 done).

---

## 7. Data

### 7.1 Training targets
- **DONE** Akita's five cell types located: GM12878, H1, IMR90, HCT116, HFF,
  with matched ChIP (GM12878 13/13, IMR90 13/13, HCT116 12/13, H1 12/13,
  HFF 7/13 -- gaps declared as `missing_tracks`, never zero-filled).
- **OPEN** Multi-cell training run using them.

### 7.2 Replicates
- **DONE** Processed replicate matrices are almost never published (4DN: 253
  merged / 127 unreplicated; ENCODE: zero per-replicate Hi-C; 1 of 330 GEO
  series). **But raw reads are abundant: 54,898 ENA Hi-C runs, 469 studies with
  ≥2 replicates, 163 TB.**
- **RUNNING** Galaxy pipeline (galaxy-heidel): minibwa `hic` mode → hg19, mates
  mapped **independently**, then HiCExplorer `hicBuildMatrix` at 1 kb.
  Histories tagged `hicpredict` + `hicraw`.
  - HCT116 rep1/rep2 (PRJNA754527, MboI) -- one of Akita's cell types
  - MOLM-13 AML (PRJNA610800) -- MLL-AF9, FLT3-ITD; enzyme UNCONFIRMED
  - GM13977 ×16 (PRJNA1074296) -- the depth series
- **OPEN** Leukaemia shortlist: T-ALL (PRJNA757417), GATA3/ALL (PRJNA608940),
  EBF1 (PRJNA727325), K562+GM12878 Hi-TrAC (PRJNA746963), trisomy-8 HSC
  (PRJDB16171). 30 studies total.
- **OPEN** Very deep pairs (>10B reads/replicate): PRJNA480492, PRJNA778248.

### 7.3 Perturbation data held
- **DONE** HCT116 cohesin degron: untreated control **and** 6 h auxin, same lab
  and pipeline -- a matched pair, far stronger than two unrelated datasets.
- **DONE** Franke/Lupiáñez rearrangement alleles, G9a knockout, mouse limb.

---

## 8. Perturbation and mechanism

- **DONE** Three knockout types implemented as one tool: protein removal,
  binding-site removal, DNA edit (insertion/deletion/duplication).
- **HAZARD, FIXED** A duplicated sequence must inherit its source's ChIP signal
  (`src_map`); zeroing it created an artificial cliff and a spurious boundary
  that looked like a neo-TAD result.
- **FAILED/OPEN** Mundlos neo-TAD reproduction. The probe was measured at the
  seam instead of the duplicated boundary element (true dip 0.011 vs 0.757).
  **Needs re-running with the corrected probe and the CTCF-motif model.**
- **PARTIAL** Synthetic rule-trained model gets the direction right but only
  ~14% of true boundary strength.
- **OPEN** The core thesis: folding mechanics are cell-type independent, cell
  identity entering only through the ChIP tracks. Not yet tested at scale.
  Note V2 evidence is supportive: AkitaV2 added a species and more cell types
  and gained only 0.05, with cell-type specificity "only slightly improved".

---

## 9. Applications

- **PARTIAL** Web tool: FastAPI + canvas, upper/lower triangle, region edits,
  regional knockouts. **OPEN**: real HiGlass tile serving; sequence edits are
  accepted by the server but only drawn as gaps.
- **OPEN** Cross-species transfer (mouse → human) with name-mapped tracks.
- **PARTIAL** Capture Hi-C prediction (`~/src/pchic-unet`): fragment-level
  pairwise reaches AUROC 0.693; the window-based approach was near chance (0.55).

---

## 10. Standing rules learned the hard way

1. Predictions in a signed target space must **not** be clamped at 0 -- that
   deleted the negative half and made the score meaningless.
2. Score against the **test-span** target, never the whole-genome one
   (distance-only: 0.74 there, 0.24 on Akita's).
3. `max_zero_fraction` does not filter obs/exp matrices. Use mappability.
4. Compute the **trivial baseline** for every metric, every time.
5. Look for existing tooling before writing your own -- the normaliser, the
   Akita matrices and the prediction rasters all already existed.
6. Pin `numpy<2`; installing packages has broken torch and pyBigWig here.
7. `~/src/Hi-cGAN` is a manuscript repo: read-only.
