# Stamp of Disapproval — explanation and oral defense

## Result

The detected positions draw **S A R S A**, so the recovered flag is **`SIGMOID_SARSA`**. The final detector generated one box for each of the 130 pages. The submission folder is `../lmml-2026/task_kazakhstan/`. The flag was read from the detector's raw overlays; the detector does not receive the word or any letter templates. Platform acceptance has not been tested.

## What the task asks

Each of five albums has 26 pages. Every page has three stamps, and exactly one small altered region. Detect each region automatically. Putting the 26 positions of an album on the same coordinate sheet reveals one letter. Read the albums in order. The 30 labelled exhibits are separate stamps with masks showing altered regions. There are no album masks or untouched reference copies.

The output requirements are a flag and a URL containing the detector, its outputs and a write-up. The public README is the submission method and reproduction record. These companion study notes stay local and outside Git.

## Why compression history matters

JPEG is a lossy image format: saving an image can discard information. The supplied hint says stamps were compressed before alteration, then the assembled page was saved again at high quality. Thus authentic pixels usually retain traces of the earlier save, while most newly edited pixels only carry the later compression.

Looking for strange denominations or comparing against a catalogue would not satisfy the task. The solution instead measures a numerical inconsistency in how regions were saved. It never uses another copy of a stamp as a reference.

## JPEG concepts used in the solution

JPEG divides the brightness image into 8 × 8 blocks. The discrete cosine transform, or DCT, describes a block using 64 coefficients. A coefficient is the weight of a particular cosine pattern. The DC coefficient represents the average brightness; the AC coefficients describe changing patterns, from broad gradients to fine detail.

After centering brightness around 128, the orthonormal DCT is `C = D @ block @ D.T`. Here `D` is an 8 × 8 matrix whose rows are cosine patterns. The implementation constructs this matrix explicitly.

Quantization rounds a coefficient to a grid of allowed values. For coefficient `C` and quantization step `Q`, JPEG approximately replaces it with `Q * round(C / Q)`. A large step discards more detail. A quality setting chooses a table of steps: higher quality generally means smaller steps.

A quantization lattice is simply this set of allowed multiples: for `Q = 5`, values near ..., -10, -5, 0, 5, 10, ... are expected. The later JPEG save and pixel rounding perturb those values, so exact equality is not required.

The phase is the position of the old block grid relative to the current crop. Pasting or cropping a stamp can shift its grid. There are 8 possible horizontal offsets and 8 vertical offsets, giving 64 combinations. Testing only the page's own JPEG grid would miss most of the first-compression signal.

## Step by step

1. **Locate stamps.** Compare pixels with the dark stock-book background, close small gaps in the binary mask, and take sufficiently large connected components. A connected component is a set of touching foreground pixels. This rule comes from the supplied baseline. Ignore an 18-pixel interior border during point selection to reduce perforation artifacts.

2. **Recover the earlier compression.** Test all 64 block phases and quality settings 55 through 96 on each stamp independently. Calculate DCT coefficients on each candidate grid. Measure how strongly sufficiently large coefficients cluster around the quantization multiples. The periodic score uses `cos(2*pi*C/Q)`: it is near 1 at multiples of `Q`. Coefficients near zero are excluded from this estimation because zero is a multiple of every step and would not identify the original table. The recovered exhibit qualities range from 82 to 92.

3. **Measure local violations.** On the chosen grid, calculate `e = abs(C/Q - round(C/Q))`. Values near zero are compatible with the earlier quantization. Values closer to 0.5 lie between its allowed values. Pixel clipping and the final save also disturb this pattern, so the solution uses multiple statistics rather than treating a single large error as proof of an edit.

4. **Build 200 features.** A feature is one numerical measurement provided to the classifier. Features include the 64 coefficient errors, coefficient magnitudes, local means of errors in different frequency bands, the number of active coefficients, DC error, brightness, variance, saturation fractions and differences from surrounding blocks. Neighbourhoods of 1, 3, 5 and 9 blocks give both local detail and context. No page coordinate, filename, album identifier or letter shape is used as a feature.

5. **Learn from the exhibits.** A gradient-boosted tree classifier combines small decision trees, each improving the current predictions. It is trained to distinguish mask-positive from mask-negative block centres. The final model uses 58,363 valid training blocks, including 1,256 positive centres. It uses 180 boosting iterations, at most 15 leaves per tree, learning rate 0.08, regularization 3 and random seed 2026. Labels are never obtained from album pages.

6. **Predict the album pages.** Recover each stamp's compression parameters, compute its features and predict a score per block. Reconstruct the page score map, smooth it over a 21 × 21 pixel neighbourhood and select the strongest valid point across the page's stamps. Save a 50 × 50 box around that point. The box is a localization aid, not an exact segmentation of all altered pixels.

7. **Decode the output.** Plot the unchanged predicted centres on one sheet per album. The sheets read S, A, R, S, A. No points were moved by hand to improve the letters. A separate rendering script adds album/letter captions only after detection.

## Actual checks and interpretation

Three-fold validation keeps entire exhibits together. This matters because randomly splitting pixels from one stamp between training and validation would let the model see almost the same image in both sets and could produce misleading results.

The held-out centre is inside the supplied mask for 25/30 exhibits with 3 × 3 block smoothing: 8/10, 8/10 and 9/10. This is 83.3% exhibit localization success under that rule. The block-level ROC AUC values are 0.948758, 0.989673 and 0.896213. ROC AUC measures how often a positive block receives a higher score than a negative block; it does not directly measure the correctness of a page's single selected box.

The final output has 130 rows, 26 per album, with a matching score map and annotated image for every row. Eleven pages have scores below 0.5. Their exact names and scores are recorded in `outputs/summary.json`. A classifier score is not a calibrated probability that a page's chosen box is correct.

There are no album ground-truth masks, so an album accuracy percentage cannot honestly be reported. The readable five-letter overlay is the decoding evidence. Isolated weak predictions remain visible and have not been silently replaced with hand-picked regions.

The packaged training command was rerun separately and reproduced the same fold metrics. Input checksums identify the exact dataset. The public implementation is CPU-only and does not require the larger experimental dependencies.

## Earlier experiments and why they were not used

Before using the compression hint, appearance-based features were unreliable. A residual is an image minus a smoothed version; it emphasizes fine texture, edges and noise. A first residual/recompression classifier found 13/30 held-out regions; a revision found 12/30. Adding a pretrained noise representation reached 14/30. A small segmentation network found 1/6 held-out regions. These approaches confused authentic printed artwork and blank paper with manipulation.

The final detector uses the recovered first JPEG grid and quantization table. This turns the hint into a measurable, stamp-specific reference. The abandoned appearance models, pretrained weights and letter-fitting experiments are not dependencies of the submitted detector. The final flag and boxes come from the compression detector's unrestricted predictions.

## Common mistakes and limitations

- Measuring only the final page JPEG grid misses the shifted first grid.
- Raw recompression error also responds to genuine edges and texture.
- Near-zero coefficients alone do not identify a quality setting.
- Completely flat or clipped blocks can imitate or destroy quantization evidence.
- Altered rectangles may cover partially untouched pixels, and blocks crossing their boundaries contain a mixture of both histories.
- A fixed 50 × 50 output box can be smaller or larger than the actual alteration.
- The method assumes the first JPEG quantization evidence survives. Strong resizing or repeated low-quality recompression could remove it.
- Holding out whole exhibits avoids image leakage but 30 exhibits still give a small validation set. The final hint-guided design was selected on this set, so the number is development validation rather than an independent benchmark.

## Short oral defense

“I detected edits through JPEG compression history. I estimated the earlier compression quality and its shifted 8 × 8 grid for each stamp, then measured where DCT coefficients stopped matching that quantization pattern. A classifier trained on the exhibit masks selected one region per page. The five overlays read SARSA. Whole-exhibit validation localized 25 of 30 labelled examples; album accuracy is unknown because album masks were not supplied.”

În română: „Am folosit urmele compresiei JPEG. Pentru fiecare timbru am estimat calitatea primei compresii și poziția grilei de blocuri de 8 × 8 pixeli. Apoi am căutat regiunile în care coeficienții DCT nu mai respectă cuantizarea inițială. Clasificatorul a fost antrenat pe măștile exemplelor, iar suprapunerea pozițiilor detectate produce SARSA. Pe exemple complet separate de antrenare, centrul detectat a fost corect în 25 din 30 de cazuri.”

## Likely questions

**Why does a changed digit affect compression evidence?** Its newly drawn pixels were not quantized in the stamp's original save. Its block coefficients can therefore fall between the old quantization multiples.

**Why search 64 phases?** The stamp can be shifted relative to the page. The first JPEG grid is not necessarily aligned with the later page grid.

**Why learn a classifier if the physics is known?** Rounding, clipping, final recompression and mixed boundary blocks make single-coefficient rules unreliable. The classifier combines several physically motivated measurements using the labelled examples.

**Did the model know the word SARSA?** No. It uses no language model, dictionary or letter templates. The word was read from the final overlays.

**Were any regions chosen manually?** No. Each box centre is selected by the same score-map maximum rule. Visual inspection was used to read the resulting letters, as the task requires.

**Why is this not catalogue matching?** Every stamp is compared with its own estimated compression pattern. No original or externally found stamp image is used.

**What is the difference between AUC and localization success?** AUC compares the ranking of positive and negative blocks. Localization success checks whether the chosen image point lies in the mask. The latter is closer to this task's output.

**Does 25/30 mean 83.3% of album boxes are correct?** No. It is measured only on held-out exhibits. Album masks are unavailable, and the two image settings can differ.

**What is SARSA?** It is the recovered five-letter word. Separately, SARSA is also the name of an on-policy reinforcement-learning method, conventionally expanded as State–Action–Reward–State–Action; that meaning was not used to detect regions or decode them.
