# Replacement manuscript text

## II-E. Datasets

To evaluate AQUA-SENTINEL under realistic synthetic-aperture-radar conditions, the study replaces the former optical LADOS corpus with two complementary SAR datasets. The **Refined Deep-SAR Oil Spill (SOS)** dataset is used as the primary pixel-level segmentation corpus. The released refinement corrects segmentation masks with significant annotation errors; approximately 38% of training masks and 50% of validation masks were manually revised. The underlying SOS collection contains 8,070 256×256 labelled samples derived from Sentinel-1A VV and ALOS PALSAR HH imagery. The Sentinel-1 and PALSAR samples originate from different source groups and are therefore treated as separate SAR observations rather than artificially paired modalities.

The **DARTIS 2019** dataset is used exclusively for external look-alike and object-level evaluation. DARTIS contains 1,365 Sentinel-1 image patches containing 3,225 annotated oil objects and 2,990 no-oil patches containing look-alikes or other remarkable SAR signatures. Oil objects are provided as Pascal VOC annotations, while the no-oil patches do not have pixel-level oil masks. Consequently, DARTIS is not converted into pseudo-segmentation masks and is not used to compute pixel Dice or IoU.

This two-dataset design separates the two central evaluation requirements of the proposed system: precise pixel-level delineation on a curated segmentation corpus and robustness to oil-like false alarms on an independent SAR look-alike corpus.

## II-F. Preprocessing

For Refined Deep-SAR, the released 256×256 SAR images and their corresponding binary masks are used directly after deterministic intensity normalization. Each image is converted to a single floating-point backscatter channel. For numerical stability, the input is normalized using the 1st and 99th intensity percentiles and clipped to [0,1]. Masks are resized with nearest-neighbour interpolation when required and converted to binary values.

The physics-guided branch derives two auxiliary maps from the normalized SAR intensity. The first is an adaptive logarithmic intensity map,

P₁(x)=c log(1+I(x)),  c=255/log(1+Imax),

and the second is the normalized magnitude of the discrete Laplacian,

P₂(x)=|∇²I(x)|/maxᵤ|∇²I(u)|.

The two maps are concatenated with the SAR representation through 1×1 projections. No synthetic optical image is generated from SAR data and no DARTIS bounding box is rasterized as a pixel-level ground truth mask.

For DARTIS, the released Sentinel-1 VV image patches are retained with their original object annotations. The oil subset is used for object-level localization evaluation, whereas the no-oil subset is used to quantify false alarms caused by look-alike and other remarkable SAR signatures.

## III. Experimental Setup

All segmentation experiments use a fixed random seed of 42. The primary Refined Deep-SAR experiment follows the training recipe specified for AQUA-SENTINEL: Adam optimizer with an initial learning rate of 10⁻⁴, batch size 4, 35 epochs, global gradient-norm clipping at 1.0, and ReduceLROnPlateau with factor 0.5 and patience 3. During epochs 0–9 the encoder is frozen and only the newly introduced physics projections, fusion/constraint components, decoder and auxiliary head are optimized; from epoch 10 onward all trainable layers are fine-tuned.

The segmentation objective is

Lseg=0.5LDice+0.3LBCE+0.2Lfocal,

L=Lseg+0.3Lfp,

where the focal-loss parameters are α=0.8 and γ=2. The auxiliary frame-level target is one for an empty reference mask and zero otherwise.

Because DARTIS is object-annotated rather than pixel-annotated, its external evaluation uses object IoU, precision, recall and false-positive rate rather than pixel Dice/IoU. A detection is considered a true positive when its IoU with a reference oil object reaches the predefined object-IoU threshold. No-oil DARTIS patches provide a direct test of look-alike rejection.

## III-A. Table III — Dataset composition and split

| Dataset | Role | Unit | Train | Validation | Test / External |
|---|---|---|---:|---:|---:|
| Refined Deep-SAR | Pixel segmentation | image-mask pair | **RUN SCRIPT** | **RUN SCRIPT** | **RUN SCRIPT** |
| DARTIS 2019 | Look-alike/object evaluation | image patch/object | not used | not used | **2,990 no-oil + 1,365 oil patches** |

The final Refined Deep-SAR counts must be generated from the exact downloaded release and final split manifest. Counts from the previous LADOS experiment, including 7,528 tiles, must not be transferred to this table.

## III-B. Evaluation metrics

For Refined Deep-SAR, pooled precision, recall and F1 are computed from pixel-level TP, FP and FN. Per-image Dice and IoU are then averaged over the evaluation set. When both the reference and prediction are empty, Dice and IoU are defined as 1.0. This convention is applied consistently to the validation and test results.

For DARTIS, object-level precision, recall and F1 are calculated from matched oil detections, while the no-oil subset is used to report false-positive behaviour. This evaluation is intentionally complementary to pixel segmentation metrics.

## III-C. Deployment constraint

The post-inference environmental constraint is retained as a deployment operator rather than as a source of training labels. A plausibility field is calculated as

Φ(x)=W(x)T(x),

where W(x) is normalized distance-transform water membership and T(x) is the complement of the normalized multi-scale Laplacian response. The final confidence-dependent threshold is

θ(γ)=θmax−Δθγ,

where γ=1−ρ, θmax=0.60, θmin=0.20 and Δθ=0.40. The constrained mask is accepted only where the binary network prediction is positive and Φ(x) exceeds the confidence-adaptive threshold.

## Critical wording change

The original phrase “optical and radar streams” should **not** be retained when the experiments use only Refined Deep-SAR and DARTIS. Both are SAR datasets. A genuine optical–SAR or dual-SAR cross-modal experiment requires co-registered paired observations. Until such a dataset is added, the paper should describe the tested model as a physics-guided SAR segmentation framework and present the dual-encoder module as an extensible architecture rather than as a demonstrated cross-modal fusion experiment.
