# AQUA-SENTINEL: revised DARTIS + Refined Deep-SAR implementation

**Important dataset correction:** DARTIS is not a pixel-level segmentation corpus. It contains Sentinel-1 VV image patches, oil objects in Pascal VOC XML, and no-oil/look-alike patches without pixel masks. Also, Refined Deep-SAR contains Sentinel-1/VV and PALSAR/HH samples but they are not documented as co-registered pairs. Therefore the scientifically valid configuration is:

- **Primary pixel-level segmentation:** Refined Deep-SAR Oil Spill (SOS), evaluated as a single-SAR segmentation corpus unless true paired samples are independently established.
- **External look-alike/object-level evaluation:** DARTIS 2019.
- **Optional external polarimetric segmentation:** QPOSD, which can support a genuinely multi-channel/polarimetric version of the architecture.

This preserves valid segmentation supervision without manufacturing masks from DARTIS bounding boxes. The manuscript's current optical/radar cross-modal fusion claim must be revised unless a co-registered optical/SAR or paired-SAR dataset is added.

## Architecture

The repository retains the dual-encoder implementation as a research component, but the DARTIS + Refined Deep-SAR datasets do not by themselves establish co-registered two-stream inputs. The publication-ready experiment should therefore use the single-stream SAR path on Refined Deep-SAR, while retaining the physics features, lookalike head and PIECL constraint. If a paired optical/SAR or genuinely co-registered dual-SAR dataset is later added, the Gate/dual-encoder path can be activated.

## Repository

```text
AQUA-SENTINEL/
├── configs/
│   ├── refined_deep_sar.yaml
│   └── dartis_eval.yaml
├── docs/
│   ├── DATASET.md
│   ├── EQUATIONS.md
│   ├── EXPERIMENTAL_SETUP.md
│   └── HIGHLIGHTED_REMARKS.md
├── scripts/
│   ├── download_refined_deep_sar.py
│   ├── download_dartis.md
│   ├── make_refined_manifest.py
│   └── evaluate_dartis.py
├── src/
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── physics.py
│   ├── postprocess.py
│   └── train.py
├── tests/test_metrics.py
├── requirements.txt
└── README.md
```

## 1. Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download Refined Deep-SAR

```bash
python scripts/download_refined_deep_sar.py --out data/raw/refined_deep_sar
```

This downloads the official Zenodo images.zip and masks.zip. The dataset is about 1.2 GB for the listed image archive plus masks; do not commit it to GitHub.

## 3. Obtain DARTIS

Download DARTIS from PANGAEA DOI **10.1594/PANGAEA.980773**. The official companion repository is `yi-jie-yang/dataset_DARTIS_2019`. Its official structure tool organizes images as:

```text
oil/coast/
oil/water/
no_oil/coast/c0 ...
no_oil/water/c0 ...
```

Oil annotations are Pascal VOC XML. No-oil patches have no pixel annotations.

## 4. Build the paired Refined Deep-SAR manifest

After inspecting the downloaded archive and placing the two modalities under separate directories:

```bash
python scripts/make_refined_manifest.py \
  --images data/raw/refined_deep_sar/images \
  --masks data/raw/refined_deep_sar/masks \
  --secondary data/raw/refined_deep_sar/secondary \
  --out data/processed/refined_manifest.json
```

The script deliberately requires exact matching stems. If the release uses modality-specific suffixes, edit the matching rule after inspecting the actual filenames rather than silently pairing unrelated images.

## 5. Train

```bash
python -m src.train \
  --config configs/refined_deep_sar.yaml \
  --manifest data/processed/refined_manifest.json \
  --data-root data/raw/refined_deep_sar \
  --out outputs/refined_deep_sar
```

Ablations:

```bash
python -m src.train ... --no-physics
python -m src.train ... --no-fp-head
```

## 6. DARTIS external evaluation

Generate predictions in CSV form:

```text
image,xmin,ymin,xmax,ymax,score
S1_....jpg,100,120,180,220,0.91
```

Then:

```bash
python scripts/evaluate_dartis.py \
  --pred outputs/dartis_predictions.csv \
  --xml-root data/raw/dartis \
  --iou 0.5
```

For the no-oil DARTIS subsets, report false-positive rate and precision. Do not report segmentation Dice/IoU because there is no pixel mask.

## 7. Reproduce the manuscript's threshold study

For the Refined Deep-SAR segmentation test, evaluate thresholds 0.30–0.70. Select an operating threshold using validation only. If the manuscript retains the rule “lowest validation threshold reaching 99% precision,” explicitly verify it from the new validation sweep.

## 8. Table III

Table III must now be generated for **Refined Deep-SAR**, not LADOS. The old value of 7,528 tiles must not be carried forward.

## 9. Reproducibility warnings

- Do not claim DARTIS is a segmentation dataset.
- Do not turn bounding boxes into pseudo-masks and call them ground truth.
- Do not claim dual-SAR fusion until Sentinel-1/ALOS correspondence is verified in the actual downloaded release.
- Do not report manuscript numerical results until the new training/evaluation runs reproduce them.
- For final publication, use three seeds for ablations and report mean ± SD.

## Dataset references

DARTIS: Yang, Singha, Goldman & Schütte (2025), ESSD, DOI 10.5194/essd-17-6807-2025; PANGAEA DOI 10.1594/PANGAEA.980773.

Refined Deep-SAR: Zuenko & Khaidarova (2025), Zenodo DOI 10.5281/zenodo.15298010.

QPOSD: Jamal et al. (2026), Data in Brief 66, 112773, DOI 10.1016/j.dib.2026.112773; Zenodo DOI 10.5281/zenodo.19258036.
