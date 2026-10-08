# Dataset design

## Primary segmentation corpus: Refined Deep-SAR Oil Spill (SOS)

The refined release is the pixel-level segmentation corpus for AQUA-SENTINEL. It is the improved Deep-SAR SOS dataset; approximately 38% of training masks and 50% of validation masks were manually corrected. Source: Zenodo DOI 10.5281/zenodo.15298010.

The Deep-SAR family contains Sentinel-1A VV and ALOS PALSAR HH samples, but these are separate source groups rather than documented co-registered pairs. Therefore they should not be concatenated into a two-stream sample unless the exact downloaded release provides an explicit one-to-one correspondence. For the publication experiment, treat Refined Deep-SAR as a single-SAR segmentation corpus and report sensor-stratified results (PALSAR and Sentinel-1) when possible. The dual-stream implementation is retained for future genuinely paired data.

## External look-alike/object corpus: DARTIS 2019

DARTIS contains 1,365 image patches with 3,225 oil objects and 2,990 no-oil patches. It is Sentinel-1 VV imagery from the Eastern Mediterranean in 2019. Oil objects are annotated in Pascal VOC XML; no-oil patches have no pixel annotation. Therefore DARTIS is used for object-level external evaluation and look-alike rejection, not pixel Dice/IoU training.

## Optional QPOSD

QPOSD is a separate external polarimetric segmentation corpus with 41 UAVSAR L-band scenes, nine-channel coherence-matrix representation, and four pixel classes: sea surface, oil spill, look-alike, land. It is optional and is not required by the core DARTIS + Refined Deep-SAR pipeline.
