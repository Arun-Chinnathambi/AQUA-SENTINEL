# Author-response remarks after replacing LADOS

1. **Dataset acquisition sentence:** Replace the former LADOS acquisition statement. Refined Deep-SAR is the segmentation dataset; DARTIS is the external Sentinel-1 look-alike/object dataset.
2. **Table III:** Recompute all counts from the actual Refined Deep-SAR manifest. Do not carry over LADOS counts.
3. **Empty-empty Dice/IoU:** Use the corrected convention: both empty => 1.
4. **Threshold 0.50:** Verify on the validation set only using the stated rule; never choose it from test data.
5. **No-physics / no-FP-head:** Retrain under the same split and schedule; preferably three seeds and mean ± SD.
6. **U-Net / DeepLabv3+:** Retrain on the same Refined Deep-SAR split if they are claimed as baselines.
7. **DARTIS limitation:** It is object-level, so it cannot support pixel Dice/IoU without additional pixel annotation. Do not rasterize VOC boxes and call them ground truth masks.
8. **Dual-encoder wording:** Refined Deep-SAR contains Sentinel-1/VV and PALSAR/HH sample groups, but a co-registered pair is not established by the dataset description. Do not claim dual-SAR fusion from simply combining the two folders. Either revise the architecture to a single-stream SAR model for this experiment or add a genuinely paired dataset.
