# DARTIS acquisition

DARTIS is distributed through PANGAEA under DOI 10.1594/PANGAEA.980773. The official companion code is:
https://github.com/yi-jie-yang/dataset_DARTIS_2019

Download the dataset from PANGAEA, then run the official `structure_dataset.py` if desired. DARTIS contains `oil/` and `no_oil/` subsets; oil annotations are Pascal VOC XML object annotations. The no-oil subset has no pixel-level masks. Do **not** convert bounding boxes into segmentation ground truth for pixel Dice/IoU.
