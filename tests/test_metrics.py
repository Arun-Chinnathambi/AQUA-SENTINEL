import numpy as np
from src.metrics import per_image_overlap

def test_empty_empty_is_one():
 d,i=per_image_overlap(np.zeros((1,4,4),dtype=np.uint8),np.zeros((1,4,4),dtype=np.uint8)); assert d==1 and i==1
