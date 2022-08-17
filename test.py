import matplotlib.pyplot as plt
import numpy as np
import utils
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.decomposition import PCA
import random
d = {}
a = [[[1],[2]],2,3,[4]]
d["a"] = a
for k, values in d.items():
    print(type(values))
random.shuffle(a)
print(a)

