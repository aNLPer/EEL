import matplotlib.pyplot as plt
import numpy as np
import utils
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.decomposition import PCA

y_pred = [[0.2,0.5,0.1,0.2],
          [0.8,0.1,0.05,0.05],
          [0.2,0.5,0.1,0.2],
          [0.1,0.1,0.1,0.7],
          [0.8,0.1,0.05,0.05],
          [0.1,0.1,0.7,0.1],
          [0.2,0.5,0.1,0.2],
          [0.1,0.1,0.1,0.7],
          [0.1,0.1,0.7,0.1],
          [0.1,0.1,0.7,0.1],
          [0.2,0.5,0.1,0.2],
          [0.1,0.1,0.1,0.7],
          [0.8,0.1,0.05,0.05],
          [0.1,0.1,0.7,0.1],
          [0.2,0.5,0.1,0.2]]
y_pred = np.array(y_pred)

# y_pred = [1,0,1,3,0,2,1,3,2,2,1,3,0,2,1]
y = [1,2,0,3,0,1,2,3,3,1,0,2,3,0,1]
y = np.array(y)

cm = utils.ConfusionMatrix(4)
cm.updateMat(y_pred, y)
print(cm.getMaF())
print(cm.getMaP())
print(cm.getMaR())
print(cm.get_acc())


y_p = np.argmax(y_pred, axis=1).flatten()
y_ = y.flatten()

print(f1_score(y_,y_p,average="macro"))
print(precision_score(y_, y_p, average="macro"))
print(recall_score(y_, y_p, average="macro"))
print(accuracy_score(y_, y_p))

