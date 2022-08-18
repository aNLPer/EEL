import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

vecArr = np.random.rand(1000,512) #构造1000个512维向量, 换成实际需要聚类的向量即可
tsneData = TSNE().fit_transform(vecArr)

#开始进行可视化
f = plt.figure(figsize=(10,10))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(tsneData[:,0], tsneData[:,1])
plt.xlim(-50,50)
plt.ylim(-50,50)
ax.axis('off')
ax.axis('tight')
plt.show()