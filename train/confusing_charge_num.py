"""
Epoch: 17  Train_loss: 2.874128  Valid_loss: 110.183821
Charge_Acc: 0.8573  Charge_MP: 0.8173  Charge_MR: 0.8506  Charge_F1: 0.8289
Article_Acc: 0.8656  Article_MP: 0.7954   Article_MR: 0.8204 Article_F1: 0.8028
Penalty_Acc: 0.371  Penalty_MP: 0.3223  Penalty_MR: 0.3301  Penalty_F1: 0.3076
Time: 4.53min

Epoch: 19  Train_loss: 2.706189  Valid_loss: 110.501951
Charge_Acc: 0.8623  Charge_MP: 0.829  Charge_MR: 0.8501  Charge_F1: 0.8348
Article_Acc: 0.8691  Article_MP: 0.8069   Article_MR: 0.8192 Article_F1: 0.8077
Penalty_Acc: 0.3805  Penalty_MP: 0.3426  Penalty_MR: 0.3265  Penalty_F1: 0.3174
Time: 4.72min

Epoch: 37  Train_loss: 1.911496  Valid_loss: 121.241307
Charge_Acc: 0.8801  Charge_MP: 0.8511  Charge_MR: 0.8567  Charge_F1: 0.8527
Article_Acc: 0.884  Article_MP: 0.82.50   Article_MR: 0.8334 Article_F1: 0.828
Penalty_Acc: 0.398  Penalty_MP: 0.3613  Penalty_MR: 0.3525  Penalty_F1: 0.3527
Time: 4.71min

Epoch: 36  Train_loss: 1.940528  Valid_loss: 119.631664
Charge_Acc: 0.8769  Charge_MP: 0.8481  Charge_MR: 0.8503  Charge_F1: 0.8471
Article_Acc: 0.8831  Article_MP: 0.8289   Article_MR: 0.8229 Article_F1: 0.8226
Penalty_Acc: 0.39.99  Penalty_MP: 0.3506  Penalty_MR: 0.3523  Penalty_F1: 0.3451
Time: 4.54min
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

x = np.array([2, 4, 8, 16])
y_article_acc = np.array([86.56, 86.91, 88.40, 88.31])
y_article_mp = np.array([79.54, 80.69, 82.50, 82.89])
y_article_mr = np.array([82.04, 81.92, 83.29, 82.29])
y_article_f1 = np.array([80.28, 80.77, 82.80, 82.26])

y_charge_acc = np.array([85.73, 86.23, 88.01, 87.69])
y_charge_mp = np.array([81.73, 82.90, 85.11, 84.81])
y_charge_mr = np.array([85.06, 85.01, 85.67, 85.03])
y_charge_f1 = np.array([82.89, 83.48, 85.27, 84.73])
y_penalty_acc = np.array([37.10, 38.05, 39.8, 40.01])
y_penalty_mp = np.array([32.23, 34.26, 36.13, 35.06])
y_penalty_mr = np.array([33.01, 32.65, 35.25, 35.23])
y_penalty_f1 = np.array([30.76, 31.74, 35.27, 34.51])

# 绘制article 图
fig = plt.figure()
ax = plt.axes()

plt.title("Article Prediction")
plt.xlabel("confusing charge number")
ax.plot(x, y_article_acc, "-^", label="Acc")
ax.plot(x, y_article_mp, "-o", label="MP")
ax.plot(x, y_article_mr, "-s", label="MR")
ax.plot(x, y_article_f1, "-x", label="F1")
plt.axis('equal')
plt.legend()


# 绘制charge图
fig = plt.figure()
ax = plt.axes()
plt.title("Charge Prediction")
plt.xlabel("confusing charge number")
ax.plot(x, y_charge_acc, "-^", label="Acc")
ax.plot(x, y_charge_mp, "-o", label="MP")
ax.plot(x, y_charge_mr, "-s", label="MR")
ax.plot(x, y_charge_f1, "-x", label="F1")
plt.axis('equal')
plt.legend()


# 绘制penalty图
fig = plt.figure()
ax = plt.axes()
plt.title("Term of Penalty Prediction")
plt.xlabel("confusing charge number")
ax.plot(x, y_penalty_acc, "-^", label="Acc")
ax.plot(x, y_penalty_mp, "-o", label="MP")
ax.plot(x, y_penalty_mr, "-s", label="MR")
ax.plot(x, y_penalty_f1, "-x", label="F1")
plt.axis('equal')
plt.legend()


plt.show()

