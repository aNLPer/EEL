"""
Epoch: 26  Train_loss: 2.3413  Valid_loss: 112.927841
Charge_Acc: 0.8715  Charge_MP: 0.8358  Charge_MR: 0.8546  Charge_F1: 0.8425
Article_Acc: 0.8772  Article_MP: 0.8187   Article_MR: 0.8263 Article_F1: 0.819
Penalty_Acc: 0.3909  Penalty_MP: 0.3489  Penalty_MR: 0.3306  Penalty_F1: 0.3275
Time: 4.52min

Epoch: 33  Train_loss: 2.107505  Valid_loss: 115.204936
Charge_Acc: 0.8766  Charge_MP: 0.8484  Charge_MR: 0.8552  Charge_F1: 0.85
Article_Acc: 0.8835  Article_MP: 0.8291   Article_MR: 0.8265 Article_F1: 0.8257
Penalty_Acc: 0.3911  Penalty_MP: 0.351  Penalty_MR: 0.3434  Penalty_F1: 0.3399
Time: 4.73min

Epoch: 37  Train_loss: 1.911496  Valid_loss: 121.241307
Charge_Acc: 0.8801  Charge_MP: 0.8572  Charge_MR: 0.8567  Charge_F1: 0.8527
Article_Acc: 0.884  Article_MP: 0.8250   Article_MR: 0.8334 Article_F1: 0.828
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
y_article_acc = np.array([87.72, 88.35, 88.40, 88.31])
y_article_mp = np.array([81.87, 82.91, 82.50, 82.89])
y_article_mr = np.array([82.63, 82.65, 83.34, 82.29])
y_article_f1 = np.array([81.90, 82.57, 82.80, 82.26])

y_charge_acc = np.array([87.15, 87.66, 88.01, 87.69])
y_charge_mp = np.array([83.58, 84.84, 85.72, 84.81])
y_charge_mr = np.array([85.46, 85.52, 85.67, 85.03])
y_charge_f1 = np.array([84.25, 85.00, 85.27, 84.73])

y_penalty_acc = np.array([39.09, 39.11, 39.8, 39.99])
y_penalty_mp = np.array([34.89, 35.10, 36.13, 35.06])
y_penalty_mr = np.array([33.06, 34.34, 35.25, 35.23])
y_penalty_f1 = np.array([32.76, 33.99, 35.27, 34.51])

# 绘制article 图
fig = plt.figure()
ax = plt.axes()
ax.patch.set_facecolor('#ececf6')
ax.spines[['right','top','left','bottom']].set_color('none')
plt.grid(color='w', linestyle='solid',  linewidth="1")
plt.title("Article Prediction")
plt.xlabel("confusing charge number")

ax.plot(x, y_article_acc/100, "-^", label="Acc")
ax.plot(x, y_article_mp/100, "-o", label="MP")
ax.plot(x, y_article_mr/100, "-s", label="MR")
ax.plot(x, y_article_f1/100, "-x", label="F1")
# plt.tight_layout()
plt.legend()
plt.savefig('image_article.png')


# 绘制charge图
fig = plt.figure()
ax = plt.axes()
ax.patch.set_facecolor('#ececf6')
ax.spines[['right','top','left','bottom']].set_color('none')
plt.grid(color='w', linestyle='solid',  linewidth="1")
plt.title("Charge Prediction")
plt.xlabel("confusing charge number")

ax.plot(x, y_charge_acc/100, "-^", label="Acc")
ax.plot(x, y_charge_mp/100, "-o", label="MP")
ax.plot(x, y_charge_mr/100, "-s", label="MR")
ax.plot(x, y_charge_f1/100, "-x", label="F1")
plt.legend()
plt.savefig('image_charge.png')

# 绘制penalty图
fig = plt.figure()
ax = plt.axes()
ax.patch.set_facecolor('#ececf6')
ax.spines[['right','top','left','bottom']].set_color('none')
plt.grid(color='w', linestyle='solid',  linewidth="1")
plt.title("Term of Penalty Prediction")
plt.xlabel("confusing charge number")

ax.plot(x, y_penalty_acc/100, "-^", label="Acc")
ax.plot(x, y_penalty_mp/100, "-o", label="MP")
ax.plot(x, y_penalty_mr/100, "-s", label="MR")
ax.plot(x, y_penalty_f1/100, "-x", label="F1")
plt.legend()
plt.savefig('image_penalty.png')

plt.show()

