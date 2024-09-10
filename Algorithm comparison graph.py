import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 功率
x_points1 = [-12 + i * 2 for i in range(9)]
#左侧数据
y_proposed1 = [
4.675278418,
5.017769462,
5.340737685,
5.586147343,
5.784663468,
5.940822007,
6.056219085,
6.140995036,
6.20210169,
]
y_bm1 = [
4.138271748,
4.442887689,
4.695218763,
4.901220961,
5.071297742,
5.181879491,
5.291725215,
5.361351628,
5.444826884
]

y_wo1 = [
4.116100385,
4.300529248,
4.432047816,
4.502004776,
4.546372864,
4.575684076,
4.596106367,
4.612250468,
4.626562452,
]
y_ddpg1 = [
4.107794891,
4.351880748,
4.545978784,
4.694634407,
4.804176646,
4.88297857,
4.939917154,
4.983326374,
5.020641828,
]
#右侧数据
y_proposed1r = [
2.348846181,
2.827549999,
3.369007577,
3.831405028,
4.236410918,
4.564180437,
4.805430306,
4.980180731,
5.097815561
]
y_bm1r = [
2.028642799,
2.463185512,
2.818244237,
3.096908775,
3.341086213,
3.529330983,
3.725263518,
3.854704227,
3.993953652
]

y_wo1r = [
1.152003181,
1.294977764,
1.402267938,
1.468376527,
1.512484541,
1.541868307,
1.561470334,
1.575206863,
1.585153479
]
y_ddpg1r = [
1.85763898,
2.19691243,
2.492612135,
2.734897849,
2.922249135,
3.061477972,
3.164008563,
3.242625773,
3.309840902
]
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x_points1, y_proposed1, label='所提方案', marker='o')
ax1.plot(x_points1, y_ddpg1, label='对比方案1', marker='s')
ax1.plot(x_points1, y_bm1, label='对比方案2', marker='x')
ax1.plot(x_points1, y_wo1, label='对比方案3', marker='*')
ax1.set_xticks(x_points1)
ax1.grid(False)
ax1.set_xlabel('基站发射功率 (dBm)')
ax1.set_ylabel('系统效用')

ax2 = ax1.twinx()
ax2.plot(x_points1, y_proposed1r, label='所提方案', marker='o',linestyle='--')
ax2.plot(x_points1, y_ddpg1r, label='对比方案1', marker='s',linestyle='--')
ax2.plot(x_points1, y_bm1r, label='对比方案2', marker='x',linestyle='--')
ax2.plot(x_points1, y_wo1r, label='对比方案3', marker='*',linestyle='--')
ax2.grid(False)
ax2.set_ylabel('安全和速率(bit/s/Hz)')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

#效用单独图
plt.figure(figsize=(8, 6))
plt.plot(x_points1, y_proposed1, label='所提方案', marker='o')
plt.plot(x_points1, y_ddpg1, label='对比方案1', marker='s')
plt.plot(x_points1, y_bm1, label='对比方案2', marker='x')
plt.plot(x_points1, y_wo1, label='对比方案3', marker='*')
plt.xticks(x_points1)
plt.xlabel('基站发射功率 (dBm)')
plt.ylabel('系统效用')
plt.legend()
plt.grid(False)
plt.show()

#系统和安全速率单独图
plt.figure(figsize=(8, 6))
plt.plot(x_points1, y_proposed1r, label='所提方案', marker='o')
plt.plot(x_points1, y_ddpg1r, label='对比方案1', marker='s')
plt.plot(x_points1, y_bm1r, label='对比方案2', marker='x')
plt.plot(x_points1, y_wo1r, label='对比方案3', marker='*')
plt.xticks(x_points1)
plt.xlabel('基站发射功率 (dBm)')
plt.ylabel('系统和安全速率(bit/s/Hz)')
plt.legend()
plt.grid(False)
plt.show()


# 单元数
x_points2 = [10 + i * 10 for i in range(4)]
y_proposed2 = [5.032294332, 5.498373219, 6.20210169, 6.42107105]
y_ddpg2 = [4.21122605, 4.566922197, 5.020641828, 5.242149502737958]
y_wo2 = [3.73122303, 4.044242564, 4.626562452, 4.802465013]
y_bm2=[4.731103781, 4.935822848, 5.362418225, 5.653988832]
plt.figure(figsize=(8, 6))
plt.plot(x_points2, y_proposed2, label='所提方案', marker='o')
plt.plot(x_points2, y_ddpg2, label='对比方案1', marker='s')
plt.plot(x_points2, y_bm2, label='对比方案2', marker='x')
plt.plot(x_points2, y_wo2, label='对比方案3', marker='*')
plt.xticks(x_points2)
plt.xlabel('STAR-RIS单元个数')
plt.ylabel('系统效用')
plt.legend()
plt.grid(False)
plt.show()

# 感知目标数
x_points3 = [2 + i * 2 for i in range(5)]
y_proposed3 = [3.966880948, 5.032294332, 6.615313112, 8.281715925, 9.281630418]
y_ddpg3=[3.253151643, 4.21122605, 5.627602855, 7.388409602, 8.420011985]
y_wo3=[3.211521951, 3.73122303, 4.179145126, 4.64860605, 5.06874446]
y_bm3=[3.525144216, 4.731103781, 5.926185913, 7.59543257, 8.89142234]

plt.figure(figsize=(8, 6))
plt.plot(x_points3, y_proposed3, label='所提方案', marker='o')
plt.plot(x_points3, y_ddpg3, label='对比方案1', marker='s')
plt.plot(x_points3, y_bm3, label='对比方案2', marker='x')
plt.plot(x_points3, y_wo3, label='对比方案3', marker='*')
plt.xticks(x_points3)
plt.xlabel('感知目标个数')
plt.ylabel('系统效用')
plt.legend()
plt.grid(False)
plt.show()
