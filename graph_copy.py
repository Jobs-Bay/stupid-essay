import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据定义
x_points1 = [-12 + i * 2 for i in range(9)]
y_proposed1 = [4.675278418, 5.017769462, 5.340737685, 5.586147343, 5.784663468, 5.940822007, 6.056219085, 6.140995036, 6.20210169]
y_bm1 = [4.138271748, 4.442887689, 4.695218763, 4.901220961, 5.071297742, 5.181879491, 5.291725215, 5.361351628, 5.444826884]
y_wo1 = [4.116100385, 4.300529248, 4.432047816, 4.502004776, 4.546372864, 4.575684076, 4.596106367, 4.612250468, 4.626562452]
y_ddpg1 = [4.107794891, 4.351880748, 4.545978784, 4.694634407, 4.804176646, 4.88297857, 4.939917154, 4.983326374, 5.020641828]

# 最大增幅百分比标识计算函数
def find_max_increase(proposed, comparison, x_points):
    increases = [(p - c) / c * 100 for p, c in zip(proposed, comparison)]
    max_index = increases.index(max(increases))
    return x_points[max_index], proposed[max_index], comparison[max_index], max(increases)

# 定义箭头样式
arrow_styles = [{'arrowstyle': '<->', 'color': 'orange'},
                {'arrowstyle': '<->', 'color': 'green'},
                {'arrowstyle': '<->', 'color': 'red'}]

# 绘制带最大增幅标识的图
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x_points1, y_proposed1, label='所提方案', marker='o')
ax1.plot(x_points1, y_ddpg1, label='对比方案1', marker='s')
ax1.plot(x_points1, y_bm1, label='对比方案2', marker='x')
ax1.plot(x_points1, y_wo1, label='对比方案3', marker='*')

# 标出最大增幅百分比并用不同箭头样式
for y_data, label, style in zip([y_ddpg1, y_bm1, y_wo1], ['对比方案1', '对比方案2', '对比方案3'], arrow_styles):
    x_max, y_max_proposed, y_max_comp, increase = find_max_increase(y_proposed1, y_data, x_points1)
    ax1.annotate('', xy=(x_max, y_max_proposed), xytext=(x_max, y_max_comp),
                 arrowprops=dict(**style))
    ax1.text(x_max, (y_max_proposed + y_max_comp) / 2, f'{label}增幅: {increase:.2f}%', ha='right')

ax1.set_xticks(x_points1)
ax1.set_xlabel('基站发射功率 (dBm)')
ax1.set_ylabel('系统效用')
ax1.legend()
plt.grid(False)
plt.show()

# 系统和安全速率图，并添加所有对比方案和增幅标注
y_proposed1r = [2.348846181, 2.827549999, 3.369007577, 3.831405028, 4.236410918, 4.564180437, 4.805430306, 4.980180731, 5.097815561]
y_bm1r = [2.028642799, 2.463185512, 2.818244237, 3.096908775, 3.341086213, 3.529330983, 3.725263518, 3.854704227, 3.993953652]
y_wo1r = [1.152003181, 1.294977764, 1.402267938, 1.468376527, 1.512484541, 1.541868307, 1.561470334, 1.575206863, 1.585153479]
y_ddpg1r = [1.85763898, 2.19691243, 2.492612135, 2.734897849, 2.922249135, 3.061477972, 3.164008563, 3.242625773, 3.309840902]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_points1, y_proposed1r, label='所提方案', marker='o')
ax.plot(x_points1, y_ddpg1r, label='对比方案1', marker='s')
ax.plot(x_points1, y_bm1r, label='对比方案2', marker='x')
ax.plot(x_points1, y_wo1r, label='对比方案3', marker='*')

# 添加阈值线 y = 0.4
ax.axhline(y=0.4, color='r', linestyle='--', label='安全阈值')

# 标出每个对比方案与所提方案的最大增幅百分比并用不同箭头样式
for y_data, label, style in zip([y_ddpg1r, y_bm1r, y_wo1r], ['对比方案1', '对比方案2', '对比方案3'], arrow_styles):
    x_max, y_max_proposed, y_max_comp, increase = find_max_increase(y_proposed1r, y_data, x_points1)
    ax.annotate('', xy=(x_max, y_max_proposed), xytext=(x_max, y_max_comp),
                arrowprops=dict(**style))
    ax.text(x_max, (y_max_proposed + y_max_comp) / 2, f'{label}增幅: {increase:.2f}%', ha='right')

ax.set_xticks(x_points1)
ax.set_xlabel('基站发射功率 (dBm)')
ax.set_ylabel('系统和安全速率(bit/s/Hz)')
ax.legend()
plt.grid(False)
plt.show()

# STAR-RIS单元个数图
x_points2 = [10 + i * 10 for i in range(4)]
y_proposed2 = [5.032294332, 5.498373219, 6.20210169, 6.42107105]
y_ddpg2 = [4.21122605, 4.566922197, 5.020641828, 5.242149502737958]
y_wo2 = [3.73122303, 4.044242564, 4.626562452, 4.802465013]
y_bm2 = [4.731103781, 4.935822848, 5.362418225, 5.653988832]

# 绘制STAR-RIS单元个数图，标出最大增幅百分比并用不同箭头样式
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_points2, y_proposed2, label='所提方案', marker='o')
ax.plot(x_points2, y_ddpg2, label='对比方案1', marker='s')
ax.plot(x_points2, y_bm2, label='对比方案2', marker='x')
ax.plot(x_points2, y_wo2, label='对比方案3', marker='*')

# 标出最大增幅
for y_data, label, style in zip([y_ddpg2, y_bm2, y_wo2], ['对比方案1', '对比方案2', '对比方案3'], arrow_styles):
    x_max, y_max_proposed, y_max_comp, increase = find_max_increase(y_proposed2, y_data, x_points2)
    ax.annotate('', xy=(x_max, y_max_proposed), xytext=(x_max, y_max_comp),
                arrowprops=dict(**style))
    ax.text(x_max, (y_max_proposed + y_max_comp) / 2, f'{label}增幅: {increase:.2f}%', ha='right')

ax.set_xticks(x_points2)
ax.set_xlabel('STAR-RIS单元个数')
ax.set_ylabel('系统效用')
ax.legend()
plt.grid(False)
plt.show()

# 感知目标个数图
x_points3 = [2 + i * 2 for i in range(5)]
y_proposed3 = [3.966880948, 5.032294332, 6.615313112, 8.281715925, 9.281630418]
y_ddpg3 = [3.253151643, 4.21122605, 5.627602855, 7.388409602, 8.420011985]
y_wo3 = [3.211521951, 3.73122303, 4.179145126, 4.64860605, 5.06874446]
y_bm3 = [3.525144216, 4.731103781, 5.926185913, 7.59543257, 8.89142234]

# 绘制感知目标个数图，标出最大增幅百分比并用不同箭头样式
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_points3, y_proposed3, label='所提方案', marker='o')
ax.plot(x_points3, y_ddpg3, label='对比方案1', marker='s')
ax.plot(x_points3, y_bm3, label='对比方案2', marker='x')
ax.plot(x_points3, y_wo3, label='对比方案3', marker='*')

# 标出最大增幅
for y_data, label, style in zip([y_ddpg3, y_bm3, y_wo3], ['对比方案1', '对比方案2', '对比方案3'], arrow_styles):
    x_max, y_max_proposed, y_max_comp, increase = find_max_increase(y_proposed3, y_data, x_points3)
    ax.annotate('', xy=(x_max, y_max_proposed), xytext=(x_max, y_max_comp),
                arrowprops=dict(**style))
    ax.text(x_max+1, (y_max_proposed + y_max_comp) / 2.1, f'{label}增幅: {increase:.2f}%', ha='right')

ax.set_xticks(x_points3)
ax.set_xlabel('感知目标个数')
ax.set_ylabel('系统效用')
ax.legend()
plt.grid(False)
plt.show()

