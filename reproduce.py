import os
import shutil
from brokenaxes import brokenaxes
import numpy as np
import matplotlib.pyplot as plt
# from DDPG.bad_mapping_env import STAR_RIS_env
from proposed.STAR_RIS_env import STAR_RIS_env
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
os.chmod('reproduce.py', 0o777)
env = STAR_RIS_env(antenna_num=4, user_num=4, element_num=30, power_limit=30, target_num=4, eve_num=1)
#读取npy文件
def read_npy(file_path):
    data = np.load(file_path)
    return data

data1 = read_npy('ddpg/数据/decay_random/LA=1e-05,LC=1e-05,GAMMA=0.5/M=4,K=4,N=30,P=0,T=4,F=1/ep_reward_list.npy')
data2 = read_npy('proposed/数据/decay_random/x=25/LA=1e-05,LC=1e-05,GAMMA=0.5/M=4,K=4,N=30,P=0,T=4,F=1/ep_reward_list.npy')
data3 = read_npy('proposed/数据/decay_random/LA=1e-06,LC=1e-06,GAMMA=0.5/M=4,K=4,N=30,P=0,T=4,F=1/ep_reward_list.npy')
data4 = read_npy('proposed/数据/decay_random/x=25/LA=1e-05,LC=1e-05,GAMMA=0.5/M=4,K=4,N=30,P=0,T=4,F=1/ep_reward_list.npy')
# data5 = read_npy('DDPG/DDPG/数据/bad_mapping/LA=0.0001,LC=0.0002,GAMMA=0.001/M=10,K=8,N=30,P=40,T=4,F=4/ep_reward_list.npy')
# data6 = read_npy('DDPG/DDPG/数据/bad_mapping/LA=0.0001,LC=0.0002,GAMMA=0.001/M=10,K=8,N=30,P=10,T=4,F=4/ep_reward_list.npy')
# data7 = read_npy('DDPG/DDPG/数据/bad_mapping/LA=0.0001,LC=0.0002,GAMMA=0.001/M=10,K=8,N=30,P=30,T=4,F=4/ep_reward_list.npy')
# data8 = read_npy('DDPG/DDPG/数据/bad_mapping/LA=0.0001,LC=0.0002,GAMMA=0.001/M=10,K=8,N=30,P=40,T=4,F=4/ep_reward_list.npy')
# data9 = read_npy('DDPG/DDPG/数据/bad_mapping/LA=0.0001,LC=0.0002,GAMMA=0.001/M=10,K=8,N=30,P=30,T=4,F=4/a_list.npy')
# data10 = read_npy('DDPG/DDPG/数据/utility_targ_not_eve/LA=0.0001,LC=0.0002,GAMMA=0.001/M=10,K=8,N=30,P=30,T=4,F=4/a_list.npy')
temp = []
avg1 = []
avg2 = []
avg3 = []
avg4 = []
avg5 = []
avg6 = []
avg7 = []
avg8 = []
avg9 = []
# for i in range(len(data1)):
#     # avg1.append(np.sum(data1[i]))
#     avg1.append(np.mean(data1[max(0, i-1000):i+1]))  #计算滑动窗口平均
#
# for i in range(len(data2)):
#     avg2.append(np.mean(data2[max(0, i-1000):i+1]))  #计算滑动窗口平均
#
# for i in range(len(data3)):
#     avg3.append(np.mean(data3[max(0, i-1000):i+1]))  #计算滑动窗口平均

# for i in range(len(data4)):
#     avg4.append(np.mean(data4[max(0, i-1000):i+1]))  #计算滑动窗口平均
#
# for i in range(len(data5)):
#     avg5.append(np.mean(data5[max(0, i-1000):i+1]))  #计算滑动窗口平均
#
# for i in range(len(data6)):
#     avg6.append(np.mean(data6[max(0, i-1000):i+1]))  #计算滑动窗口平均
#
# for i in range(len(data7)):
#     avg7.append(np.mean(data7[max(0, i-1000):i+1]))  #计算滑动窗口平均
#
# for i in range(len(data8)):
#     avg8.append(np.mean(data8[max(0, i-1000):i+1]))  #计算滑动窗口平均
#
# for i in range(len(data9)):
#     avg9.append(np.mean(data9[max(0, i-1000):i+1]))  #计算滑动窗口平均
#
# temp = max(avg1)
# print(avg1[999])
# temp = max(avg2)
# print(avg2[999])
# temp = max(avg3)
# print(avg3[999])
# temp = max(avg4)
# print(avg4[999])
# temp = max(avg5)
# print(avg5[999])
# temp = max(avg6)
# print(avg6[999])
# temp = max(avg7)
# print(avg7[999])
# temp = max(avg8)
# print(avg8[999])
# temp = max(avg9)
# print(avg9[999])



for i in range(len(data1)):
    # avg1.append(np.sum(data1[i]))

    temp.append(np.sum(data1[i]))
    avg1.append(np.mean(temp[max(0, i-100):i+1]))  #计算滑动窗口平均
temp = []
for i in range(len(data2)):

    temp.append(np.sum(data2[i]))
    avg2.append(np.mean(temp[max(0, i-100):i+1]))  #计算滑动窗口平均
temp = []
for i in range(len(data3)):

    temp.append(np.sum(data3[i]))
    avg3.append(np.mean(temp[max(0, i-100):i+1]))  #计算滑动窗口平均
temp = []
for i in range(len(data4)):

    temp.append(np.sum(data4[i]))
    avg4.append(np.mean(temp[max(0, i-100):i+1]))  #计算滑动窗口平均
# temp = []
# for i in range(len(data5)):
#     temp.append(np.sum(data5[i]))
#     avg5.append(np.mean(temp[max(0, i - 1000):i + 1]))  # 计算滑动窗口平均
#
# temp = []
# for i in range(len(data6)):
#     temp.append(np.sum(data6[i]))
#     avg6.append(np.mean(temp[max(0, i - 1000):i + 1]))  # 计算滑动窗口平均
#
# temp = []
# for i in range(len(data7)):
#     temp.append(np.sum(data7[i]))
#     avg7.append(np.mean(temp[max(0, i - 1000):i + 1]))  # 计算滑动窗口平均
#
# temp = []
# for i in range(len(data8)):
#     temp.append(np.sum(data8[i]))
#     avg8.append(np.mean(temp[max(0, i - 1000):i + 1]))  # 计算滑动窗口平均

# temp = []
# for i in range(len(data9)):
#     temp.append(np.sum(data9[i]))
#     avg9.append(np.mean(temp[max(0, i - 1000):i + 1]))  # 计算滑动窗口平均

# for i in range(len(data6)):
#     avg6.append(np.mean(data6[max(0, i-1000):i+1]))  #计算滑动窗口平均
# print(avg1)
temp = max(avg1)
print(avg1[999])
# temp = max(avg2)
# print(avg2[999])
# temp = max(avg3)
# print(avg3[999])
# temp = max(avg4)
# print(avg4[999])
# temp = max(avg5)
# print(avg5[999])
# temp = max(avg6)
# print(avg6[999])
# temp = max(avg7)
# print(avg7[999])
# temp = max(avg8)
# print(avg8[999])
# temp = max(avg9)
# print(avg9[999])
#画图
plt.figure(figsize=(8, 6))
# plt.plot(avg1, label='x=20', color='blue')
# plt.plot(avg2, label='x=25', color='red')
# plt.plot(avg3, label='x=30', color='green')
bax = brokenaxes(
                 ylims=((0, 2000), (3000, 16000)),  # 设置y轴裂口范围
                 hspace=0.1,  # y轴裂口宽度
                 despine=True,  # 是否y轴只显示一个裂口
                 diag_color='r',  # 裂口斜线颜色
                 )
bax.plot(avg1, label='所提方案去掉自适应动作噪声', color='blue')
bax.plot(avg2, label='所提方案', color='red')
# bax.plot(avg3, label='学习率=0.000001', color='green')
# bax.plot()
# plt.plot(avg4, label='x=100', color='black')
# plt.plot(avg5, label='5', color='black')
# plt.plot(avg6, label='7.5', color='purple')
# plt.plot(avg7, label='10', color='orange')
# plt.plot(avg8, label='20', color='pink')
# plt.plot(avg9, label='15', color='brown')
# plt.xlabel('回合数')
# plt.ylabel('平均奖励')
bax.legend(loc=3)
bax.set_xlabel('回合数')
bax.set_ylabel('累计奖励')
# plt.legend()
plt.show()





