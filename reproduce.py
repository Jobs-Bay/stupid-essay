import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
# from DDPG.bad_mapping_env import STAR_RIS_env
from proposed.STAR_RIS_env import STAR_RIS_env

os.chmod('reproduce.py', 0o777)
env = STAR_RIS_env(antenna_num=4, user_num=4, element_num=30, power_limit=30, target_num=4, eve_num=1)
#读取npy文件
def read_npy(file_path):
    data = np.load(file_path)
    return data

data1 = read_npy('proposed/数据/utility_targ_not_eve/LA=0.0001,LC=0.0001,GAMMA=0.001/M=4,K=4,N=30,P=30,T=4,F=1/rad_list_list.npy')
# data2 = read_npy('DDPG/DDPG/数据/utility_targ_not_eve/LA=0.0001,LC=0.0001,GAMMA=0.001/M=10,K=8,N=30,P=20,T=4,F=4/rad_list_list.npy')
# data3 = read_npy('DDPG/DDPG/数据/bad_mapping/LA=0.0001,LC=0.0001,GAMMA=0.001/M=10,K=8,N=30,P=35,T=4,F=4/ep_reward_list.npy')
# data4 = read_npy('DDPG/DDPG/数据/bad_mapping/LA=0.0001,LC=0.0001,GAMMA=0.001/M=10,K=8,N=30,P=40,T=4,F=4/ep_reward_list.npy')
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
    avg1.append(np.mean(temp[max(0, i-1000):i+1]))  #计算滑动窗口平均
# temp = []
# for i in range(len(data2)):
#
#     temp.append(np.sum(data2[i]))
#     avg2.append(np.mean(temp[max(0, i-1000):i+1]))  #计算滑动窗口平均
# temp = []
# for i in range(len(data3)):
#
#     temp.append(np.sum(data3[i]))
#     avg3.append(np.mean(temp[max(0, i-1000):i+1]))  #计算滑动窗口平均
# temp = []
# for i in range(len(data4)):
#
#     temp.append(np.sum(data4[i]))
#     avg4.append(np.mean(temp[max(0, i-1000):i+1]))  #计算滑动窗口平均
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
# plt.plot(avg1, label='lr_a=0.0001,lr_c=0.0002', color='blue')
# plt.plot(avg2, label='lr_a=0.001,lr_c=0.002', color='red')
# plt.plot(avg3, label='lr_a=0.00001,lr_c=0.00002', color='green')
# plt.plot(avg4, label='2.5', color='yellow')
# plt.plot(avg5, label='5', color='black')
# plt.plot(avg6, label='7.5', color='purple')
# plt.plot(avg7, label='10', color='orange')
# plt.plot(avg8, label='20', color='pink')
# plt.plot(avg9, label='15', color='brown')

# plt.xlabel('Episode')
# plt.ylabel('Average Reward')
# plt.legend()
# plt.show()

def decode_W(data):
    W_j = []
    W_real = data[19999][4 * env.element_num:4 * env.element_num + env.antenna_num * (env.antenna_num + env.user_num)]
    W_imag = data[19999][4 * env.element_num + env.antenna_num * (
                env.antenna_num + env.user_num):4 * env.element_num + env.antenna_num * (
                env.antenna_num + env.user_num) + env.antenna_num * (env.antenna_num + env.user_num)]
    W_before = W_real.reshape(env.antenna_num, env.antenna_num + env.user_num) + 1j * W_imag.reshape(
        env.antenna_num, env.antenna_num + env.user_num)

    W = env.normalize_W(W_before, np.abs(data[560439][-1]) * env.power_limit)
    # W = env.normalize_W(W_before, env.power_limit)
    for i in range(env.antenna_num + env.user_num):
        W_j.append(np.linalg.norm(W[:, i]) ** 2)
    return W_j
#[0.05523541523603342, 0.05472175921283506, 0.05510731128341555, 0.055222747644426695, 0.055116538876930736, 0.05501841179571963, 0.05477317693417103, 0.05474838360015017, 0.05490678289136646, 0.05510505990564684, 0.05500736659198873, 0.05507103268397, 0.055315108873935016, 0.0553730564332471, 0.055184487167768824, 0.05520592755966988, 0.055189472136402275, 0.05503152869133251]


# W = decode_W(data10)
# print(np.sum(W))
# print(W)

