"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.14.0
gym 0.15.3
"""
import os

import shutil
import tensorflow as tf
import numpy as np
from numpy import ndarray
import time
import matplotlib.pyplot as plt
from STAR_RIS_env import STAR_RIS_env

#####################  hyper parameters  ####################
MAX_EPISODES = 1000
# MAX_EPISODES = 50000

LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
GAMMA = 0.5  # optimal reward discount
# GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 10000 #可以变大一点  滑动窗口平均 列表数据做一下滑动窗口平均
BATCH_SIZE = 64
OUTPUT_GRAPH = False


###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.actor_eval = self._build_a(name='Actor/eval')
        self.actor_target = self._build_a(name='Actor/target')
        self.critic_eval = self._build_c(name='Critic/eval')
        self.critic_target = self._build_c(name='Critic/target')

        self.actor_eval_optimizer = tf.keras.optimizers.Adam(self.LR_A)
        self.critic_eval_optimizer = tf.keras.optimizers.Adam(self.LR_C)

        self.update_target_network(self.actor_target, self.actor_eval, 1.0)
        self.update_target_network(self.critic_target, self.critic_eval, 1.0)

        if OUTPUT_GRAPH:
            tf.summary.create_file_writer("logs/")

    def _build_a(self, name):
        class ScalingLayer(tf.keras.layers.Layer):
            def __init__(self, scale):
                super(ScalingLayer, self).__init__()
                self.scale = scale

            def call(self, inputs):
                return inputs * self.scale

        inputs = tf.keras.layers.Input(shape=(self.s_dim,))
        net = tf.keras.layers.Dense(400, activation='relu')(inputs)
        net = tf.keras.layers.Dense(300, activation='relu')(net)
        net = tf.keras.layers.Dense(10, activation='relu')(net)
        outputs = tf.keras.layers.Dense(self.a_dim, activation='tanh')(net)
        outputs = ScalingLayer(self.a_bound[1])(outputs)
        return tf.keras.Model(inputs, outputs, name=name)

    def _build_c(self, name):
        s_inputs = tf.keras.layers.Input(shape=(self.s_dim,))
        a_inputs = tf.keras.layers.Input(shape=(self.a_dim,))
        concat = tf.keras.layers.Concatenate()([s_inputs, a_inputs])
        net = tf.keras.layers.Dense(400, activation='relu')(concat)
        net = tf.keras.layers.Dense(300, activation='relu')(net)
        net = tf.keras.layers.Dense(10, activation='relu')(net)
        outputs = tf.keras.layers.Dense(1)(net)
        return tf.keras.Model([s_inputs, a_inputs], outputs, name=name)

    @tf.function
    def update_target_network(self, target, source, tau):
        for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.actor_eval(s)[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        bs = tf.convert_to_tensor(bs, dtype=tf.float32)
        ba = tf.convert_to_tensor(ba, dtype=tf.float32)
        br = tf.convert_to_tensor(br, dtype=tf.float32)
        bs_ = tf.convert_to_tensor(bs_, dtype=tf.float32)

        mse_loss = tf.keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:
            a = self.actor_eval(bs, training=True)
            q = self.critic_eval([bs, a], training=True)
            a_loss = -tf.reduce_mean(q)
        a_grads = tape.gradient(a_loss, self.actor_eval.trainable_variables)
        self.actor_eval_optimizer.apply_gradients(zip(a_grads, self.actor_eval.trainable_variables))

        with tf.GradientTape() as tape:
            q = self.critic_eval([bs, ba], training=True)
            q_ = self.critic_target([bs_, self.actor_target(bs_)], training=True)
            q_target = br + self.GAMMA * q_
            td_error = mse_loss(q_target, q)
        c_grads = tape.gradient(td_error, self.critic_eval.trainable_variables)
        self.critic_eval_optimizer.apply_gradients(zip(c_grads, self.critic_eval.trainable_variables))

        self.update_target_network(self.actor_target, self.actor_eval, self.TAU)
        self.update_target_network(self.critic_target, self.critic_eval, self.TAU)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1



###############################  training  ####################################
np.random.seed(1578)
tf.compat.v1.set_random_seed(1578)

env = STAR_RIS_env(antenna_num=4, user_num=4, element_num=30, power_limit=20, target_num=4, eve_num=1)
MAX_EP_STEPS = 1000
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = [-1, 1]  # [-1,1]

ddpg = DDPG(a_dim, s_dim, a_bound)

# var = 1  # control exploration
var = 0.01  # control exploration
t1 = time.time()
ep_reward_list = []
avg_reward_list = []
P_list_list = []
rad_list_list = []
sec_list_list = []
eta_list_list = []
a_list = []
# s_normal = StateNormalization()

for i in range(MAX_EPISODES):
    env.reset_env()
    # W = np.ones((env.antenna_num, env.element_num + env.user_num), dtype=float)
    a = np.random.rand(env.action_dim)
    s, _, _, _, _, _ = env.step(a)
    ep_reward = 0
    ep_eta = 0
    rad_list = []
    sec_list = []
    P_list = []
    eta_list = []
    j = 0
    while j < MAX_EP_STEPS:
        # Add exploration noise
        # a = ddpg.choose_action(s_normal.state_normal(s))
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), *a_bound)  # 高斯噪声add randomness to action selection for exploration
        # s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(a)
        s_, r, P, sum_sec, sum_rad, eta = env.step(a)
        # if step_redo:
        #     continue
        # if reset_dist:
        #     a[2] = -1
        # if offloading_ratio_change:
        #     a[3] = -1
        # ddpg.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))  # 训练奖励缩小10倍
        ddpg.store_transition(s, a, r, s_)  # 训练奖励缩小10倍
        if ddpg.pointer > MEMORY_CAPACITY:
            # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            ddpg.learn()
        s = s_
        ep_reward += r
        ep_eta += eta
        rad_list.append(sum_rad)
        sec_list.append(sum_sec)
        P_list.append(P)
        a_list.append(a)
        # eta_list.append(eta)
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % var, 'Average reward: %.2f' % np.mean(ep_reward_list), 'episode eta: %.2f' % np.mean(eta_list_list))
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            avg_reward_list = np.append(avg_reward_list, np.mean(ep_reward_list))
            rad_list_list.append(rad_list)
            sec_list_list.append(sec_list)
            P_list_list.append(P_list)
            eta_list_list.append(ep_eta)
            # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
            # file_name = 'output.txt'
            # with open(file_name, 'a') as file_obj:
            #     file_obj.write("\n======== This episode is done ========" + "Episode:" + str(i) + "Reward" + str(ep_reward)+ "avg" + str(np.mean(ep_reward_list)))  # 本episode结束
            break
        j = j + 1

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)
# 将ep_reward_list、avg_reward_list保存到文件中
# np.save('avg_reward_list.npy', avg_reward_list)
# np.save('ep_reward_list.npy', ep_reward_list)
# 构建文件夹路径
# folder_path = f"./DDPG/数据/proposed/LA={LR_A},LC={LR_C},GAMMA={GAMMA}/M={env.antenna_num},K={env.user_num},N={env.element_num},P={env.power},T={env.target_num},F={env.eve_num}"
# folder_path = f"./DDPG/数据/random/alpha=0.45,beta=0.45,gamma=0.1/LA={LR_A},LC={LR_C},GAMMA={GAMMA}/M={env.antenna_num},K={env.user_num},N={env.element_num},P={env.power_limit},T={env.target_num},F={env.eve_num}"
# folder_path = f"./DDPG/数据/proposed/alpha=0.33,beta=0.33,gamma=0.33/LA={LR_A},LC={LR_C},GAMMA={GAMMA}/M={env.antenna_num},K={env.user_num},N={env.element_num},P={env.power},T={env.target_num},F={env.eve_num}"
# folder_path = f"./DDPG/数据/MRT/alpha=0.45,beta=0.45,gamma=0.1/LA={LR_A},LC={LR_C},GAMMA={GAMMA}/M={env.antenna_num},K={env.user_num},N={env.element_num},P={env.power_limit},T={env.target_num},F={env.eve_num}"
# folder_path = f"./DDPG/数据/proposed_eta_objective/LA={LR_A},LC={LR_C},GAMMA={GAMMA}/M={env.antenna_num},K={env.user_num},N={env.element_num},P={env.power},T={env.target_num},F={env.eve_num}"
folder_path = f"数据/utility_targ_not_eve/LA={LR_A},LC={LR_C},GAMMA={GAMMA}/M={env.antenna_num},K={env.user_num},N={env.element_num},P={env.power},T={env.target_num},F={env.eve_num}"


# 确保文件夹存在，如果不存在则创建
os.makedirs(folder_path, exist_ok=True)
file_path1 = os.path.join(folder_path, "ep_reward_list.npy")
file_path2 = os.path.join(folder_path, "rad_list_list.npy")
file_path3 = os.path.join(folder_path, "sec_list_list.npy")
file_path4 = os.path.join(folder_path, "P_list_list.npy")
file_path5 = os.path.join(folder_path, "eta_list_list.npy")
file_path6 = os.path.join(folder_path, "a_list.npy")
np.save(file_path1, ep_reward_list)
np.save(file_path2, rad_list_list)
np.save(file_path3, sec_list_list)
np.save(file_path4, P_list_list)
np.save(file_path5, eta_list_list)
# np.save(file_path6, a_list)
# 将ep_reward_list和avg_reward_list的曲线画到一张图上，并标注
# plt.plot(ep_reward_list, label='ep_reward_list')
plt.plot(avg_reward_list, label='avg_reward_list')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
print('Running time: ', time.time() - t1)
plt.savefig(f"{folder_path}/ep_reward_list.png")

# plt.plot(ep_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.show()
