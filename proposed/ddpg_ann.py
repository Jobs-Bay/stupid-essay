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
MAX_EPISODES = 3000
# MAX_EPISODES = 50000
x = 25
LR_A = 1e-5  # learning rate for actor
LR_C = 1e-5  # learning rate for critic
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
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励
        self.pointer = 0
        self.sess = tf.Session()
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.GAMMA = GAMMA
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))

        # transition = np.hstack((s, [a], [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')
            # return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################
seed = 2
np.random.seed(3407)
tf.set_random_seed(seed)

env = STAR_RIS_env(antenna_num=4, user_num=4, element_num=10, power_limit=0, target_num=10, eve_num=1)
MAX_EP_STEPS = 1000
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = [-1, 1]  # [-1,1]

ddpg = DDPG(a_dim, s_dim, a_bound)

# var = 0.1  # control exploration
var = 0.01
# control exploration
t1 = time.time()
ep_reward_list = []
avg_reward_list = []
P_list_list = []
rad_list_list = []
sec_list_list = []
eta_list_list = []
a_list = []
beta = 1
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
        a = ddpg.choose_action(s)
        a = np.clip(a + beta * np.sqrt(var) * np.random.randn(a_dim), *a_bound)  # 高斯噪声add randomness to action selection for exploration
        s_, r, P, sum_sec, sum_rad, eta = env.step(a)
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
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'variance: %.3f' % np.var(ep_reward_list), 'Average reward: %.2f' % np.mean(ep_reward_list), 'episode eta: %.2f' % np.mean(eta_list_list))
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            avg_reward_list = np.append(avg_reward_list, np.mean(ep_reward_list))
            rad_list_list.append(rad_list)
            sec_list_list.append(sec_list)
            P_list_list.append(P_list)
            eta_list_list.append(ep_eta)
            break
        j = j + 1
    if i >= x:
        batch = env.normalize_state(ep_reward_list[i - x:i - 1])
        # 求ep_reward_list的方差
        beta = np.var(batch)
        # 取beta的自然对数
        beta = np.exp(- beta)
        print(beta)
#save model



# 将ep_reward_list、avg_reward_list保存到文件中
# np.save('avg_reward_list.npy', avg_reward_list)
# np.save('ep_reward_list.npy', ep_reward_list)
# 构建文件夹路径
folder_path = f"数据/decay_random/x={x}/LA={LR_A},LC={LR_C},GAMMA={GAMMA}/M={env.antenna_num},K={env.user_num},N={env.element_num},P={env.power},T={env.target_num},F={env.eve_num}"


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
print(seed)
os.makedirs(folder_path, exist_ok=True)
saver = tf.train.Saver()
saver.save(ddpg.sess, folder_path)
print('Model saved')
