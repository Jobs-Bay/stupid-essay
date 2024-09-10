import tensorflow as tf
import numpy as np
from proposed.test_env import STAR_RIS_env as proposed_env

MAX_EPISODES = 1000
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
GAMMA = 0.5  # optimal reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
MEMORY_CAPACITY = 10000 #可以变大一点  滑动窗口平均 列表数据做一下滑动窗口平均
BATCH_SIZE = 64
OUTPUT_GRAPH = False

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
np.random.seed(1) #禁止修改
proposed_env = proposed_env(antenna_num=4, user_num=4, element_num=30, power_limit=4, target_num=4, eve_num=1)
ddpg = DDPG(a_dim=proposed_env.action_dim, s_dim=proposed_env.state_dim, a_bound=[-1, 1])
saver = tf.train.Saver()
with ddpg.sess as sess:
    # 恢复模型参数
    saver.restore(sess, "proposed/数据/decay_random/x=25/LA=1e-05,LC=1e-05,GAMMA=0.5/M=4,K=4,N=30,P=0,T=4,F=1")  # 请根据你的模型路径修改
    print("Model restored.")
    num_test_episodes = 10
    max_steps = 200
    episode_eta = 0
    for episode in range(num_test_episodes):
        proposed_env.reset_env()
        a = np.random.rand(proposed_env.action_dim)
        s, _, _, _, _, _ = proposed_env.step(a)
        episode_reward = 0
        for step in range(max_steps):
            a = ddpg.choose_action(s)  # 测试时不添加噪声
            s_, r, P, sum_sec, sum_rad, eta = proposed_env.step(a)
            s = s_
            episode_reward += r
            episode_eta += eta
        print(f"Test Episode {episode + 1}, Reward: {episode_eta / max_steps / num_test_episodes}")
"""
    注意：
        测试时不要修改随机种子
        模型练完之后及时验证固定
        ####除了环境参数不要修改任何地方，不然会影响随机结果####
        最后修改时间：2024/07/29
"""