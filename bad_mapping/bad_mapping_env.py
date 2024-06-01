import math
import random
import matplotlib.pyplot as plt
import numpy as np


class STAR_RIS_env(object):
    """路径衰落系数"""
    alpha_BR = 2.3  # BS - RIS
    alpha_Ru = 2.3  # RIS - user
    alpha_Bu = 2.3  # BS - user
    alpha_Bt = 2.7  # BS - target
    alpha_Rt = 2.7  # RIS - target
    alpha_Rf = 2.3  # RIS - eavesdropper
    alpha_Bf = 2.3  # BS - eavesdropper

    """Rician factor"""
    K_dB = 3
    K = 10 ** (K_dB / 10)

    C0_dB = -30
    C0 = 10 ** (C0_dB / 10)

    """noise power"""
    sigmak = 10 ** (-12)
    sigmat = 10 ** (-13)
    sigmaj = 10 ** (-12)

    def __init__(self, antenna_num, user_num, element_num, power_limit, target_num, eve_num):
        self.antenna_num = antenna_num
        self.user_num = user_num
        self.indoor_user_num = 1
        self.outdoor_user_num = self.user_num - self.indoor_user_num
        self.element_num = element_num
        self.power = power_limit
        self.power_limit = 10 ** (power_limit / 10) /1000
        self.target_num = target_num
        self.eve_num = eve_num
        self.indoor_eve_num = 0
        self.outdoor_eve_num = 1
        self.action_dim = 4 * self.element_num + 2 * self.antenna_num * (self.antenna_num + self.user_num)
        # 由于star-ris的系数矩阵由两部分组成，一个是发射一个是反射的，所以有2*N个参数，因为神经网络只接受实数，所以将系数分为实部和虚部，所以有4*N个参数
        # 因为预编码矩阵是一个M*（M+K）的矩阵，同理，有2*M*（M+K）个参数
        # 外加一个功率分配因子
        # 所以动作维度为4*N+2*M*(M+K) + 1
        # 前N位是发射的实部，接着N位是发射的虚部，再接着N位是反射的实部，再接着N位是反射的虚部，接着M*(M+K)位是预编码矩阵的实部，再接着M*(M+K)位是预编码矩阵的虚部
        self.state_dim = self.antenna_num + self.user_num + 1 + self.user_num * (self.eve_num ) + self.target_num - 1
        # 状态维度由预编码矩阵每列的L2范数的平方、能量效率、所有的安全通信速率、所有目标的雷达估计速率组成
        self.Phi_t = np.eye(self.element_num, dtype=complex)
        self.Phi_r = np.eye(self.element_num, dtype=complex)
        self.W = np.random.randn(self.antenna_num, self.antenna_num + self.user_num) + 1j * np.random.randn(self.antenna_num, self.antenna_num + self.user_num)
        """位置信息"""
        self.BS_loc = [[-20, 20]]
        self.STAR_RIS_loc = [[0, 0]]
        self.indoor_user_loc, self.indoor_eve_loc, self.outdoor_user_loc, self.outdoor_eve_loc, self.target_loc = self.generate_loc()  # 生成各个坐标
        self.BS2STAR_RIS, self.BS2target, self.BS2user, self.BS2eve, self.STAR_RIS2outdoor_user, self.STAR_RIS2outdoor_eve, \
        self.STAR_RIS2target, self.STAR_RIS2indoor_user, self.STAR_RIS2indoor_eve = self.get_Azimuth()  # 获得方位角
        self.dBS2STAR_RIS, self.dBS2target, self.dBS2user, self.dBS2eve, self.dSTAR_RIS2outdoor_user, self.dSTAR_RIS2outdoor_eve, \
        self.dSTAR_RIS2target, self.dSTAR_RIS2indoor_user, self.dSTAR_RIS2indoor_eve = self.get_distance()  # 获得距离
        """路径衰落"""
        self.PL_BS2STAR_RIS, self.PL_BS2target, self.PL_BS2user, self.PL_BS2eve, self.PL_STAR_RIS2outdoor_user, self.PL_STAR_RIS2outdoor_eve, \
        self.PL_STAR_RIS2target, self.PL_STAR_RIS2indoor_user, self.PL_STAR_RIS2indoor_eve = self.get_pathloss()   # 获得路径损耗
        """信道"""
        self.H_dt, self.G, self.H_du, self.H_de, self.H_rt, self.H_ru, self.H_rf = self.generate_channel()  # 生成信道

    def reset_env(self):
        """
        函数用途：重置环境（位置信息、路径损耗、信道）
        参数列表：无
        返回值：无
        """
        """位置信息"""
        self.BS_loc = [[-20, 20]]
        self.STAR_RIS_loc = [[0, 0]]
        self.indoor_user_loc, self.indoor_eve_loc, self.outdoor_user_loc, self.outdoor_eve_loc, self.target_loc = self.generate_loc()  # 生成各个坐标
        self.BS2STAR_RIS, self.BS2target, self.BS2outdoor_user, self.BS2outdoor_eve, self.STAR_RIS2outdoor_user, self.STAR_RIS2outdoor_eve, \
        self.STAR_RIS2target, self.STAR_RIS2indoor_user, self.STAR_RIS2indoor_eve = self.get_Azimuth()  # 获得方位角
        self.dBS2STAR_RIS, self.dBS2target, self.dBS2outdoor_user, self.dBS2outdoor_eve, self.dSTAR_RIS2outdoor_user, self.dSTAR_RIS2outdoor_eve, \
        self.dSTAR_RIS2target, self.dSTAR_RIS2indoor_user, self.dSTAR_RIS2indoor_eve = self.get_distance()  # 获得距离
        """路径衰落"""
        self.PL_BS2STAR_RIS, self.PL_BS2target, self.PL_BS2outdoor_user, self.PL_BS2outdoor_eve, self.PL_STAR_RIS2outdoor_user, self.PL_STAR_RIS2outdoor_eve, \
        self.PL_STAR_RIS2target, self.PL_STAR_RIS2indoor_user, self.PL_STAR_RIS2indoor_eve = self.get_pathloss()   # 获得路径损耗
        """信道"""
        self.H_dt, self.G, self.H_du, self.H_df, self.H_rt, self.H_ru, self.H_rf = self.generate_channel()  # 生成信道

    def calculte_reward(self):
        #todo:写奖励函数
        """
        函数用途：计算奖励
        参数列表：无
        返回值：奖励
        """
        pass

    def step(self, action):  # 前N位是发射的实部，接着N位是发射的虚部，再接着N位是反射的实部，再接着N位是反射的虚部，接着M*(M+K)位是预编码矩阵的实部，再接着M*(M+K)位是预编码矩阵的虚部
        """
        函数用途：环境的一次迭代
        参数列表：动作
        返回值：下一个状态、奖励、是否结束、其他信息
        """

        SNR_t = np.array([0] * self.target_num, dtype=float)
        SNR_e = np.array([0] * self.user_num, dtype=float)  # （窃听者+目标，用户）
        SNR_u = np.array([0] * self.user_num, dtype=float)
        R_rad = np.array(self.target_num * [0], dtype=float)  # （目标，用户）
        R_k = np.array([0] * self.user_num, dtype=float)
        R_f = np.array([0] * self.user_num, dtype=float) # （窃听者+目标，用户）
        R_sec = np.array([0] * self.user_num, dtype=float)
        Phi_t_real = action[:self.element_num]
        Phi_t_imag = action[self.element_num:2 * self.element_num]
        Phi_r_real = action[2 * self.element_num:3 * self.element_num]
        Phi_r_imag = action[3 * self.element_num:4 * self.element_num]
        W_real = action[4 * self.element_num:4 * self.element_num + self.antenna_num * (self.antenna_num + self.user_num)]
        W_imag = action[4 * self.element_num + self.antenna_num * (self.antenna_num + self.user_num):4 * self.element_num + self.antenna_num * (self.antenna_num + self.user_num) + self.antenna_num * (self.antenna_num + self.user_num)]
        W_before = W_real.reshape(self.antenna_num, self.antenna_num + self.user_num) + 1j * W_imag.reshape(self.antenna_num, self.antenna_num + self.user_num)
        Phi_t, Phi_r = self.normalize_Phi(Phi_t_real, Phi_t_imag, Phi_r_real, Phi_r_imag)
        self.Phi_t = Phi_t * np.eye(self.element_num, dtype=complex)
        self.Phi_r = Phi_r * np.eye(self.element_num, dtype=complex)
        W = self.normalize_W(W_before, self.power_limit)
        # 算每个目标上的雷达估计速率
        for i in range(self.target_num):
            h_dt = ((self.H_dt[:, i]).reshape(self.antenna_num, 1) + self.G @ self.Phi_r @ self.H_rt[:, i].reshape(self.element_num, 1)) @(self.H_dt[:, i].reshape(self.antenna_num, 1) + self.G @ self.Phi_r @ self.H_rt[:, i].reshape(self.element_num, 1)).T
            #求self.W.conjugate.T @ h_dt.conjugate.T @ h_dt @ W的trace
            h = W.conjugate().T @ (h_dt.conjugate().T @ h_dt) @ W
            # print(np.trace(h))
            SNR_t[i] = np.trace(h) / self.sigmat
            R_rad[i] = np.log2(1 + SNR_t[i])
        # 算每个用户的速率
        for i in range(self.outdoor_user_num):
            h_du = self.H_du[:, i].reshape(self.antenna_num, 1).T + self.H_ru[:, i].reshape(self.element_num, 1).T  @ self.Phi_r @ self.G.T
            # 求h_du @ W[:, i]的模的平方，除以(h_du @ W[:, j]，j≠i  + sigmak)
            SNR_u[i] = np.abs(h_du @ W[:, i]) ** 2 / (np.sum(np.abs(h_du @ W) ** 2) - np.abs(h_du @ W[:, i]) ** 2 + self.sigmak)
            R_k[i] = np.log2(1 + SNR_u[i])
        for i in range(self.indoor_user_num):
            h_du = self.H_du[:, i + self.outdoor_user_num].reshape(self.antenna_num, 1).T + self.H_ru[:, i + self.outdoor_user_num].reshape(self.element_num, 1).T @ self.Phi_t @ self.G.T
            SNR_u[i + self.outdoor_user_num] = np.abs(h_du @ W[:, i + self.outdoor_user_num]) ** 2 / (np.sum(np.abs(h_du @ W) ** 2) - np.abs(h_du @ W[:, i + self.outdoor_user_num]) ** 2 + self.sigmak)
            R_k[i + self.outdoor_user_num] = np.log2(1 + SNR_u[i + self.outdoor_user_num])
        # 算每个窃听者的速率
        h_df = self.H_de.reshape(self.antenna_num, 1).T + self.H_rf[:, 0].reshape(self.element_num, 1).T @ self.Phi_r @ self.G.T
        for j in range(self.user_num):
            SNR_e[j] = np.abs(h_df @ W[:, j]) ** 2 / (np.sum(np.abs(h_df @ W) ** 2) - np.abs(h_df @ W[:, j]) ** 2 + self.sigmaj)
            R_f[j] = np.log2(1 + SNR_e[j])
        # 算安全速率和雷达估计速率
        for i in range(self.user_num):
            if R_k[i] > R_f[i]:
                R_sec[i] = R_k[i] - R_f[i]
            else:
                R_sec[i] = 0
        eta = ((np.sum(R_sec)) ** 0.33 * np.sum(SNR_t) ** 0.33) / (np.linalg.norm(W, 'fro') ** 0.66)
        # 求W每一列的L2范数的平方，并将其储存起来
        W_norm = np.array([0] * (self.antenna_num + self.user_num), dtype=float)
        for i in range(self.antenna_num + self.user_num):
            W_norm[i] = np.linalg.norm(W[:, i]) ** 2
        state = W_norm
        state = np.append(state, R_sec)
        state = np.append(state, SNR_t)
        # state = np.append(state, SNR_t)

        reward = ((np.sum(R_sec)) ** 0.33 * np.sum(SNR_t) ** 0.33) / (np.linalg.norm(W, 'fro') ** 0.66)
        #如果R_sec中有小于0.2的只值，则reward为0
        for i in range(self.user_num):
            if R_sec[i] < 0.1:
                reward = 0.5 * reward
                break
        for i in range(self.target_num):
            if SNR_t[i] < 0.5:
                reward = 0.5 * reward
                break

        return state, reward, np.linalg.norm(W, 'fro') ** 2, np.sum(R_sec), np.sum(SNR_t), eta

    def normalize_state(self, state):
        """
        函数用途：对状态进行归一化
        参数列表：状态
        返回值：归一化后的状态
        """
        state = (state - np.min(state)) / (np.max(state) - np.min(state))
        return state

    def normalize_W(self, W_before, P_max):
        """
        函数用途：因为神经网络激活函数的限制，需要对W进行归一化
        :param W_before: 大小为天线数*（天线数+用户数）的矩阵
        :return:W_after: 大小为天线数*（天线数+用户数）的矩阵
        """
        P = np.array([0] * (self.antenna_num + self.user_num), dtype=float)
        P_tilde = np.array([0] * (self.antenna_num + self.user_num), dtype=float)
        # 创建一个大小为天线数*（天线数+用户数）的元素都为1+1j的复矩阵，赋值给W_max
        W_max = np.ones((self.antenna_num, self.antenna_num + self.user_num)) + 1j * np.ones((self.antenna_num, self.antenna_num + self.user_num))
        # P_max除以W_max的F范数，并赋值给λ
        λ = P_max / (np.linalg.norm(W_max, 'fro')) ** 2
        W_after = W_before * math.sqrt(λ)
        # λ = P_max / (np.linalg.norm(W_max, 'fro'))
        # W_after = W_before * λ
        return W_after

    def normalize_Phi(self, Phi_t_real, Phi_t_imag, Phi_r_real, Phi_r_imag):
        """
        函数用途：因为神经网络激活函数的限制，需要对Φ进行归一化
        :param Phi_t_real, Phi_r_real, Phi_t_imag, Phi_r_imag: 长度为self.element_num的数组
        :return:Phi_t, Phi_r: 长度为self.element_num的数组
        """
        theta_t = np.array([0] * self.element_num, dtype=float)
        theta_r = np.array([0] * self.element_num, dtype=float)
        beta_t = np.array([0] * self.element_num, dtype=float)
        beta_r = np.array([0] * self.element_num, dtype=float)
        Phi_t = np.array([0] * self.element_num, dtype=complex)
        Phi_r = np.array([0] * self.element_num, dtype=complex)
        for i in range(self.element_num):
            theta_t[i] = np.arctan(Phi_t_imag[i] / Phi_t_real[i])
            theta_r[i] = np.arctan(Phi_r_imag[i] / Phi_r_real[i])
            beta_t[i] = np.sqrt(Phi_t_real[i] ** 2 + Phi_t_imag[i] ** 2) / (np.sqrt(Phi_t_real[i] ** 2 + Phi_t_imag[i] ** 2) + np.sqrt(Phi_r_real[i] ** 2 + Phi_r_imag[i] ** 2))
            beta_r[i] = np.sqrt(Phi_r_real[i] ** 2 + Phi_r_imag[i] ** 2) / (np.sqrt(Phi_t_real[i] ** 2 + Phi_t_imag[i] ** 2) + np.sqrt(Phi_r_real[i] ** 2 + Phi_r_imag[i] ** 2))
            Phi_t[i] = math.sqrt(beta_t[i]) * math.cos(theta_t[i]) + 1j * math.sin(theta_t[i]) * math.sqrt(beta_t[i])
            Phi_r[i] = math.sqrt(beta_r[i]) * math.cos(theta_r[i]) + 1j * math.sin(theta_r[i]) * math.sqrt(beta_r[i])
        return Phi_t, Phi_r

    def get_pathloss(self):
        """
        函数用途：计算各个链路的路径损耗
        参数列表：无
        返回值：各个链路的路径损耗
        """
        PL_BS2target = np.array([0] * self.target_num, dtype=float)
        PL_BS2user = np.array([0] * self.user_num, dtype=float)
        PL_BS2eve = np.array([0] * self.eve_num, dtype=float)
        PL_STAR_RIS2outdoor_user = np.array([0] * self.outdoor_user_num, dtype=float)
        PL_STAR_RIS2outdoor_eve = np.array([0] * self.outdoor_eve_num, dtype=float)
        PL_STAR_RIS2target = np.array([0] * self.target_num, dtype=float)
        PL_STAR_RIS2indoor_user = np.array([0] * self.indoor_user_num, dtype=float)
        PL_STAR_RIS2indoor_eve = np.array([0] * self.indoor_eve_num, dtype=float)

        # 计算BS到STAR-RIS、BS到目标、BS到室外用户、BS到室外窃听者、STAR-RIS到室外用户、STAR-RIS到室外窃听者、STAR-RIS到目标、STAR-RIS到室内用户、STAR-RIS到室内窃听者的路径损耗
        PL_BS2STAR_RIS = math.sqrt(self.C0 * (self.dBS2STAR_RIS ** (-self.alpha_BR)))
        for i in range(self.target_num):
            PL_BS2target[i] = math.sqrt(self.C0 * (self.dBS2target[i] ** (-self.alpha_Bt)))
        for i in range(self.user_num):
            PL_BS2user[i] = math.sqrt(self.C0 * (self.dBS2user[i] ** (-self.alpha_Bu)))
        for i in range(self.eve_num):
            PL_BS2eve[i] = math.sqrt(self.C0 * (self.dBS2eve[i] ** (-self.alpha_Bf)))
        for i in range(self.outdoor_user_num):
            PL_STAR_RIS2outdoor_user[i] = math.sqrt(self.C0 * (self.dSTAR_RIS2outdoor_user[i] ** (-self.alpha_Ru)))
        for i in range(self.outdoor_eve_num):
            PL_STAR_RIS2outdoor_eve[i] = math.sqrt(self.C0 * (self.dSTAR_RIS2outdoor_eve[i] ** (-self.alpha_Rf)))
        for i in range(self.target_num):
            PL_STAR_RIS2target[i] = math.sqrt(self.C0 * (self.dSTAR_RIS2target[i] ** (-self.alpha_Rt)))
        for i in range(self.indoor_user_num):
            PL_STAR_RIS2indoor_user[i] = math.sqrt(self.C0 * (self.dSTAR_RIS2indoor_user[i] ** (-self.alpha_Ru)))
        for i in range(self.indoor_eve_num):
            PL_STAR_RIS2indoor_eve[i] = math.sqrt(self.C0 * (self.dSTAR_RIS2indoor_eve[i] ** (-self.alpha_Rf)))

        return PL_BS2STAR_RIS, PL_BS2target, PL_BS2user, PL_BS2eve, PL_STAR_RIS2outdoor_user, PL_STAR_RIS2outdoor_eve, PL_STAR_RIS2target, PL_STAR_RIS2indoor_user, PL_STAR_RIS2indoor_eve

    def generate_loc(self):
        """
        函数用途：生成目标、用户、窃听者的位置
        参数列表：无
        返回值：室外室外窃听者和用户的位置，目标的位置
        """
        # 生成用户、目标和窃听者的位置其中用户分为室内和室外用户，窃听者分为室内和室外窃听者，目标只有室外目标
        # 室内用户的位置横坐标取值范围为[5,11],室内窃听者[5,11]
        indoor_user_loc_x = np.random.randint(0, 6, self.indoor_user_num)
        indoor_eve_loc_x = np.random.randint(5, 11, self.indoor_eve_num)
        # 室内用户坐标纵坐标取值范围为[0,5],室内窃听者[0,5]
        indoor_user_loc_y = np.random.randint(0, 6, self.indoor_user_num)
        indoor_eve_loc_y = np.random.randint(0, 6, self.indoor_eve_num)
        indoor_user_loc = np.vstack((indoor_user_loc_x, indoor_user_loc_y)).T
        indoor_eve_loc = np.vstack((indoor_eve_loc_x, indoor_eve_loc_y)).T
        # 室外用户的位置横坐标取值范围为[-15,-10],室外窃听者[-15,-10]
        outdoor_user_loc_x = np.random.randint(-15, -4, self.outdoor_user_num)
        outdoor_eve_loc_x = np.random.randint(-5, 1, self.outdoor_eve_num)
        # 室外用户坐标纵坐标取值范围为[0, 10],室外窃听者[0,10]
        outdoor_user_loc_y = np.random.randint(5, 19, self.outdoor_user_num)
        outdoor_eve_loc_y = np.random.randint(-10, -4, self.outdoor_eve_num)
        outdoor_user_loc = np.vstack((outdoor_user_loc_x, outdoor_user_loc_y)).T
        outdoor_eve_loc = np.vstack((outdoor_eve_loc_x, outdoor_eve_loc_y)).T
        # 室外目标的位置横坐标取值范围为[-30,-5]
        target_loc_x = np.random.randint(-15, -4, self.target_num)
        # 室外目标的位置纵坐标取值范围为[-10, 0]
        target_loc_y = np.random.randint(-5, 1, self.target_num)
        target_loc = np.vstack((target_loc_x, target_loc_y)).T

        return indoor_user_loc, indoor_eve_loc, outdoor_user_loc, outdoor_eve_loc, target_loc

    def generate_channel(self):
        """
        函数用途：生成各类信道
        参数列表：无
        返回值：基站到目标的信道矩阵、基站到STAR-RIS的信道矩阵、基站到室外用户的信道矩阵、基站到室外窃听者的信道矩阵、STAR-RIS到室外用户的信道矩阵、
        STAR-RIS到室外窃听者的信道矩阵、STAR-RIS到目标的信道矩阵、STAR-RIS到室内用户的信道矩阵、STAR-RIS到室内窃听者的信道矩阵
        """

        """ 基站侧  """
        # 生成一个大小为天线数*目标数的复矩阵，其中每一列的元素为cos(-n*pi*sin(BS2target[i]))-1j*sin(-n*pi*sin(BS2target[i]))，赋值给H_dt_LOS
        H_dt_LOS = np.zeros((self.antenna_num, self.target_num), dtype=complex)
        for i in range(self.target_num):
            for n in range(self.antenna_num):
                H_dt_LOS[n, i] = self.PL_BS2target[i] * np.cos(-n * math.pi * np.sin(self.BS2target[i])) - self.PL_BS2target[i] * 1j * np.sin(-n * math.pi * np.sin(self.BS2target[i]))
        H_dt = H_dt_LOS

        # 生成一个大小为天线数*元素数的复矩阵，其中每一列的元素为cos(-n*pi*sin(BS2STAR_RIS))-1j*sin(-n*pi*sin(BS2STAR_RIS))，赋值给G
        G = np.zeros((self.antenna_num, self.element_num), dtype=complex)
        for i in range(self.element_num):
            for n in range(self.antenna_num):
                G[n, i] = self.PL_BS2STAR_RIS * np.cos(-n * math.pi * np.sin(self.BS2STAR_RIS)) - self.PL_BS2STAR_RIS * 1j * np.sin(-n * math.pi * np.sin(self.BS2STAR_RIS))

        # 生成一个大小为天线数*用户数的服从复高斯分布的矩阵，赋值给H_du_outdoor
        H_du = np.zeros((self.antenna_num, self.user_num), dtype=complex)
        for i in range(self.user_num):
            for n in range(self.antenna_num):
                H_du[n, i] = self.PL_BS2user[i] * np.cos(-n * math.pi * np.sin(self.BS2user[i])) - self.PL_BS2user[i] * 1j * np.sin(-n * math.pi * np.sin(self.BS2user[i]))

        H_de = np.zeros((self.antenna_num, self.eve_num), dtype=complex)
        for i in range(self.eve_num):
            for n in range(self.antenna_num):
                H_de[n, i] = self.PL_BS2eve[i] * np.cos(-n * math.pi * np.sin(self.BS2eve[i])) - self.PL_BS2eve[
                    i] * 1j * np.sin(-n * math.pi * np.sin(self.BS2eve[i]))

        """STAR-RIS侧"""
        # 生成一个大小为元素数*目标数的复矩阵，其中每一列的元素为cos(-n*pi*sin(STAR_RIS2target[i]))-1j*sin(-n*pi*sin(STAR_RIS2target[i]))，赋值给H_rt_LOS
        H_rt_LOS = np.zeros((self.element_num, self.target_num), dtype=complex)
        for i in range(self.target_num):
            for n in range(self.element_num):
                H_rt_LOS[n, i] = self.PL_STAR_RIS2target[i] * np.cos(-n * math.pi * np.sin(self.STAR_RIS2target[i])) - self.PL_STAR_RIS2target[i] * 1j * np.sin(-n * math.pi * np.sin(self.STAR_RIS2target[i]))
        H_rt = H_rt_LOS

        # 生成一个大小为元素数*室外用户数的复矩阵，其中每一列的元素为cos(-n*pi*sin(STAR_RIS2outdoor_user[i]))-1j*sin(-n*pi*sin(STAR_RIS2outdoor_user[i]))，赋值给H_ru_LOS
        H_ru_LOS = np.zeros((self.element_num, self.outdoor_user_num), dtype=complex)
        for i in range(self.outdoor_user_num):
            for n in range(self.element_num):
                H_ru_LOS[n, i] = np.cos(-n * math.pi * np.sin(self.STAR_RIS2outdoor_user[i])) - 1j * np.sin(-n * math.pi * np.sin(self.STAR_RIS2outdoor_user[i]))
        # 生成一个大小为元素数*室外用户数的服从高斯分布的复矩阵，赋值给H_ru_NLOS
        H_ru_NLOS = np.random.randn(self.element_num, self.outdoor_user_num) + 1j * np.random.randn(self.element_num, self.outdoor_user_num)
        H_ru_outdoor = math.sqrt(self.K / (self.K + 1)) * H_ru_LOS + math.sqrt(1 / (self.K + 1)) * H_ru_NLOS
        for i in range(self.outdoor_user_num):
            for n in range(self.element_num):
                H_ru_outdoor[n, i] = self.PL_STAR_RIS2outdoor_user[i] * H_ru_outdoor[n, i]

        # 生成一个大小为元素数*室内用户数的复矩阵，其中每一列的元素为cos(-n*pi*sin(STAR_RIS2outdoor_user[i]))-1j*sin(-n*pi*sin(STAR_RIS2outdoor_user[i]))，赋值给H_ru_LOS
        H_ru_LOS = np.zeros((self.element_num, self.indoor_user_num), dtype=complex)
        for i in range(self.indoor_user_num):
            for n in range(self.element_num):
                H_ru_LOS[n, i] = np.cos(-n * math.pi * np.sin(self.STAR_RIS2indoor_user[i])) - 1j * np.sin(-n * math.pi * np.sin(self.STAR_RIS2indoor_user[i]))
        # 生成一个大小为元素数*室外用户数的服从高斯分布的复矩阵，赋值给H_ru_NLOS
        H_ru_NLOS = np.random.randn(self.element_num, self.indoor_user_num) + 1j * np.random.randn(self.element_num, self.indoor_user_num)
        H_ru_indoor = math.sqrt(self.K / (self.K + 1)) * H_ru_LOS + math.sqrt(1 / (self.K + 1)) * H_ru_NLOS
        for i in range(self.indoor_user_num):
            for n in range(self.element_num):
                H_ru_indoor[n, i] = self.PL_STAR_RIS2indoor_user[i] * H_ru_indoor[n, i]
        # 将H_ru_outdoor和H_ru_indoor拼接在一起，赋值给H_ru
        H_ru = np.hstack((H_ru_outdoor, H_ru_indoor))

        # 生成一个大小为元素数*室外窃听者数的复矩阵，其中每一列的元素为cos(-n*pi*sin(STAR_RIS2outdoor_eve[i]))-1j*sin(-n*pi*sin(STAR_RIS2outdoor_eve[i]))，赋值给H_rf_LOS
        H_rf_LOS = np.zeros((self.element_num, self.outdoor_eve_num), dtype=complex)
        for i in range(self.outdoor_eve_num):
            for n in range(self.element_num):
                H_rf_LOS[n, i] = np.cos(-n * math.pi * np.sin(self.STAR_RIS2outdoor_eve[i])) - 1j * np.sin(-n * math.pi * np.sin(self.STAR_RIS2outdoor_eve[i]))
        # 生成一个大小为元素数*室外窃听者数的服从高斯分布的复矩阵
        H_rf_NLOS = np.random.randn(self.element_num, self.outdoor_eve_num) + 1j * np.random.randn(self.element_num, self.outdoor_eve_num)
        H_rf_outdoor = self.PL_STAR_RIS2outdoor_eve * math.sqrt(self.K / (self.K + 1)) * H_rf_LOS + math.sqrt(1 / (self.K + 1)) * H_rf_NLOS
        # 生成一个大小为元素数*室内窃听者数的复矩阵，其中每一列的元素为cos(-n*pi*sin(STAR_RIS2indoor_eve[i]))-1j*sin(-n*pi*sin(STAR_RIS2indoor_eve[i]))，赋值给H_rf_LOS
        H_rf_LOS = np.zeros((self.element_num, self.indoor_eve_num), dtype=complex)
        for i in range(self.indoor_eve_num):
            for n in range(self.element_num):
                H_rf_LOS[n, i] = np.cos(-n * math.pi * np.sin(self.STAR_RIS2indoor_eve[i])) - 1j * np.sin(-n * math.pi * np.sin(self.STAR_RIS2indoor_eve[i]))
        # 生成一个大小为元素数*室内窃听者数的服从高斯分布的复矩阵
        H_rf_NLOS = np.random.randn(self.element_num, self.indoor_eve_num) + 1j * np.random.randn(self.element_num, self.indoor_eve_num)
        H_rf_indoor = self.PL_STAR_RIS2indoor_eve * math.sqrt(self.K / (self.K + 1)) * H_rf_LOS + math.sqrt(1 / (self.K + 1)) * H_rf_NLOS
        # 将H_rf_outdoor和H_rf_indoor拼接在一起，赋值给H_rf
        H_rf = np.hstack((H_rf_outdoor, H_rf_indoor))


        return H_dt, G, H_du, H_de, H_rt, H_ru, H_rf

    def azimuthAngle(self, x1, y1, x2, y2):
        """
        函数用途：计算两点之间的方位角
        参数列表：点1的横纵坐标，点2的横纵坐标
        返回值：对应的方位角（弧度制）
        """
        angle = 0.0
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1:
                angle = 0.0
            elif y2 < y1:
                angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        return angle

    def get_Azimuth(self):
        """
        函数用途：计算室外目标、室外窃听者、目标、STAR-RIS之于BS的方位角；计算室外目标、室外窃听者、目标、室内用户、室内窃听者之于STAR-RIS的方位角
        参数列表：无
        返回值：室外目标、室外窃听者、目标、STAR-RIS之于BS的方位角；计算室外目标、室外窃听者、目标、室内用户、室内窃听者之于STAR-RIS的方位角（弧度制）
        """
        # 计算BS到STAR-RIS、BS到目标、BS到室外用户、BS到室外窃听者、STAR-RIS到室外用户、STAR-RIS到室外窃听者、STAR-RIS到目标、STAR-RIS到室内用户、STAR-RIS到室内窃听者的方位角
        BS2target = np.array([0] * self.target_num, dtype=float)
        BS2user = np.array([0] * self.user_num, dtype=float)
        BS2eve = np.array([0] * self.eve_num, dtype=float)
        STAR_RIS2outdoor_user = np.array([0] * self.outdoor_user_num, dtype=float)
        STAR_RIS2outdoor_eve = np.array([0] * self.outdoor_eve_num, dtype=float)
        STAR_RIS2target = np.array([0] * self.target_num, dtype=float)
        STAR_RIS2indoor_user = np.array([0] * self.indoor_user_num, dtype=float)
        STAR_RIS2indoor_eve = np.array([0] * self.indoor_eve_num, dtype=float)

        # BS2STAR_RIS = abs(math.atan2(self.STAR_RIS_loc[0][1] - self.BS_loc[0][1], self.STAR_RIS_loc[0][0] - self.BS_loc[0][0]))
        BS2STAR_RIS = self.azimuthAngle(self.BS_loc[0][0], self.BS_loc[0][1], self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1])
        for i in range(self.target_num):
            # BS2target[i] = abs(math.atan2(self.target_loc[i][1] - self.BS_loc[0][1], self.target_loc[i][0] - self.BS_loc[0][0]))
            BS2target[i] = self.azimuthAngle(self.BS_loc[0][0], self.BS_loc[0][1], self.target_loc[i][0], self.target_loc[i][1])
        for i in range(self.outdoor_user_num):
            # BS2outdoor_user[i] = abs(math.atan2(self.outdoor_user_loc[i][1] - self.BS_loc[0][1], self.outdoor_user_loc[i][0] - self.BS_loc[0][0]))
            BS2user[i] = self.azimuthAngle(self.BS_loc[0][0], self.BS_loc[0][1], self.outdoor_user_loc[i][0], self.outdoor_user_loc[i][1])
        for i in range(self.indoor_user_num):
            # BS2outdoor_user[i] = abs(math.atan2(self.outdoor_user_loc[i][1] - self.BS_loc[0][1], self.outdoor_user_loc[i][0] - self.BS_loc[0][0]))
            BS2user[i + self.outdoor_user_num] = self.azimuthAngle(self.BS_loc[0][0], self.BS_loc[0][1], self.indoor_user_loc[i][0], self.indoor_user_loc[i][1])
        for i in range(self.outdoor_eve_num):
            # BS2outdoor_eve[i] = abs(math.atan2(self.outdoor_eve_loc[i][1] - self.BS_loc[0][1], self.outdoor_eve_loc[i][0] - self.BS_loc[0][0]))
            BS2eve[i] = self.azimuthAngle(self.BS_loc[0][0], self.BS_loc[0][1], self.outdoor_eve_loc[i][0], self.outdoor_eve_loc[i][1])
        for i in range(self.outdoor_user_num):
            # STAR_RIS2outdoor_user[i] = abs(math.atan2(self.outdoor_user_loc[i][1] - self.STAR_RIS_loc[0][1], self.outdoor_user_loc[i][0] - self.STAR_RIS_loc[0][0]))
            STAR_RIS2outdoor_user[i] = self.azimuthAngle(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.outdoor_user_loc[i][0], self.outdoor_user_loc[i][1])
        for i in range(self.outdoor_eve_num):
            # STAR_RIS2outdoor_eve[i] = abs(math.atan2(self.outdoor_eve_loc[i][1] - self.STAR_RIS_loc[0][1], self.outdoor_eve_loc[i][0] - self.STAR_RIS_loc[0][0]))
            STAR_RIS2outdoor_eve[i] = self.azimuthAngle(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.outdoor_eve_loc[i][0], self.outdoor_eve_loc[i][1])
        for i in range(self.target_num):
            # STAR_RIS2target[i] = abs(math.atan2(self.target_loc[i][1] - self.STAR_RIS_loc[0][1], self.target_loc[i][0] - self.STAR_RIS_loc[0][0]))
            STAR_RIS2target[i] = self.azimuthAngle(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.target_loc[i][0], self.target_loc[i][1])
        for i in range(self.indoor_user_num):
            # STAR_RIS2indoor_user[i] = abs(math.atan2(self.indoor_user_loc[i][1] - self.STAR_RIS_loc[0][1], self.indoor_user_loc[i][0] - self.STAR_RIS_loc[0][0]))
            STAR_RIS2indoor_user[i] = self.azimuthAngle(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.indoor_user_loc[i][0], self.indoor_user_loc[i][1])
        for i in range(self.indoor_eve_num):
            # STAR_RIS2indoor_eve[i] = abs(math.atan2(self.indoor_eve_loc[i][1] - self.STAR_RIS_loc[0][1], self.indoor_eve_loc[i][0] - self.STAR_RIS_loc[0][0]))
            STAR_RIS2indoor_eve[i] = self.azimuthAngle(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.indoor_eve_loc[i][0], self.indoor_eve_loc[i][1])

        return BS2STAR_RIS, BS2target, BS2user, BS2eve, STAR_RIS2outdoor_user, STAR_RIS2outdoor_eve, STAR_RIS2target, STAR_RIS2indoor_user, STAR_RIS2indoor_eve

    def distance(self, x1, y1, x2, y2):
        """
        函数用途：计算两点之间的距离
        参数列表：点1的横纵坐标，点2的横纵坐标
        返回值：对应的距离
        """
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_distance(self):
        """
        函数用途：计算BS到STAR-RIS、BS到目标、BS到室外用户、BS到室外窃听者、STAR-RIS到室外用户、STAR-RIS到室外窃听者、STAR-RIS到目标、STAR-RIS到室内用户、STAR-RIS到室内窃听者的距离
        参数列表：无
        返回值：BS到STAR-RIS、BS到目标、BS到室外用户、BS到室外窃听者、STAR-RIS到室外用户、STAR-RIS到室外窃听者、STAR-RIS到目标、STAR-RIS到室内用户、STAR-RIS到室内窃听者的距离
        """
        BS2STAR_RIS = self.distance(self.BS_loc[0][0], self.BS_loc[0][1], self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1])
        BS2target = np.array([0] * self.target_num, dtype=float)
        BS2user = np.array([0] * self.user_num, dtype=float)
        BS2eve = np.array([0] * self.eve_num, dtype=float)
        STAR_RIS2outdoor_user = np.array([0] * self.outdoor_user_num, dtype=float)
        STAR_RIS2outdoor_eve = np.array([0] * self.outdoor_eve_num, dtype=float)
        STAR_RIS2target = np.array([0] * self.target_num, dtype=float)
        STAR_RIS2indoor_user = np.array([0] * self.indoor_user_num, dtype=float)
        STAR_RIS2indoor_eve = np.array([0] * self.indoor_eve_num, dtype=float)

        for i in range(self.target_num):
            BS2target[i] = self.distance(self.BS_loc[0][0], self.BS_loc[0][1], self.target_loc[i][0], self.target_loc[i][1])
        for i in range(self.outdoor_user_num):
            BS2user[i] = self.distance(self.BS_loc[0][0], self.BS_loc[0][1], self.outdoor_user_loc[i][0], self.outdoor_user_loc[i][1])
        for i in range(self.indoor_user_num):
            BS2user[i + self.outdoor_user_num] = self.distance(self.BS_loc[0][0], self.BS_loc[0][1], self.indoor_user_loc[i][0], self.indoor_user_loc[i][1])
        for i in range(self.eve_num):
            BS2eve[i] = self.distance(self.BS_loc[0][0], self.BS_loc[0][1], self.outdoor_eve_loc[i][0], self.outdoor_eve_loc[i][1])
        for i in range(self.outdoor_user_num):
            STAR_RIS2outdoor_user[i] = self.distance(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.outdoor_user_loc[i][0], self.outdoor_user_loc[i][1])
        for i in range(self.outdoor_eve_num):
            STAR_RIS2outdoor_eve[i] = self.distance(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.outdoor_eve_loc[i][0], self.outdoor_eve_loc[i][1])
        for i in range(self.target_num):
            STAR_RIS2target[i] = self.distance(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.target_loc[i][0], self.target_loc[i][1])
        for i in range(self.indoor_user_num):
            STAR_RIS2indoor_user[i] = self.distance(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.indoor_user_loc[i][0], self.indoor_user_loc[i][1])
        for i in range(self.indoor_eve_num):
            STAR_RIS2indoor_eve[i] = self.distance(self.STAR_RIS_loc[0][0], self.STAR_RIS_loc[0][1], self.indoor_eve_loc[i][0], self.indoor_eve_loc[i][1])

        return BS2STAR_RIS, BS2target, BS2user, BS2eve, STAR_RIS2outdoor_user, STAR_RIS2outdoor_eve, STAR_RIS2target, STAR_RIS2indoor_user, STAR_RIS2indoor_eve




# env = STAR_RIS_env(antenna_num=4, user_num=4, element_num=8, power_limit=30, target_num=4, eve_num=1)
# a = np.random.rand(env.action_dim)
# s, _, _, _, _, _ = env.step(a)
# print(s)
# print("")
# print("基站到RIS的衰落：",env.PL_BS2STAR_RIS)
# print("基站到目标的衰落：",env.PL_BS2target)
# print("基站到用户的衰落：",env.PL_BS2user)
# print("基站到窃听者的衰落：",env.PL_BS2eve)
# print("RIS到用户的衰落：",env.PL_STAR_RIS2outdoor_user)
# print("RIS到窃听者的衰落",env.PL_STAR_RIS2outdoor_eve)
# print("RIS到目标的衰落：",env.PL_STAR_RIS2target)
# print("RIS到室内用户的衰落：",env.PL_STAR_RIS2indoor_user)
# print("RIS到室内窃听者的衰落：",env.PL_STAR_RIS2indoor_eve)
# print("基站到用户的距离：",env.dBS2user)
# print("基站到窃听者的距离：",env.dBS2eve)
# print("基站到用户的信道：",env.H_du)
# print("基站到窃听者的信道：",env.H_de)
# #画出BS,STAR-RIS,user,eve,target的位置
#
# plt.scatter(env.BS_loc[0][0], env.BS_loc[0][1], c='r', marker='o', label='BS')
# plt.scatter(env.STAR_RIS_loc[0][0], env.STAR_RIS_loc[0][1], c='b', marker='o', label='STAR-RIS')
# for i in range(env.outdoor_user_num):
#     plt.scatter(env.outdoor_user_loc[i][0], env.outdoor_user_loc[i][1], c='g', marker='o', label='outdoor_user')
# for i in range(env.eve_num):
#     plt.scatter(env.outdoor_eve_loc[i][0], env.outdoor_eve_loc[i][1], c='y', marker='o', label='outdoor_eve')
# for i in range(env.target_num):
#     plt.scatter(env.target_loc[i][0], env.target_loc[i][1], c='m', marker='o', label='target')
# plt.figure()
# plt.show()
#生成一个长度为env.action_dim的随机数组

