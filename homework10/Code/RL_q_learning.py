import numpy as np
import pandas as pd

class QLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        """
        初始化 Q-learning 算法
        
        参数:
        actions: 动作列表
        learning_rate: 学习率
        reward_decay: 奖励衰减率
        e_greedy: 探索率
        """
        self.actions = actions  # 动作列表
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减率
        self.epsilon = e_greedy  # 探索率
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 初始化 Q 表

    def choose_action(self, observation):
        """
        根据当前状态选择动作
        
        参数:
        observation: 当前状态
        
        返回:
        选择的动作
        """
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            # 选择最优动作
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # 随机选择动作
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """
        学习更新 Q 表
        
        参数:
        s: 当前状态
        a: 当前动作
        r: 奖励
        s_: 下一个状态
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # 计算目标值
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新 Q 值

    def check_state_exist(self, state):
        """
        检查状态是否存在于 Q 表中，如果不存在则添加该状态
        
        参数:
        state: 状态
        """
        if state not in self.q_table.index:
            # 添加新状态到 Q 表
            new_state = pd.DataFrame([[0]*len(self.actions)], index=[state], columns=self.q_table.columns)
            self.q_table = pd.concat([self.q_table, new_state])
