import os
import random
import copy
import numpy as np
import torch
from torch import nn, optim
from agent_dir.agent import Agent
from collections import deque

class QNetwork(nn.Module):
    # 初始化函数，接收输入大小、隐藏层大小和输出大小作为参数
    def __init__(self, input_size, hidden_size, output_size):
        # 调用父类的初始化方法
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    # 定义前向传播函数，接收输入数据
    def forward(self, inputs):
        x = torch.relu(self.fc1(inputs))
        x = torch.relu(self.fc2(x))
        return self.output(x)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        # 创建一个双端队列，其最大长度为buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        # 返回双端队列的长度，即缓冲区中元素的数量
        return len(self.buffer)

    def push(self, *transition):
        # 将经验数据添加到双端队列的末尾
        self.buffer.append(transition)

    def sample(self, batch_size):
        # 从双端队列中随机抽取batch_size个元素
        return random.sample(self.buffer, batch_size)

class AgentDQN(Agent):
    def __init__(self, env, args):
        super(AgentDQN, self).__init__(env)
        self.env = env
        self.args = args
        self.all_rewards = [] 

        # 设置随机种子
        self.seed = args.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # 初始化经验回放缓冲区和Q网络
        self.replay_buffer = ReplayBuffer(buffer_size=args.buffer_size)
        self.q_network = QNetwork(env.observation_space.shape[0], args.hidden_size, env.action_space.n)
        self.target_q_network = copy.deepcopy(self.q_network)

        # 初始化优化器、学习率调度器和损失函数
        self.lr_min = args.lr_min
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_decay_freq, gamma=args.lr_decay)
        self.loss_fn = nn.MSELoss()
        
        # 初始化参数
        self.gamma = args.gamma  # 折扣因子
        self.batch_size = args.batch_size  # 批大小
        # 更新目标网络的频率
        self.update_target_freq = args.update_target_freq
        # epsilon贪心策略的epsilon值
        self.epsilon = args.epsilon
        # epsilon贪心策略的最小epsilon值
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        # 设置日志目录
        self.log_dir = './logs'
        os.makedirs(self.log_dir, exist_ok=True)
        # 记录一局的奖励
        self.episode_rewards = 0

    # 更新Q网络
    def update_q_network(self):
        # 从经验回放缓冲区中采样
        state, action, reward, next_state, done = zip(*self.replay_buffer.sample(self.batch_size))

        # 将采样数据转换为张量
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(done)

        # 计算当前Q值和下一个状态的Q值
        q_values = self.q_network(state)
        next_q_values = self.target_q_network(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # 计算损失并更新网络
        loss = self.loss_fn(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 更新目标网络
    def update_target_network(self):
        self.target_q_network = copy.deepcopy(self.q_network)

    # 选择动作
    def make_action(self,state):
        # 根据epsilon选择动作
        if random.random() > self.epsilon:
            # 使用Q网络选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.q_network(state_tensor).max(1)[1].item()
        else:
            # 随机选择动作
            action = self.env.action_space.sample()
        return action
    
    # 记忆模块
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        # 经验回放缓冲区中的经验数据足够，更新Q网络
        if len(self.replay_buffer) > self.batch_size:
            self.update_q_network()
            # 更新epsilon值
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min
            # 更新学习率
            if self.scheduler.get_last_lr()[0] > self.lr_min:
                self.scheduler.step()
    
    # 开始训练
    def train(self):
        state, _ = self.env.reset()
        step = 0
        for frame_idx in range(1, self.args.n_frames + 1):
            done = False
            self.episode_rewards = 0
            while not done:
                step += 1
                # 随机选择动作
                action = self.make_action(state)
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # 记忆
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.episode_rewards += reward
                # 每隔一定轮数更新目标网络
                if step % self.update_target_freq == 0:
                    self.update_target_network()
                # 如果当前episode结束，重置环境
                if done:
                    self.all_rewards.append(self.episode_rewards)
                    state, _ = self.env.reset()
                    print(f"{frame_idx} reward {self.episode_rewards} epsilon {self.epsilon} step {step}")
                    break
            #循环外部
            