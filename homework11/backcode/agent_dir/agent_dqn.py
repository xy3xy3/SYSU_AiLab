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
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.relu3 = nn.ReLU()
        # 输出层
        self.output = nn.Linear(hidden_size, output_size)

    # 定义前向传播函数，接收输入数据
    def forward(self, inputs):
        x = self.relu1(self.fc1(inputs))
        x = self.relu2(self.fc2(x))
        # x = self.relu3(self.fc3(x))
        x = self.output(x)
        return x


class ReplayBuffer:
    # 初始化函数，接收一个参数buffer_size，表示缓冲区的大小
    def __init__(self, buffer_size):
        # 设置缓冲区的大小
        self.buffer_size = buffer_size
        # 创建一个双端队列，其最大长度为buffer_size
        self.buffer = deque(maxlen=buffer_size)

    # 定义一个特殊方法，用于返回缓冲区中元素的数量
    def __len__(self):
        # 返回双端队列的长度，即缓冲区中元素的数量
        return len(self.buffer)

    # 定义一个方法，用于向缓冲区中添加一个或多个经验数据
    def push(self, *transition):
        # 将经验数据添加到双端队列的末尾
        self.buffer.append(transition)

    # 定义一个方法，用于从缓冲区中随机抽取一批经验数据
    def sample(self, batch_size):
        # 从双端队列中随机抽取batch_size个元素
        return random.sample(self.buffer, batch_size)

    # 定义一个方法，用于清空缓冲区中的所有经验数据
    def clean(self):
        # 清空双端队列，即删除所有元素
        self.buffer.clear()


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
        self.q_network = QNetwork(
            env.observation_space.shape[0], args.hidden_size, env.action_space.n
        )
        self.target_q_network = copy.deepcopy(self.q_network)

        # 初始化优化器、学习率调度器和损失函数
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
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
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_rewards = []
        self.losses = []
        self.best_reward = -float("inf")

    # 更新Q网络
    def update_q_network(self):
        # 从经验回放缓冲区中采样
        state, action, reward, next_state, done = zip(
            *self.replay_buffer.sample(self.batch_size)
        )

        # 将采样数据转换为张量
        state = torch.FloatTensor(np.array(state))
        action = torch.cat(action).unsqueeze(1)
        reward = torch.cat(reward)
        next_state = torch.cat(next_state)
        done = torch.cat(done)

        # 计算当前Q值和下一个状态的Q值
        q_values = self.q_network(state)
        next_q_values = self.target_q_network(next_state)

        q_value = q_values.gather(1, action).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # 计算损失并更新网络
        loss = self.loss_fn(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

    # 软更新目标网络
    def soft_update_target_network(self):
        self.target_q_network = copy.deepcopy(self.q_network)

    # 选择动作
    def make_action(self, observation, test=True):
        state = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            action = self.q_network(state).max(1)[1].item()
        return action

    # 运行代理
    def train(self):
        state, _ = self.env.reset()
        for frame_idx in range(1, self.args.n_frames + 1):
            done = False
            cnt = 0
            while not done:
                # 根据epsilon选择动作
                if random.random() > self.epsilon:
                    # 使用Q网络选择动作
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action = self.q_network(state_tensor).max(1)[1].item()
                else:
                    # 随机选择动作
                    action = self.env.action_space.sample()

                # 执行动作并存储经验
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                action_tensor = torch.LongTensor([action])
                reward_tensor = torch.FloatTensor([reward if not done else -10])
                done_tensor = torch.FloatTensor([done])
                self.episode_rewards.append(reward)
                state = next_state
                if done:
                    # 重置环境
                    state, _ = self.env.reset()
                    # 计算当前回合的总奖励
                    episode_reward = sum(self.episode_rewards)
                    self.all_rewards.append(episode_reward)
                    self.episode_rewards = []
                    print(f"{frame_idx} reward {episode_reward}")
                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward
                # 将经验存储到经验回放缓冲区
                self.replay_buffer.push(
                    state, action_tensor, reward_tensor, next_state_tensor, done_tensor
                )
                # 经验回放缓冲区中的经验数据足够，更新Q网络
                if len(self.replay_buffer) > self.batch_size:
                    self.update_q_network()
                # 更新计数器
                cnt = cnt + 1
                # 每隔一定步数更新目标网络
                if cnt % self.update_target_freq == 0 or done:
                    self.soft_update_target_network()
            # 更新epsilon
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay
