import argparse
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from agent_dir.agent_dqn import AgentDQN

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def parse():
    parser = argparse.ArgumentParser(description="Run Experiments with Different Seeds")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11038, 63415, 81247, 31472],
        help="list of seeds",
    )
    parser.add_argument(
        "--log_dir", default="./logs", type=str, help="directory for logs"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 删除logs/*.xy3dn文件，遍历
    for file in os.listdir("./logs"):
        if file.endswith(".xy3dn"):
            os.remove(os.path.join("./logs", file))
    args = parse()

    params = {
        "env_name": "CartPole-v0",
        # 隐藏层大小
        "hidden_size": 128,
        # 学习率
        "lr": 1e-3,
        "lr_min": 5e-4,
        # 学习率衰减
        "lr_decay": 0.9,
        # 更新多少次Q网络后衰减
        "lr_decay_freq": 500,
        "gamma": 0.95,
        # 跑多少轮
        "n_frames": 100,
        # 缓冲区大小
        "buffer_size": 4000,
        # 批大小
        "batch_size": 128,
        # 每隔一定步数更新目标网络
        "update_target_freq": 10,
        #多大概率从经验回放中选择动作
        "epsilon": 1.0,
        "epsilon_min": 5e-4,
        "epsilon_decay": 0.996,
    }

    all_rewards = []
    for seed in args.seeds:
        env = gym.make(params["env_name"], render_mode="rgb_array")
        args = argparse.Namespace(
            env_name=params["env_name"],
            hidden_size=params["hidden_size"],
            lr=params["lr"],
            lr_min=params["lr_min"],
            lr_decay=params["lr_decay"],
            lr_decay_freq=params["lr_decay_freq"],
            gamma=params["gamma"],
            n_frames=params["n_frames"],
            seed=seed,
            batch_size=params["batch_size"],
            buffer_size=params["buffer_size"],
            update_target_freq=params["update_target_freq"],
            epsilon = params["epsilon"],
            epsilon_min = params["epsilon_min"],
            epsilon_decay = params["epsilon_decay"],
            log_dir=args.log_dir,
        )

        agent = AgentDQN(env, args)
        agent.train()
        rewards = agent.all_rewards
        all_rewards.append(rewards)

    # 统一所有奖励的最小长度
    min_length = min([len(rewards) for rewards in all_rewards])
    all_rewards = [rewards[:min_length] for rewards in all_rewards]

    # 转换为numpy数组
    all_rewards = np.array(all_rewards)
    rewards_std = np.std(all_rewards, axis=0)

    # 创建SummaryWriter
    writer = SummaryWriter(args.log_dir)

    # 创建一个新的图
    plt.figure()

    # 记录每个实验的奖励和标准差
    for i in range(min_length):
        for j, rewards in enumerate(all_rewards):
            writer.add_scalar(f"实验{j+1}/奖励", rewards[i], i)

    for j, rewards in enumerate(all_rewards):
        plt.plot(rewards[:min_length], label=f"实验{j+1} 奖励")

    plt.plot(rewards_std[:min_length], label="奖励标准差", color="gray")

    plt.legend()
    plt.xlabel("批次")
    plt.ylabel("值")
    plt.title("实验奖励和标准差")

    # 保存图表
    plt.savefig(args.log_dir + "/merged_plot.png")

    # 关闭SummaryWriter
    writer.close()

    print(f"Rewards: {all_rewards}")
    print(f"奖励标准差: {rewards_std}")
