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
        default=[11038, 73022, 96801],
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

    fixed_params = {
        "env_name": "CartPole-v0",
        # 隐藏层大小
        "hidden_size": 32,
        # 学习率
        "lr": 0.005,
        "gamma": 0.99,
        # 跑多少轮
        "n_frames": 1000,
        # 批大小
        "batch_size": 32,
        # 缓冲区大小
        "buffer_size": 1000,
        # 多久更新Q目标网络
        "update_target_freq": 10,
        #多大概率从经验回放中选择动作
        "epsilon": 1.0  ,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.99,
        "use_cuda": True,
    }

    all_rewards = []
    for seed in args.seeds:
        env = gym.make(fixed_params["env_name"], render_mode="rgb_array")
        args = argparse.Namespace(
            env_name=fixed_params["env_name"],
            hidden_size=fixed_params["hidden_size"],
            lr=fixed_params["lr"],
            gamma=fixed_params["gamma"],
            n_frames=fixed_params["n_frames"],
            seed=seed,
            batch_size=fixed_params["batch_size"],
            buffer_size=fixed_params["buffer_size"],
            update_target_freq=fixed_params["update_target_freq"],
            epsilon = fixed_params["epsilon"],
            epsilon_min = fixed_params["epsilon_min"],
            epsilon_decay = fixed_params["epsilon_decay"],
            use_cuda=fixed_params["use_cuda"],
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
