"""
强化学习迷宫示例。

红色矩形：          探索者。
黑色矩形：          地狱       [奖励 = -1]。
黄色圆形目标：      天堂        [奖励 = +1]。
所有其他状态：       平地        [奖励 = 0]。

此脚本是该示例的主要部分，控制了更新方法。
"""

import time
from maze_env import Maze
from RL_q_learning import QLearning
from RL_sarsa import Sarsa

# 更新函数，用于执行学习过程
def update():
    结束 = False  # 表示是否结束一局游戏
    奖励 = 0  # 表示当前步的奖励
    for episode in range(300):  # 进行300局游戏
        print(f"开始新的一局: {episode}")
        if 结束 == True and 奖励 == 1:
            time.sleep(1)  # 如果成功到达天堂，暂停120秒
        结束 = False
        # 初始化观察
        观察值 = env.reset()

        # 选择初始行动
        行动 = RL.choose_action(str(观察值))
        while True:
            # 更新环境
            env.render()

            # 强化学习采取行动并获得下一个观察值和奖励
            下一个观察值, 奖励, 完成 = env.step(行动)

            # 强化学习选择下一个行动（针对Sarsa）
            下一个行动 = RL.choose_action(str(下一个观察值))

            # 强化学习从这个转移中学习
            if isinstance(RL, QLearning):
                RL.learn(str(观察值), 行动, 奖励, str(下一个观察值))
            elif isinstance(RL, Sarsa):
                RL.learn(str(观察值), 行动, 奖励, str(下一个观察值), 下一个行动)

            # 交换观察值和行动
            观察值, 行动 = 下一个观察值, 下一个行动

            # 当本局游戏结束时，退出循环
            if 完成:
                结束 = True
                if 奖励 == 1:  # 检查奖励是否表示成功（天堂）
                    print("成功！暂停观察...")
                    env.render()
                break

    # 游戏结束
    print('游戏结束')
    env.destroy()

# 程序主入口
if __name__ == "__main__":
    env = Maze()  # 创建迷宫环境

    # 构建强化学习类
    # RL = QLearning(actions=list(range(env.n_actions)))
    RL = Sarsa(actions=list(range(env.n_actions)))  # 使用Sarsa算法

    env.after(100, update)  # 在100毫秒后开始更新
    env.mainloop()  # 进入主循环
