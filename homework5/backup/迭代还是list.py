import time
import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numba

# 设置中文字体为SimHei
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

#图表绘制
def plot_results(cities, best_path, fitness_history, best_iteration):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # 绘制最佳路径
    ax[0].set_title("最佳路径")
    city_x, city_y = zip(*cities)  # 解包城市坐标以分别获取x和y坐标列表
    ax[0].plot(city_x, city_y, "bo")  # 绘制所有城市点
    best_path_x = [
        cities[i][0] for i in best_path + [best_path[0]]
    ]  # 获取最佳路径的x坐标
    best_path_y = [
        cities[i][1] for i in best_path + [best_path[0]]
    ]  # 获取最佳路径的y坐标
    ax[0].plot(best_path_x, best_path_y, "r-")  # 绘制最佳路径
    ax[0].set_xlabel("x坐标")
    ax[0].set_ylabel("y坐标")

    # 绘制适应度历史
    ax[1].set_title("迭代过程中的适应度")
    ax[1].plot(fitness_history, "b-")
    ax[1].set_xlabel("迭代次数")
    ax[1].set_ylabel("适应度")
    ax[1].axvline(
        x=best_iteration, color="r", linestyle="--", label=f"最佳迭代: {best_iteration}"
    )
    ax[1].legend()

    plt.show()

# 适应度计算
@numba.jit(nopython=True)
def calculate_fitness_numba(cities, individual):
    distance = 0.0
    for i in range(len(individual)):
        idx1, idx2 = individual[i], individual[(i + 1) % len(individual)]
        city1, city2 = cities[idx1], cities[idx2]
        distance += np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
    return distance
    # return 1 / (distance + 1e-6)

# 选择过程
def selection(population, fitnesses, population_size):
    # 对适应度进行归一化处理
    fitnesses /= fitnesses.sum()
    # 使用轮盘赌选择方法选择下一代个体
    selected_indices = np.random.choice(
        np.arange(len(population)), size=population_size, replace=True, p=fitnesses
    )
    # 根据选择的索引构建选择后的种群
    return [population[i] for i in selected_indices]
#顺序构造交叉算法
@numba.jit(nopython=True)
def sequential_constructive_crossover_numba(cities, parent1, parent2):
    num_cities = cities.shape[0]
    child = np.full(num_cities, -1, dtype=np.int32)  # 使用numpy数组代替列表
    child[0] = parent1[0]
    
    for i in range(1, num_cities):
        current_city = child[i - 1]
        # 查找当前城市在父代路径中的索引
        index1 = np.where(parent1 == current_city)[0][0]
        index2 = np.where(parent2 == current_city)[0][0]
        
        # 获取下一个城市
        next_city1 = parent1[(index1 + 1) % num_cities]
        next_city2 = parent2[(index2 + 1) % num_cities]
        
        # 检查下一个城市是否已在子代路径中
        if not np.any(child == next_city1) and not np.any(child == next_city2):
            # 手动计算两点之间的距离
            dist1 = np.sqrt((cities[current_city][0] - cities[next_city1][0])**2 + (cities[current_city][1] - cities[next_city1][1])**2)
            dist2 = np.sqrt((cities[current_city][0] - cities[next_city2][0])**2 + (cities[current_city][1] - cities[next_city2][1])**2)
            child[i] = next_city1 if dist1 < dist2 else next_city2
        elif not np.any(child == next_city1):
            child[i] = next_city1
        elif not np.any(child == next_city2):
            child[i] = next_city2
        else:
            # 找一个未使用的城市
            for city in range(num_cities):
                if not np.any(child == city):
                    child[i] = city
                    break
    return child
#顺序交叉
@numba.jit(nopython=True)
def order_crossover(parent1, parent2):
    size = len(parent1)
    child = np.full(size, -1, dtype=np.int32)  # 初始化子代为-1

    # 随机选择交叉点
    start, end = np.sort(np.random.choice(size, 2, replace=False))

    # Step 1: 从第一个父代中复制一段到子代
    child[start:end+1] = parent1[start:end+1]

    # Step 2: 从第二个父代填充剩余的部分
    position = (end + 1) % size  # 当前填充位置
    for i in range(size):
        # 如果当前位置已经填充，则跳过
        if parent2[i] in child:
            continue
        child[position] = parent2[i]
        position = (position + 1) % size  # 更新填充位置

    return child


def show_run_time(start, info):
    end = time.time()
    run_time = end - start
    print(f"{info}: {run_time:.4f}秒")


class GeneticAlgTSP:
    # 初始化遗传算法类
    def __init__(
        self, filename: str, population_size=1000, mutation_rate=0.65
    ):
        #备份0.72 0.027
        """
        初始化遗传算法解决TSP问题的类。
        :param filename: 包含城市坐标的文件名。
        :param population_size: 种群的大小。
        :param mutation_rate: 变异率。
        """
        self.fitness_history = []
        self.cities = self.read_tsp_data(filename)
        self.population_size = population_size
        # 初始化种群
        self.population = self.initialize_population()
        # 定义交叉率和变异率
        self.mutation_rate = mutation_rate

    # 从文件读取城市坐标
    def read_tsp_data(self, filename: str):
        """
        从文件中读取TSP城市的坐标。
        :param filename: 文件名。
        :return: 城市坐标的numpy数组。
        """
        cities = []
        with open(filename, "r") as f:
            lines = f.readlines()
            in_node_coord_section = False
            for line in lines:
                line = line.strip()
                if line.startswith("NODE_COORD_SECTION"):
                    in_node_coord_section = True
                elif in_node_coord_section and line:
                    parts = line.split()
                    if len(parts) >= 3:
                        cities.append([float(parts[1]), float(parts[2])])
                elif line.startswith("EOF"):
                    break
        return np.array(cities)

    # 初始化种群
    def initialize_population(self):
        """
        初始化种群，每个个体代表一种城市访问的路径。
        :return: 初始化的种群。
        """
        population = [
            list(range(len(self.cities))) for _ in range(self.population_size)
        ]
        for individual in population:
            random.shuffle(individual)
        return population

    # 交叉过程
    def crossover(self, parent1: list, parent2: list):
        """
        对两个个体执行交叉操作，生成新的个体。
        :param parent1: 父代个体1。
        :param parent2: 父代个体2。
        :return: 生成的子代个体。
        """
        size = len(self.cities)
        # Step 1: 随机选择下标s, t
        s, t = sorted(random.sample(range(size), 2))
        # Step 2: 交叉子串
        proto_child1 = parent1[:s] + parent2[s : t + 1] + parent1[t + 1 :]
        proto_child2 = parent2[:s] + parent1[s : t + 1] + parent2[t + 1 :]

        # Step 3: 确定映射关系
        mapping1 = {parent2[i]: parent1[i] for i in range(s, t + 1)}
        mapping2 = {parent1[i]: parent2[i] for i in range(s, t + 1)}

        # Step 4: 生成后代染色体
        for i in range(size):
            if i < s or i > t:
                while proto_child1[i] in mapping1:
                    proto_child1[i] = mapping1[proto_child1[i]]
                while proto_child2[i] in mapping2:
                    proto_child2[i] = mapping2[proto_child2[i]]
        return proto_child1, proto_child2

    # 变异过程
    def mutate(self, individual: list) -> list:
        """
        对个体执行倒置变异操作。
        :param individual: 待变异的个体。
        :return: 变异后的个体。
        """
        # 随机选择两个不同的下标
        idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
        # 将idx1到idx2之间的部分进行倒置
        individual[idx1 : idx2 + 1] = reversed(individual[idx1 : idx2 + 1])
        return individual

    # 迭代过程
    def iterate(self, num_iterations: int) -> list:
        """
        进行指定次数的迭代，优化路径。
        :param num_iterations: 迭代次数。
        :return: 最佳路径。
        """
        start_time = time.time()  # 记录开始时间
        best_ever_individual = None  # 最佳个体
        best_ever_fitness = float("+inf")  # 最佳个体的适应度
        best_iteration = 0  # 最佳迭代次数
        pool = Pool(processes=4)

        # 开始迭代
        for iteration in range(num_iterations):
            single_start_time = time.time()  # 记录开始时间
            # 选择下一代个体
            fitnesses = np.zeros(len(self.population))
            for i, individual in enumerate(self.population):
                # 单个个体的适应度计算
                fitnesses[i] = calculate_fitness_numba(
                    self.cities, np.array(individual)
                )
            new_population_indices = selection(
                np.arange(len(self.population)), fitnesses, self.population_size
            )
            new_population = [self.population[i] for i in new_population_indices]
            show_run_time(single_start_time, f"{iteration}选择操作")
            next_generation = []
            # 对每对父代个体进行交叉操作
            for i in range(0, self.population_size, 2):
                parent1 = new_population[i]
                parent2 = (
                    new_population[i + 1]
                    if i + 1 < self.population_size
                    else new_population[0]
                )
                # child = sequential_constructive_crossover_numba(self.cities,np.array(parent1, dtype=np.int32), np.array(parent2, dtype=np.int32))
                child = order_crossover(np.array(parent1, dtype=np.int32), np.array(parent2, dtype=np.int32))
                # 转列表
                child = child.tolist()
                next_generation.append(child)
            show_run_time(single_start_time, f"{iteration}交叉操作")
            # 对每个个体进行变异操作
            for i in range(len(next_generation)):
                if random.random() < self.mutation_rate:
                    next_generation[i] = self.mutate(next_generation[i])
            self.population = next_generation
            show_run_time(single_start_time, f"{iteration}变异操作")
            # 计算当前种群中适应度最高的个体
            for i, individual in enumerate(self.population):
                # 单个个体的适应度计算
                fitness = calculate_fitness_numba(self.cities, np.array(individual))
                if fitness < best_ever_fitness:
                    best_ever_individual = next_generation[i]
                    best_ever_fitness = fitness
                    best_iteration = iteration

            self.population = next_generation
            self.fitness_history.append(best_ever_fitness)

            show_run_time(single_start_time, f"{iteration}单次迭代时间")

        end_time = time.time()  # 记录结束时间
        run_time = end_time - start_time  # 计算运行时间
        pool.close()
        pool.join()
        # 返回最佳个体、最佳迭代次数和运行时间
        return best_ever_individual, best_iteration, best_ever_fitness, run_time

def main():
    ga_tsp = GeneticAlgTSP('E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\dj38.tsp')
    best_path, best_iteration,best_ever_fitness, run_time = ga_tsp.iterate(500)
    # ga_tsp = GeneticAlgTSP("E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\ch71009.tsp")
    # best_path, best_iteration, best_ever_fitness, run_time = ga_tsp.iterate(500)
    print(f"最佳路径: {best_path}")
    print(f"最佳迭代次数: {best_iteration}")
    print(f"最佳适应度: {best_ever_fitness}")
    print(f"运行时间: {run_time:.4f}秒")
    plot_results(ga_tsp.cities, best_path, ga_tsp.fitness_history, best_iteration)
# 测试不同的突变率
def test_different_mutation_rates(filename, iterations, mutation_rates):
    results = {}
    best_rate = None
    best_fitness = None
    for rate in mutation_rates:
        ga_tsp = GeneticAlgTSP(filename, mutation_rate=rate)
        best_path, best_iteration, best_ever_fitness, run_time = ga_tsp.iterate(iterations)
        results[rate] = {
            "best_path": best_path,
            "best_iteration": best_iteration,
            "best_ever_fitness": best_ever_fitness,
            "run_time": run_time
        }
        if best_fitness is None or best_ever_fitness > best_fitness:
            best_rate = rate
            best_fitness = best_ever_fitness
    return best_rate, best_fitness, results

def print_results(rate, best_fitness):
    print(f" {rate}, 最佳适应度: {best_fitness}")
def get_best_result():
    crossover_rates = [round(i * 0.01, 2) for i in range(70, 91)]  # 0.7到0.9，0.01为间隔
    mutation_rates = [round(i * 0.001 + 0.04, 3) for i in range(0, 31)]  # 0.04到0.07，间隔0.001
    best_combination, best_fitness,_ = test_different_mutation_rates("E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\dj38.tsp", 1000, mutation_rates)
    print_results(best_combination, best_fitness)
if __name__ == "__main__":
    # get_best_result()
    main()