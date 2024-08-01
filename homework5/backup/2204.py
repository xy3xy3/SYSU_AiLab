import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numba

# 设置中文字体为SimHei
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 图表绘制
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


def rank_selection(population, fitnesses, population_size):
    # 计算个体的排名
    ranked_indices = np.argsort(fitnesses)  # 返回从小到大排序的索引
    ranked_indices = ranked_indices[::-1]  # 适应度越高（数值越小），排名越前，这里假设是最小化问题，所以反转数组

    # 分配选择概率（线性排名选择）
    rank_weights = np.linspace(start=1, stop=0, num=len(population))  # 生成一个线性减少的权重数组
    # 确保权重和为1
    rank_weights /= rank_weights.sum()

    # 根据权重选择个体
    selected_indices = np.random.choice(ranked_indices, size=population_size, replace=True, p=rank_weights)

    # 根据选择的索引构建选择后的种群
    return [population[i] for i in selected_indices]


def sorted_selection(population, fitnesses, population_size):
    # 根据适应度进行排序，获取排序后的索引
    sorted_indices = np.argsort(fitnesses)

    # 选择适应度最好的个体，这里假设是最小化问题
    selected_indices = sorted_indices[:population_size]

    # 根据选择的索引构建选择后的种群
    return [population[i] for i in selected_indices]


# 顺序构造交叉算法
@numba.jit(nopython=True)
def sequential_constructive_crossover(cities, parent1, parent2):
    num_cities = cities.shape[0]
    child = np.full(num_cities, -1, dtype=np.int32)  # 初始化子代
    in_child = np.zeros(num_cities, dtype=np.bool_)  # 跟踪城市是否已在子代中

    # 建立从城市到其在parent1和parent2中位置的映射
    parent1_pos = np.empty(num_cities, dtype=np.int32)
    parent2_pos = np.empty(num_cities, dtype=np.int32)
    for i in range(num_cities):
        parent1_pos[parent1[i]] = i
        parent2_pos[parent2[i]] = i

    child[0] = parent1[0]
    in_child[parent1[0]] = True

    for i in range(1, num_cities):
        current_city = child[i - 1]
        index1 = parent1_pos[current_city]
        index2 = parent2_pos[current_city]

        # 确定两个潜在的下一个城市
        next_city1 = parent1[(index1 + 1) % num_cities]
        next_city2 = parent2[(index2 + 1) % num_cities]

        if in_child[next_city1] and in_child[next_city2]:
            # 如果两个潜在的下一个城市都已经在子代中，寻找下一个未使用的城市
            for city in range(num_cities):
                if not in_child[city]:
                    next_city = city
                    break
        else:
            # 选择两个潜在城市中未被添加且距离当前城市较近的城市
            if not in_child[next_city1] and not in_child[next_city2]:
                dist1 = np.sqrt((cities[current_city][0] - cities[next_city1][0]) ** 2 + (cities[current_city][1] - cities[next_city1][1]) ** 2)
                dist2 = np.sqrt((cities[current_city][0] - cities[next_city2][0]) ** 2 + (cities[current_city][1] - cities[next_city2][1]) ** 2)
                next_city = next_city1 if dist1 < dist2 else next_city2
            elif not in_child[next_city1]:
                next_city = next_city1
            else:
                next_city = next_city2

        child[i] = next_city
        in_child[next_city] = True

    return child


# 顺序交叉
@numba.jit(nopython=True)
def order_crossover(parent1, parent2):
    size = len(parent1)
    child = np.full(size, -1, dtype=np.int32)  # 初始化子代为-1
    marker = np.zeros(size, dtype=np.bool_)  # 标记数组，跟踪元素是否已经被添加

    # 随机选择交叉点
    start, end = np.sort(np.random.choice(size, 2, replace=False))

    # Step 1: 从第一个父代中复制一段到子代
    child[start:end + 1] = parent1[start:end + 1]
    marker[child[start:end + 1]] = True  # 标记已添加的元素

    # Step 2: 从第二个父代填充剩余的部分
    pos = (end + 1) % size  # 当前填充位置
    for gene in parent2:
        if not marker[gene]:
            child[pos] = gene
            pos = (pos + 1) % size
            if pos == start:  # 如果回到起始点，则结束填充
                break

    return child


@numba.jit(nopython=True)
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child = np.full(size, -1, dtype=np.int32)  # 初始化子代

    # 选择交叉区域
    start, end = np.sort(np.random.choice(size, 2, replace=False))

    # 预计算映射
    mapping = np.zeros(size, dtype=np.int32)
    for i in range(size):
        mapping[parent1[i]] = parent2[i]

    # 直接复制交叉区域的基因
    child[start:end + 1] = parent1[start:end + 1]

    # 填充子代的其余部分
    for i in range(size):
        if i >= start and i <= end:
            continue  # 跳过已经填充的部分
        gene = parent2[i]
        while gene in child:
            gene = mapping[gene]
        child[i] = gene

    return child
def show_run_time(start, info):
    end = time.time()
    run_time = end - start
    print(f"{info}: {run_time:.8f}秒")
def sequential_constructive_crossover_list(cities, parent1, parent2):
    num_cities = len(cities)
    child = [-1] * num_cities  # 初始化子代
    in_child = [False] * num_cities  # 跟踪城市是否已在子代中

    # 城市坐标转换为list
    cities_list = [list(city) for city in cities]

    # 建立从城市到其在parent1和parent2中位置的映射
    parent1_pos = [0] * num_cities
    parent2_pos = [0] * num_cities
    for i in range(num_cities):
        parent1_pos[parent1[i]] = i
        parent2_pos[parent2[i]] = i

    child[0] = parent1[0]
    in_child[parent1[0]] = True

    for i in range(1, num_cities):
        current_city = child[i - 1]
        index1 = parent1_pos[current_city]
        index2 = parent2_pos[current_city]

        # 确定两个潜在的下一个城市
        next_city1 = parent1[(index1 + 1) % num_cities]
        next_city2 = parent2[(index2 + 1) % num_cities]

        if in_child[next_city1] and in_child[next_city2]:
            # 如果两个潜在的下一个城市都已经在子代中，寻找下一个未使用的城市
            for city in range(num_cities):
                if not in_child[city]:
                    next_city = city
                    break
        else:
            # 选择两个潜在城市中未被添加且距离当前城市较近的城市
            dist1 = ((cities_list[current_city][0] - cities_list[next_city1][0]) ** 2 + (cities_list[current_city][1] - cities_list[next_city1][1]) ** 2) ** 0.5
            dist2 = ((cities_list[current_city][0] - cities_list[next_city2][0]) ** 2 + (cities_list[current_city][1] - cities_list[next_city2][1]) ** 2) ** 0.5
            next_city = next_city1 if dist1 < dist2 else next_city2

            if in_child[next_city1]:
                next_city = next_city2
            elif in_child[next_city2]:
                next_city = next_city1

        child[i] = next_city
        in_child[next_city] = True

    return np.array(child)

def order_crossover_list(parent1, parent2):
    #numpy转
    parent1 = parent1.tolist()
    parent2 = parent2.tolist()
    size = len(parent1)
    child = [-1] * size  # 初始化子代为-1

    # 随机选择交叉点
    start, end = sorted(random.sample(range(size), 2))

    # Step 1: 从第一个父代中复制一段到子代
    child[start:end + 1] = parent1[start:end + 1]

    # Step 2: 从第二个父代填充剩余的部分，排除已存在的元素
    fill_pos = (end + 1) % size  # 当前填充位置
    for gene in parent2:
        if gene not in child:
            child[fill_pos] = gene
            fill_pos = (fill_pos + 1) % size
            if fill_pos == start:  # 如果回到起始点，则结束填充
                break
    return np.array(child)
def pmx_crossover_list(parent1, parent2):
    #numpy转
    parent1 = parent1.tolist()
    parent2 = parent2.tolist()
    size = len(parent1)
    child = [-1] * size  # 初始化子代

    # 选择交叉区域
    start, end = sorted(random.sample(range(size), 2))

    # Step 1: 复制父代1的一部分到子代
    child[start:end+1] = parent1[start:end+1]

    # Step 2: 从父代2中填充子代的剩余部分
    for i in range(start, end + 1):
        if parent2[i] not in child:
            # 查找放置位置
            j = i
            while True:
                gene = parent2[j]
                idx = parent1.index(gene)
                if child[idx] == -1:
                    child[idx] = parent2[i]
                    break
                j = idx

    # 填充剩下的位置
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    #转回numpy
    return np.array(child)

class GeneticAlgTSP:
    # 初始化遗传算法类
    def __init__(
            self, filename: str, population_size=1000, mutation_rate=0.65
    ):
        # 备份0.72 0.027
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
        cities = np.empty((0, 2), dtype=float)  # 创建一个空的NumPy数组
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
                        city_coord = np.array([[float(parts[1]), float(parts[2])]])  # 创建一个包含城市坐标的NumPy数组
                        cities = np.vstack((cities, city_coord))  # 将城市坐标添加到NumPy数组中
                elif line.startswith("EOF"):
                    break
        return cities

    # 初始化种群
    def initialize_population(self) -> np.ndarray:
        """
        初始化种群，每个个体代表一种城市访问的路径。
        :return: 初始化的种群。
        """
        population = np.empty((self.population_size, len(self.cities)), dtype=int)  # 创建一个空的 NumPy 数组来存储种群
        for i in range(self.population_size):
            individual = np.arange(len(self.cities))  # 创建一个顺序的城市索引数组
            np.random.shuffle(individual)  # 打乱顺序，生成一个随机的个体
            population[i] = individual  # 将个体添加到种群中
        return population

    # 变异过程
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        对个体执行倒置变异操作。
        :param individual: 待变异的个体。
        :return: 变异后的个体。
        """
        # 随机选择两个不同的下标
        idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
        # 将idx1到idx2之间的部分进行倒置
        individual[idx1: idx2 + 1] = individual[idx1: idx2 + 1][::-1]
        return individual
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        #测试执行时间
        # t = time.time()
        # pmx_crossover(parent1, parent2)
        # show_run_time(t, "pmx_crossover")
        # t = time.time()
        # sequential_constructive_crossover(self.cities, parent1, parent2)
        # show_run_time(t, "sequential_constructive_crossover")
        # t = time.time()
        # order_crossover(parent1, parent2)
        # show_run_time(t, "order_crossover")
        # t = time.time()
        # pmx_crossover_list(parent1, parent2)
        # show_run_time(t, "pmx_crossover_list")
        # t = time.time()
        # order_crossover_list(parent1, parent2)
        # show_run_time(t, "order_crossover_list")
        # t = time.time()
        # sequential_constructive_crossover_list(self.cities, parent1, parent2)
        # show_run_time(t, "sequential_constructive_crossover_list")
        # os._exit(0)
        return sequential_constructive_crossover(self.cities, parent1, parent2)
    # 迭代过程
    def iterate(self, num_iterations: int):
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
            if len(self.population) > self.population_size:
                fitnesses = np.zeros(len(self.population))
                for i, individual in enumerate(self.population):
                    # 单个个体的适应度计算
                    fitnesses[i] = calculate_fitness_numba(
                        self.cities, individual
                    )
                new_population_indices = sorted_selection(
                    np.arange(len(self.population)), fitnesses, self.population_size
                )
                 # 直接使用索引获取新种群
                new_population = self.population[new_population_indices]
            else:
                new_population = self.population
            show_run_time(single_start_time, f"{iteration}选择操作")
            next_generation = []

            # 交叉、变异，并计算适应度，同时保留变异前后的child
            for i in range(0, self.population_size, 2):
                parent1 = new_population[i]
                parent2 = (
                    new_population[i + 1]
                    if i + 1 < self.population_size
                    else new_population[0]
                )

                # 交叉
                child = self.crossover(parent1, parent2)
                next_generation.append(child)
                original_child_fitness = calculate_fitness_numba(self.cities, child)  # 计算原始child的适应度

                # 更新最佳个体信息
                if original_child_fitness < best_ever_fitness:
                    best_ever_individual = child
                    best_ever_fitness = original_child_fitness
                    best_iteration = iteration

                # 变异
                if random.random() < self.mutation_rate:
                    mutated_child = self.mutate(child.copy())  # 对child进行变异，注意使用copy避免直接修改
                    next_generation.append(mutated_child)  # 添加变异后的child到下一代
                    mutated_child_fitness = calculate_fitness_numba(self.cities, mutated_child)  # 计算变异后child的适应度

                    # 更新最佳个体信息（如果变异后的child更好）
                    if mutated_child_fitness < best_ever_fitness:
                        best_ever_individual = mutated_child
                        best_ever_fitness = mutated_child_fitness
                        best_iteration = iteration

                # 将父本加入下一代（可选，根据具体算法需求决定）
                next_generation.append(parent1)
                next_generation.append(parent2)

            show_run_time(single_start_time, f"{iteration}交叉操作and变异操作and选择操作")
            self.population = np.array(next_generation)
            self.fitness_history.append(best_ever_fitness)
            show_run_time(single_start_time, f"{iteration}单次迭代时间")

        end_time = time.time()  # 记录结束时间
        run_time = end_time - start_time  # 计算运行时间
        pool.close()
        pool.join()
        # 返回最佳个体、最佳迭代次数和运行时间
        return best_ever_individual.tolist(), best_iteration, best_ever_fitness, run_time


def main():
    # ga_tsp = GeneticAlgTSP('E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\dj38.tsp')
    # best_path, best_iteration, best_ever_fitness, run_time = ga_tsp.iterate(1000)
    ga_tsp = GeneticAlgTSP("E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\ch71009.tsp")
    best_path, best_iteration, best_ever_fitness, run_time = ga_tsp.iterate(5)
    print(f"最佳路径: {best_path}")
    print(f"最佳迭代次数: {best_iteration}")
    best_ever_fitness_int = int(best_ever_fitness)
    formatted_fitness = "{:,}".format(best_ever_fitness_int)
    print(f"最佳适应度: {formatted_fitness}")
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
    best_combination, best_fitness, _ = test_different_mutation_rates(
        "E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\dj38.tsp", 1000, mutation_rates)
    print_results(best_combination, best_fitness)


if __name__ == "__main__":
    # get_best_result()
    main()