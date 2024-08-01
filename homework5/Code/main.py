import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numba
from numba import jit
# 设置中文字体为SimHei
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 图表绘制
def plot_results(cities, best_path, fitness_history, best_iteration):
    # 第一个图：最佳路径
    plt.figure(figsize=(5, 6))
    plt.title("最佳路径")
    city_x, city_y = zip(*cities)  # 解包城市坐标以分别获取x和y坐标列表
    plt.plot(city_x, city_y, "bo")  # 绘制所有城市点
    best_path_x = [cities[i][0] for i in best_path + [best_path[0]]]  # 获取最佳路径的x坐标
    best_path_y = [cities[i][1] for i in best_path + [best_path[0]]]  # 获取最佳路径的y坐标
    plt.plot(best_path_x, best_path_y, "r-")  # 绘制最佳路径
    plt.xlabel("x坐标")
    plt.ylabel("y坐标")
    plt.show()

    # 第二个图：适应度历史
    plt.figure(figsize=(5, 6))
    plt.title("迭代过程中的适应度")
    plt.plot(fitness_history, "b-")
    plt.xlabel("迭代次数")
    plt.ylabel("适应度")
    plt.axvline(x=best_iteration, color="r", linestyle="--", label=f"最佳迭代: {best_iteration}")
    plt.legend()
    plt.show()


# 适应度计算
@numba.jit(nopython=True)
def calculate_fitness_numba(cities, individual):
    distance = 0.0
    size = len(individual)
    for i in range(size):
        idx1, idx2 = individual[i], individual[(i + 1) % size]
        city1, city2 = cities[idx1], cities[idx2]
        distance += np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
    return distance
    # return 1 / (distance + 1e-6)



# 使用轮盘赌选择方法选择下一代个体
def rw_selection(population, fitnesses, population_size):
    # 对适应度进行归一化处理
    fitnesses /= fitnesses.sum()
    selected_indices = np.random.choice(
        np.arange(len(population)), size=population_size, replace=True, p=fitnesses
    )
    # 根据选择的索引构建选择后的种群
    return [population[i] for i in selected_indices]
#排名选择
@numba.jit(nopython=True)
def sorted_selection(population, fitnesses, population_size):
    # 根据适应度进行排序，获取排序后的索引
    sorted_indices = np.argsort(fitnesses)

    # 选择适应度最好的个体，这里假设是最小化问题
    selected_indices = sorted_indices[:population_size]

    # 根据选择的索引构建选择后的种群
    return [population[i] for i in selected_indices]
@jit(nopython=True)
def tournament_selection(population, fitnesses, population_size):
    """
    使用锦标赛选择算法选择下一代个体。
    
    :param population: 当前种群的数组。
    :param fitnesses: 种群个体的适应度数组。
    :param population_size: 种群大小，即选择出的个体数量。
    :return: 新一代的种群数组。
    """
    tournament_size = population_size // 4  # 锦标赛的大小
    # 初始化一个空数组来存储选择出的个体
    selected_indices = np.zeros(population_size, dtype=np.int32)
    for i in range(population_size):
        # 随机选择tournament_size个个体进行锦标赛
        participants = np.random.randint(0, len(population), tournament_size)
        # 选择出这些个体中适应度最好的个体
        best_participant = participants[np.argmin(fitnesses[participants])]
        # 将最佳个体加入到选择出的种群中
        selected_indices[i] = best_participant
    # 返回根据选择的索引构建的新一代种群
    return population[selected_indices]

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
                dist1 = np.sqrt((cities[current_city][0] - cities[next_city1][0]) ** 2 + (
                        cities[current_city][1] - cities[next_city1][1]) ** 2)
                dist2 = np.sqrt((cities[current_city][0] - cities[next_city2][0]) ** 2 + (
                        cities[current_city][1] - cities[next_city2][1]) ** 2)
                next_city = next_city1 if dist1 < dist2 else next_city2
            elif not in_child[next_city1]:
                next_city = next_city1
            else:
                next_city = next_city2

        child[i] = next_city
        in_child[next_city] = True

    return child


@numba.jit(nopython=True)
def process_individual(args):
    cities, parent1, parent2, mutation_rate = args
    child = sequential_constructive_crossover(cities, parent1, parent2)
    original_child_fitness = calculate_fitness_numba(cities, child)

    # 变异
    mutated_child = child
    mutated_child_fitness = original_child_fitness
    if random.random() < mutation_rate:
        mutated_child = mutate(child.copy())
        mutated_child_fitness = calculate_fitness_numba(cities, mutated_child)

    return child, original_child_fitness, mutated_child, mutated_child_fitness


@numba.njit
# 变异过程
def mutate(individual: np.ndarray) -> np.ndarray:
    """
    对个体执行倒置变异操作。
    :param individual: 待变异的个体。
    :return: 变异后的个体。
    """

    # 创建一个 Numba 随机数生成器
    def custom_random_sample(length):
        return np.random.choice(np.arange(length), size=2, replace=False)

    # 随机选择两个不同的下标
    idx1, idx2 = sorted(custom_random_sample(len(individual)))

    # 执行变异操作
    individual[idx1: idx2 + 1] = individual[idx1: idx2 + 1][::-1]

    return individual


def show_run_time(start, info):
    end = time.time()
    run_time = end - start
    print(f"{info}: {run_time:.8f}秒")


class GeneticAlgTSP:
    # 初始化遗传算法类
    def __init__(
            self, filename: str, population_size=600, mutation_rate=0.6
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

    def set(self, population_size=1000, mutation_rate=0.65):
        self.fitness_history = []
        self.population_size = population_size
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
                        city_coord = np.array([float(parts[1]), float(parts[2])])  # 创建一个包含城市坐标的NumPy数组
                       # 检查城市坐标是否已经存在于cities数组中    
                        if not any((cities == city_coord).all(1)):
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

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        # 测试执行时间
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
        with Pool(processes=16) as pool:
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
                    new_population_indices = rw_selection(
                        np.arange(len(self.population)), fitnesses, self.population_size
                    )
                    # 直接使用索引获取新种群
                    new_population = self.population[new_population_indices]
                else:
                    new_population = self.population
                show_run_time(single_start_time, f"{iteration}选择操作")
                next_generation = []

                # 交叉、变异，并计算适应度，同时保留变异前后的child
                # 生成任务参数列表
                size = len(new_population)
                tasks = [(self.cities, new_population[i],
                          new_population[(i + 1) % size],
                          self.mutation_rate) for i in range(0, size, 2)]

                # 使用pool.map并行处理任务
                results = pool.map(process_individual, tasks)
                # 处理结果
                next_generation = []
                # 复制父本
                for i in range(0, size, 2):
                    next_generation.append(new_population[i])
                    next_generation.append(new_population[(i + 1) % size])
                for child, original_child_fitness, mutated_child, mutated_child_fitness in results:
                    if original_child_fitness < best_ever_fitness:
                        best_ever_individual = child
                        best_ever_fitness = original_child_fitness
                        best_iteration = iteration
                    next_generation.append(child)
                    if not np.array_equal(child, mutated_child):
                        if mutated_child_fitness < best_ever_fitness:
                            best_ever_individual = mutated_child
                            best_ever_fitness = mutated_child_fitness
                            best_iteration = iteration
                        next_generation.append(mutated_child)
                show_run_time(single_start_time, f"{iteration}交叉操作and变异操作and选择操作")
                self.population = np.array(next_generation)
                self.fitness_history.append(best_ever_fitness)
                show_run_time(single_start_time, f"{iteration}单次迭代时间")

        end_time = time.time()  # 记录结束时间
        run_time = end_time - start_time  # 计算运行时间
        # 返回最佳个体、最佳迭代次数和运行时间
        return best_ever_individual.tolist(), best_iteration, best_ever_fitness, run_time


def main():
    # ga_tsp = GeneticAlgTSP('E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\dj38.tsp')
    # best_path, best_iteration, best_ever_fitness, run_time = ga_tsp.iterate(300)
    ga_tsp = GeneticAlgTSP("./rw1621.tsp")
    best_path, best_iteration, best_ever_fitness, run_time = ga_tsp.iterate(800)
    print(f"最佳路径: {best_path}")
    print(f"最佳迭代次数: {best_iteration}")
    best_ever_fitness_int = int(best_ever_fitness)
    formatted_fitness = "{:,}".format(best_ever_fitness_int)
    print(f"最佳适应度: {formatted_fitness}")
    print(f"运行时间: {run_time:.4f}秒")
    plot_results(ga_tsp.cities, best_path, ga_tsp.fitness_history, best_iteration)


# 测试不同的突变率
def test_seed():
    filename = "E:\\BaiduSyncdisk\\文档类\\AiLab\\homework5\\Code\\dj38.tsp"
    iterations = 300
    # 0.1-0.8 list
    mutation_rates = [i / 10 for i in range(6, 9)]
    # population sizes 100-1000
    population_sizes = [i for i in range(600, 1100, 100)]
    results = {}
    best_rate = None
    best_fitness = None
    best_size = None  # 跟踪最佳种群大小
    ga_tsp = GeneticAlgTSP(filename)
    for size in population_sizes:
        for rate in mutation_rates:
            ga_tsp.set(mutation_rate=rate, population_size=size)
            best_path, best_iteration, best_ever_fitness, run_time = ga_tsp.iterate(iterations)
            results[(size, rate)] = {  # 将种群大小和变异率作为键
                "best_path": best_path,
                "best_iteration": best_iteration,
                "best_ever_fitness": best_ever_fitness,
                "run_time": run_time
            }
            if best_fitness is None or best_ever_fitness > best_fitness:
                best_rate = rate
                best_fitness = best_ever_fitness
                best_size = size  # 更新最佳种群大小
    print_results(best_size, best_rate, best_fitness)  # 打印最佳种群大小和变异率

def print_results(size, rate, best_fitness):
    print(f"种群数量: {size}, 变异率: {rate}, 最佳适应度: {best_fitness}")



if __name__ == "__main__":
    # test_seed()
    main()