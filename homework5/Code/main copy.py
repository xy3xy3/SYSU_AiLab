import time
import numpy as np
import random
import matplotlib.pyplot as plt

# 设置中文字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def show_run_time(start, info):
    end = time.time()
    run_time = end - start
    print(f"{info}: {run_time:.4f}秒")


class GeneticAlgTSP:
    # 初始化遗传算法类
    def __init__(self, filename, population_size=100, crossover_rate=0.7, mutation_rate=0.01):
        """
        初始化遗传算法解决TSP问题的类。
        :param filename: 包含城市坐标的文件名。
        :param population_size: 种群的大小。
        :param crossover_rate: 交叉率。
        :param mutation_rate: 变异率。
        """
        self.fitness_history = []
        self.cities = self.read_tsp_data(filename)
        self.population_size = population_size
        # 初始化种群
        self.population = self.initialize_population()
        # 定义交叉率和变异率
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    # 从文件读取城市坐标
    def read_tsp_data(self, filename):
        """
        从文件中读取TSP城市的坐标。
        :param filename: 文件名。
        :return: 城市坐标的numpy数组。
        """
        cities = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            in_node_coord_section = False
            for line in lines:
                line = line.strip()
                if line.startswith('NODE_COORD_SECTION'):
                    in_node_coord_section = True
                elif in_node_coord_section and line:
                    parts = line.split()
                    if len(parts) >= 3:
                        cities.append([float(parts[1]), float(parts[2])])
                elif line.startswith('EOF'):
                    break
        return np.array(cities)
    
    
    # 初始化种群
    def initialize_population(self):
        """
        初始化种群，每个个体代表一种城市访问的路径。
        :return: 初始化的种群。
        """
        population = [list(range(len(self.cities))) for _ in range(self.population_size)]
        for individual in population:
            random.shuffle(individual)
        return population

    # 计算个体的适应度
    def calculate_fitness(self, individual):
        """
        计算个体的适应度，以路径长度的倒数作为适应度值。
        :param individual: 个体，表示一种城市访问路径。
        :return: 该个体的适应度值。
        """
        distance = sum(
            np.sqrt((self.cities[individual[i]][0] - self.cities[individual[(i + 1) % len(individual)]][0]) ** 2 +
                    (self.cities[individual[i]][1] - self.cities[individual[(i + 1) % len(individual)]][1]) ** 2)
            for i in range(len(individual)))
        return 1 / distance
    # 选择过程
    def selection(self):
        """
        使用轮盘赌选择方法选择下一代个体。
        :return: 选择后的种群。
        """
        # 计算每个个体的适应度
        fitness = np.array([self.calculate_fitness(individual) for individual in self.population])
        # 对适应度进行归一化处理
        fitness /= fitness.sum()
        # 使用轮盘赌选择方法选择下一代个体
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=fitness)
         # 根据选择的索引构建选择后的种群
        return [self.population[i] for i in selected_indices]
    
    # 交叉过程
    def crossover(self, parent1, parent2):
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
        proto_child1 = parent1[:s] + parent2[s:t+1] + parent1[t+1:]
        proto_child2 = parent2[:s] + parent1[s:t+1] + parent2[t+1:]
        
        # Step 3: 确定映射关系
        mapping1 = {parent2[i]: parent1[i] for i in range(s, t+1)}
        mapping2 = {parent1[i]: parent2[i] for i in range(s, t+1)}
        
        # Step 4: 生成后代染色体
        for i in range(size):
            if i < s or i > t:
                while proto_child1[i] in mapping1:
                    proto_child1[i] = mapping1[proto_child1[i]]
                while proto_child2[i] in mapping2:
                    proto_child2[i] = mapping2[proto_child2[i]]
        return proto_child1, proto_child2

    # 变异过程
    def mutate(self, individual):
        """
        对个体执行倒置变异操作。
        :param individual: 待变异的个体。
        :return: 变异后的个体。
        """
        # 随机选择两个不同的下标
        idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
        # 将idx1到idx2之间的部分进行倒置
        individual[idx1:idx2+1] = reversed(individual[idx1:idx2+1])
        return individual

    
    # 迭代过程
       # 迭代过程
    def iterate(self, num_iterations):
        """
        进行指定次数的迭代，优化路径。
        :param num_iterations: 迭代次数。
        :return: 最佳路径。
        """
        start_time = time.time()  # 记录开始时间
        best_ever_individual = None  # 最佳个体
        best_ever_fitness = float('-inf')  # 最佳个体的适应度
        best_iteration = 0  # 最佳迭代次数
        # 开始迭代
        for iteration in range(num_iterations):
            single_start_time = time.time()  # 记录开始时间
            # 选择下一代个体
            new_population = self.selection()
            show_run_time(single_start_time, "选择操作")
            next_generation = []
            # 对每对父代个体进行交叉操作
            for i in range(0, self.population_size, 2):
                parent1 = new_population[i]
                parent2 = new_population[i+1] if i + 1 < self.population_size else new_population[0]
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
            # 对每个个体进行变异操作
            for i in range(self.population_size):
                if random.random() < self.mutation_rate:
                    next_generation[i] = self.mutate(next_generation[i])
            self.population = next_generation
            # 计算当前种群中适应度最高的个体
            best_individual = max(self.population, key=self.calculate_fitness)
            best_fitness = self.calculate_fitness(best_individual)
            # 将当前最佳个体的适应度记录下来
            self.fitness_history.append(best_fitness)
            # 如果当前最佳个体的适应度优于历史最佳个体的适应度，则更新最佳个体和相关信息
            if best_fitness > best_ever_fitness:
                best_ever_individual = best_individual
                best_ever_fitness = best_fitness
                best_iteration = iteration
            
            single_end_time = time.time()  # 记录结束时间
            run_time = single_end_time - single_start_time  # 计算运行时间
            print(f"新一轮运行时间: {run_time:.2f}秒")    

        end_time = time.time()  # 记录结束时间
        run_time = end_time - start_time  # 计算运行时间

        # 返回最佳个体、最佳迭代次数和运行时间
        return best_ever_individual, best_iteration,best_fitness, run_time

    def plot_results(self, best_path, best_iteration):
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))

        # 绘制最佳路径
        ax[0].set_title('最佳路径')
        for i, city in enumerate(self.cities):
            ax[0].plot(city[0], city[1], 'bo')
            # 在城市旁边添加编号。调整文本位置以避免与点重叠。
            ax[0].text(city[0], city[1], f' {i+1}', color='blue', fontsize=9)
        for i in range(-1, len(best_path) - 1):
            start_city = self.cities[best_path[i]]
            end_city = self.cities[best_path[i + 1]]
            ax[0].plot([start_city[0], end_city[0]], [start_city[1], end_city[1]], 'r-')
        ax[0].set_xlabel('x坐标')
        ax[0].set_ylabel('y坐标')

        # 绘制适应度历史
        ax[1].set_title('迭代过程中的适应度')
        ax[1].plot(self.fitness_history, 'b-')
        ax[1].set_xlabel('迭代次数')
        ax[1].set_ylabel('适应度')
        # 绘制最佳迭代的红色竖线
        ax[1].axvline(x=best_iteration, color='r', linestyle='--', label=f'最佳迭代: {best_iteration}')
        ax[1].legend()

        plt.show()



def main():
    ga_tsp = GeneticAlgTSP('E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\ch71009.tsp')
    # ga_tsp = GeneticAlgTSP('E:\BaiduSyncdisk\文档类\AiLab\homework5\Code\dj38.tsp')
    best_path, best_iteration,best_ever_fitness, run_time = ga_tsp.iterate(10)
    print(f"最佳路径: {best_path}")
    print(f"最佳迭代次数: {best_iteration}")
    print(f"最佳适应度: {best_ever_fitness:.4f}")
    print(f"运行时间: {run_time:.4f}秒")
    ga_tsp.plot_results(best_path, best_iteration)

if __name__ == '__main__':
    main()