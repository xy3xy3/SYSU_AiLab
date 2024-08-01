import re

# 最一般合一算法
def MGU(f1, f2):
    # 获取谓词的参数列表
    def get_param(f):
        # 去除可能的否定标记以便提取参数
        if f.startswith('~'):
            f = f[1:]
        match = re.match(r'(\w+)\((.*)\)', f)
        if match:
            terms = match.group(2).split(',')
            return terms
        return []

    # 判断是否是单个变量
    def is_variable(term):
        return re.match(r'^[u-z]{1,2}$', term) is not None

    # 在term中替换所有出现的var为value
    def substitute(var, value, term):
        # 确保仅替换整个单词，避免部分替换
        return re.sub(r'\b' + var + r'\b', value, term)

    p1 = get_param(f1)
    p2 = get_param(f2)
    res = {}
    queue = list(zip(p1, p2))

    while queue:
        e1, e2 = queue.pop(0)
        check1 = is_variable(e1)
        check2 = is_variable(e2)
        if e1 == e2:
            continue
        elif check1 and check2:
            continue
        elif check1:
            if e1 in e2:
                return {}  # 变量循环，合一失败
            else:
                res[e1] = e2
                # 替换队列中剩余的项
                queue = [(substitute(e1, e2, x), y) for x, y in queue]
        elif check2:
            if e2 in e1:
                return {}  # 变量循环，合一失败
            else:
                res[e2] = e1
                # 替换队列中剩余的项
                queue = [(x, substitute(e2, e1, y)) for x, y in queue]
        else:
            match1 = re.match(r'(\w+)\((.*)\)', e1)
            match2 = re.match(r'(\w+)\((.*)\)', e2)
            if match1 and match2 and match1.group(1) == match2.group(1):
                subterms1 = match1.group(2).split(',')
                subterms2 = match2.group(2).split(',')
                queue.extend(zip(subterms1, subterms2))
            else:
                return {}  # 谓词不匹配或不是复合项，合一失败
    return res

# 一阶逻辑的归结推理
def ResolutionFOL(KB):
    # 将子句格式化为字符串
    def format_clause(clause):
        if len(clause) == 0:
            return "()"
        else:
            return "(" + ", ".join(clause) + ")"

    # 对字面量进行否定
    def negate(literal):
        if literal.startswith("~"):
            return literal[1:]
        else:
            return "~" + literal

    # 判断是否是单个变量
    def is_variable(term):
        return re.match(r'^[u-z]{1,2}$', term) is not None
    # 解析字面量
    def parse_literal(literal):
        match = re.match(r'(~?)(\w+)\((.*)\)', literal)
        if match:
            return {'neg': match.group(1) == '~', 'pred': match.group(2), 'terms': match.group(3).split(',')}
        return None

    # 对两个字面量进行最一般合一
    def unify_literals(li, lj):
        parsed_li = parse_literal(li)
        parsed_lj = parse_literal(lj)
        # 要求，谓词相同且符号不同
        if parsed_li['pred'] == parsed_lj['pred'] and parsed_li['neg']!=parsed_lj['neg']:
            res = MGU(li, lj)
            #如果res为{}，考虑可不可以消去
            if res != {}:
                return res
            # 说明无法替换变量解决，尝试是不是一模一样
            if len(parsed_li['terms']) == len(parsed_lj['terms']):
                # 判断所有都相同
                for i in range(len(parsed_li['terms'])):
                    if parsed_li['terms'][i]!=parsed_lj['terms'][i]:
                        return None
                return {"x":"x"}
            else:
                return None
        return None

    # 解析子句中的每个文字，记录其细节
    def parse_clause(clause):
        return [(i + 1, literal) for i, literal in enumerate(clause)]  # 返回形式为[(index, literal)]
    # 应用替换
    def apply_substitution(clause, substitution):
        new_clause = []
        for literal in clause:
            new_literal = literal
            for var, term in substitution.items():
                new_literal = new_literal.replace(var, term)
            new_clause.append(new_literal)
        return tuple(new_clause)
    # 生成归结步骤的描述字符串
    def generate_step_description(id, id_i, id_j, substitution, new_resolvent):
        # 过滤掉那些键和值相同的替换条目
        meaningful_substitution = {k: v for k, v in substitution.items() if k != v}

        # 检查是否有有意义的替换进行
        if meaningful_substitution:
            # 如果有有意义的替换，则添加替换信息
            substitution_str = ', '.join([f'{k}={v}' for k, v in meaningful_substitution.items()])
            return f"{id} R[{id_i},{id_j}]{{{substitution_str}}} = {format_clause(new_resolvent)}"
        else:
            # 如果没有有效的替换或者替换字典为空，则不添加替换信息
            return f"{id} R[{id_i},{id_j}] = {format_clause(new_resolvent)}"

    steps = []  # 存储推理步骤
    id = 1  # 步骤编号
    original_clauses = list(KB)  # 原始子句集
    clause_map = {}  # 子句到编号的映射
    new = set()  # 新产生的子句集

    # 记录初始子句
    for clause in original_clauses:
        clause_map[clause] = str(id)
        steps.append(f"{id}. {format_clause(clause)}")
        id += 1

    while True:
        n = len(original_clauses)
        found_empty_clause = False
        # 循环所有子句
        for i in range(n):
            for j in range(i + 1, n):
                if (i==j):
                    continue
                # 解析子句为文字list
                parsed_ci = parse_clause(original_clauses[i])
                parsed_cj = parse_clause(original_clauses[j])
                # 遍历文字
                for index_i, li in parsed_ci:
                    for index_j, lj in parsed_cj:
                        substitution = unify_literals(li, lj)
                        if substitution is not None:
                            # 删去互补的文字
                            new_resolvent = tuple(sorted(set(original_clauses[i] + original_clauses[j]) - {li, lj}))
                            # 替换变量
                            new_resolvent = apply_substitution(new_resolvent, substitution)
                            # if len(new_resolvent) == 2 and negate(new_resolvent[0]) == new_resolvent[1]:
                            #     new_resolvent = ()  # 将新解集设置为空子句表示矛盾
                            #     found_empty_clause = True  # 标记已找到矛盾
                            #     # 生成对应文字的编号
                            #     id_i = f"{clause_map[original_clauses[i]]}{chr(96 + index_i)}" if len(original_clauses[i]) > 1 else clause_map[original_clauses[i]]
                            #     id_j = f"{clause_map[original_clauses[j]]}{chr(96 + index_j)}" if len(original_clauses[j]) > 1 else clause_map[original_clauses[j]]
                            #     steps.append(generate_step_description(id, id_i, id_j, substitution, new_resolvent))
                            #     new.add(new_resolvent)
                            #     clause_map[new_resolvent] = str(id)
                            #     id += 1
                            #     break
                            # el
                            if new_resolvent not in original_clauses and new_resolvent not in new:
                                # 生成对应文字的编号
                                id_i = f"{clause_map[original_clauses[i]]}{chr(96 + index_i)}" if len(original_clauses[i]) > 1 else clause_map[original_clauses[i]]
                                id_j = f"{clause_map[original_clauses[j]]}{chr(96 + index_j)}" if len(original_clauses[j]) > 1 else clause_map[original_clauses[j]]
                                steps.append(generate_step_description(id, id_i, id_j, substitution, new_resolvent))
                                new.add(new_resolvent)
                                clause_map[new_resolvent] = str(id)
                                id += 1
                            if new_resolvent == ():
                                found_empty_clause = True
                                break
                    if found_empty_clause:
                        break
                if found_empty_clause:
                    break
            if found_empty_clause:
                break
        # 找到NIL，或者新的已经出现过，则结束
        if found_empty_clause or new.issubset(set(original_clauses)):
            # print(set(original_clauses))
            break
        original_clauses.extend(new)
        # 排序以便步骤更少
        original_clauses = sorted(original_clauses, key=lambda x: len(x))
        
    shortest_chain = find_shortest_chain(steps, KB)
    updated_chain = update_step_numbers(shortest_chain)
    return updated_chain
# 反向追踪以找到最短的推理链
def find_shortest_chain(steps, KB):
    # 解析步骤，提取步骤编号和依赖的步骤编号
    def parse_step(step):
        match = re.search(r'^(\d+)\sR\[(\d+)([a-z]?),(\d+)([a-z]?)\]', step)
        if match:
            id = int(match.group(1))
            depends = [int(match.group(2)), int(match.group(4))]
            return id, depends
        return None, []

    # 构建步骤依赖图
    step_deps = {}
    for step in steps:
        id, deps = parse_step(step)
        if id:
            step_deps[id] = deps

    # 反向追踪
    necessary_steps = set()
    queue = [max(step_deps.keys())]  # 从最后一个步骤开始追踪
    while queue:
        current = queue.pop(0)
        if current in necessary_steps:
            continue
        necessary_steps.add(current)
        if current in step_deps:  # 如果当前步骤依赖其他步骤
            queue.extend(step_deps[current])  # 将依赖步骤加入追踪队列

    # 构建最短推理链
    shortest_chain = []
    for step in steps:
        match = re.match(r'^(\d+)', step)
        if match and int(match.group(1)) in necessary_steps:
            shortest_chain.append(step)

    return shortest_chain
def update_step_numbers(shortest_chain):
    # 分析每个步骤，提取原始编号
    original_numbers = {}
    for i, step in enumerate(shortest_chain, 1):
        match = re.match(r'^(\d+)', step)
        if match:
            original_number = int(match.group(1))
            original_numbers[original_number] = i  # 将原始编号映射到新编号

    # 更新步骤中的编号和依赖关系
    updated_steps = []
    for step in shortest_chain:
        # 更新步骤编号
        new_number = original_numbers[int(re.match(r'^(\d+)', step).group(1))]
        step = re.sub(r'^\d+', str(new_number), step)

        # 更新依赖的步骤编号
        def replace_dep(match):
            dep1 = match.group(1)
            dep2 = match.group(3)
            dep1_suffix = match.group(2)
            dep2_suffix = match.group(4)
            new_dep1 = original_numbers.get(int(dep1), dep1) if dep1.isdigit() else dep1
            new_dep2 = original_numbers.get(int(dep2), dep2) if dep2.isdigit() else dep2
            return f"R[{new_dep1}{dep1_suffix},{new_dep2}{dep2_suffix}]"

        step = re.sub(r'R\[(\d+)([a-z]?),(\d+)([a-z]?)\]', replace_dep, step)
        updated_steps.append(step)

    return updated_steps
def test(KB):
    print("测试开始")
    steps = ResolutionFOL(KB)
    for step in steps:
        print(step)
if __name__ == "__main__":
    if True:
        KB = {("GradStudent(sue)",),("~GradStudent(x)","Student(x)"),("~Student(x)","HardWorker(x)"),("~HardWorker(sue)",)}
        test(KB)
        KB = {("A(tony)",),("A(mike)",),("A(john)",),("L(tony,rain)",),("L(tony,snow)",),("~A(x)","S(x)","C(x)"),("~C(y)","~L(y,rain)"),("L(z,snow)","~S(z)"),("~L(tony,u)","~L(mike,u)"),("L(tony,v)","L(mike,v)"),("~A(w)","~C(w)","S(w)")}
        test(KB)

        KB = {("On(tony,mike)",),("On(mike,john)",),("Green(tony)",),("~Green(john)",),("~On(xx,yy)","~Green(xx)","Green(yy)")}
        test(KB)
    else:
        # 自己debug
        KB = {("L(tony,u)","L(mike,u)",),("~L(tony,u)","~L(mike,u)",)}
        test(KB)