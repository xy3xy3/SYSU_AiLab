def ResolutionProp(KB):
    steps = []  # 存储归结过程的步骤
    id = 1  # 步骤编号
    original_clauses = list(KB)  # 初始的子句集
    clause_map = {}  # 映射每个子句到它的编号
    new = set()  # 存储新产生的子句

    # 初始化步骤和子句编号映射
    for clause in original_clauses:
        clause_map[clause] = str(id)
        steps.append(f"{id}. {format_clause(clause)}")
        id += 1

    while True:
        n = len(original_clauses)
        found_empty_clause = False
        # 遍历旧的，尝试产生新的子句
        for i in range(n):
            for j in range(i + 1, n):
                resolvents, used_literal_i, used_literal_j = resolve_with_details(original_clauses[i], original_clauses[j])
                for resolvent in resolvents:
                    if resolvent not in original_clauses and resolvent not in new:
                        # 格式化新的子句
                        clause_str = format_clause(resolvent)
                        step = f"{id} R[{clause_map[original_clauses[i]]}{used_literal_i},{clause_map[original_clauses[j]]}{used_literal_j}] = {clause_str}"
                        # 添加到step与新子句
                        steps.append(step)
                        new.add(resolvent)
                        clause_map[resolvent] = str(id)
                        id += 1
                    if resolvent == ():
                        # 找到了nil，可以推出
                        found_empty_clause = True
                        break
                if found_empty_clause:
                    break
            if found_empty_clause:
                break
        if found_empty_clause or new.issubset(set(original_clauses)):
            break
        original_clauses.extend(new)
    return steps

def resolve_with_details(ci, cj):
    """
    返回归结后的子句以及用到的id
    """
    resolvents = set()
    used_literal_i = ''
    used_literal_j = ''
    # 遍历两个子句，找到可以归结的文字
    for index_i, di in enumerate(ci):
        for index_j, dj in enumerate(cj):
            if di == negate(dj) or negate(di) == dj:
                # 删去被归结的
                resolvent = tuple(sorted(set(ci + cj) - {di, dj}))
                resolvents.add(resolvent)
                if len(ci) > 1:
                    used_literal_i = chr(97 + index_i)  # 用字母a、b、c等表示
                if len(cj) > 1:
                    used_literal_j = chr(97 + index_j)
    return resolvents, used_literal_i, used_literal_j

def negate(literal):
    # 取反逻辑
    if literal.startswith("~"):
        return literal[1:]
    else:
        return "~" + literal

def format_clause(clause):
    # 格式化子句为字符串
    if len(clause) == 0:
        return "()"
    else:
        return "(" + ", ".join(clause) + ")"

if __name__ == "__main__":
    KB = {("FirstGrade",), ("~FirstGrade","Child"), ("~Child",)}
    steps = ResolutionProp(KB)
    for step in steps:
        print(step)
