from LogicClass import Literal, Clause
import re


def is_variable(term):
    """
    检查一个项是否是变量。
    假设变量是小写字母表示的。
    """
    if re.match(r'^[a-z]$', term) or re.match(r'^[a-z]{2}$', term):
        return True
    else:
        return False


def apply_substitution(literal, substitution):
    """
    将替换应用于文字的项。
    """
    new_terms = [substitution.get(term, term) for term in literal.terms]
    return Literal(literal.negated, literal.predicate, new_terms)


def tuple_to_clause(t):
    literals = []
    for item in t:
        negated = False
        if item.startswith("~"):
            negated = True
            item = item[1:]

        match = re.match(r'(\w+)\((.*?)\)', item)
        if match:
            predicate = match.group(1)
            terms = match.group(2).split(',')
            literals.append(Literal(negated, predicate, terms))

    return Clause(literals)


def resolve(clause1, clause2):
    """
    尝试合并两个子句
    """
    for lit1 in clause1.literals:
        for lit2 in clause2.literals:
            # 检查谓词是否相同且一个是另一个的否定
            if lit1.predicate == lit2.predicate and lit1.negated != lit2.negated:
                # 尝试合一lit1和lit2的参数
                substitution = {}
                can_unify = True
                for term1, term2 in zip(lit1.terms, lit2.terms):
                    sub = unify(term1, term2)
                    if sub is None:
                        can_unify = False
                        break
                    substitution.update(sub)
                if can_unify:
                    # 创建新的子句，应用替换，并删除归结的文字
                    new_literals = set()
                    for lit in clause1.literals.union(clause2.literals) - {lit1, lit2}:
                        new_literal = apply_substitution(lit, substitution)
                        new_literals.add(new_literal)
                    # 构建替换信息的字符串
                    sub_str = "{"+", ".join([f"{key}={value}" for key, value in substitution.items()])+"}"
                    return Clause(new_literals), sub_str
    return None, ""  # 如果没有找到合一的文字对，返回None和空字符串


def unify(term1, term2):
    """
    尝试合一两个项，返回替换（如果合一成功）或None（如果合一失败）。
    term1和term2是字符串。
    """
    if term1 == term2:
        return {}  # 没有需要的替换
    elif is_variable(term1):
        # print(f"将{term1}替换为{term2}")
        return {term1: term2}  # 将变量term1替换为term2
    elif is_variable(term2):
        # print(f"将{term2}替换为{term1}")
        return {term2: term1}  # 将变量term2替换为term1
    else:
        return None  # 两个常量不相等，无法合一


def ResolutionFOL(KB):
    steps = []  # 存储归结步骤的列表
    clauses = [tuple_to_clause(clause) for clause in KB]  # 初始子句集
    clause_id_map = {i + 1: clauses[i] for i in range(len(clauses))}  # 子句编号映射
    next_id = len(clauses) + 1

    # 记录初始子句
    for cid, clause in clause_id_map.items():
        steps.append(f"{cid} {clause}")

    while True:
        found_resolvent = False
        # 创建一个副本，避免在遍历时修改原列表
        for i in range(len(clauses) - 1):
            for j in range(i + 1, len(clauses)):
                ci = clauses[i]
                cj = clauses[j]
                resolvent, swap_str = resolve(ci, cj)
                if resolvent is not None and resolvent not in clauses:
                    # 移除原有子句，添加新生成的子句
                    clauses.remove(ci)
                    clauses.remove(cj)
                    clauses.append(resolvent)
                    found_resolvent = True

                    # 更新子句编号映射
                    ci_id = list(clause_id_map.keys())[list(
                        clause_id_map.values()).index(ci)]
                    cj_id = list(clause_id_map.keys())[list(
                        clause_id_map.values()).index(cj)]
                    clause_id_map.pop(ci_id)
                    clause_id_map.pop(cj_id)
                    clause_id_map[next_id] = resolvent

                    # 构造归结步骤描述
                    resolvent_desc = f"R[{ci_id},{cj_id}]{swap_str} = {resolvent}"
                    steps.append(f"{next_id} {resolvent_desc}")
                    next_id += 1
                    # steps.append(f"剩下的{clauses}")

                    if not resolvent.literals:  # 如果产生了空子句
                        return steps
                    break  # 重新开始循环
            if found_resolvent:
                break  # 重新开始外层循环

        if not found_resolvent:  # 如果没有新的归结产生，终止循环
            break

    return steps
