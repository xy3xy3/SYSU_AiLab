import re

def MGU(f1, f2):
    # 解析两个公式的参数
    p1 = get_param(f1)
    p2 = get_param(f2)
    
    # 初始化结果字典和待处理的参数对
    res = {}
    queue = list(zip(p1, p2))
    
    while queue:
        # 从队列去除两个公式集合
        e1, e2 = queue.pop(0)
        if e1 == e2:
            continue  # 如果两个参数相同，则不需要合一
        elif is_variable(e1):
            if e1 in e2:
                return {}  # 如果e1作为变量出现在e2中，则合一失败
            else:
                res[e1] = e2
                queue = [(substitute(e1, e2, x), y) for x, y in queue]
        elif is_variable(e2):
            if e2 in e1:
                return {}  # 如果e2作为变量出现在e1中，则合一失败
            else:
                res[e2] = e1
                queue = [(x, substitute(e2, e1, y)) for x, y in queue]
        else:
            # 复合项f(x)之类的的处理
            match1 = re.match(r'(\w+)\((.*)\)', e1)
            match2 = re.match(r'(\w+)\((.*)\)', e2)
            if match1 and match2 and match1.group(1) == match2.group(1):
                # 剩下的项转字符串加入队列
                subterms1 = match1.group(2).split(',')
                subterms2 = match2.group(2).split(',')
                queue.extend(zip(subterms1, subterms2))
            else:
                return {}  # 如果两个复合项的谓词不同，则合一失败
    return res

def get_param(f):
    # 获取参数，返回list
    match = re.match(r'(\w+)\((.*)\)', f)
    if match:
        terms = match.group(2).split(',')
        return terms
    return []

def is_variable(term):
    # 简单的判断是否为变量的函数
    return re.match(r'^[u-z]+$', term) is not None

def substitute(var, value, term):
    # 在term中替换所有出现的var为value
    return term.replace(var, value)

# 测试
print(MGU('P(xx,a)', 'P(b,yy)'))
