from LogicClass import Literal, Clause
from Fun import resolve,ResolutionFOL,tuple_to_clause
# clause1 = Clause()
# clause1.add_literal(False, 'P', ['x','y','z'])
# clause1.add_literal(False, 'Q', ['x','y'])
# clause1.add_literal(False, 'E', ['x','y'])
# clause2 = Clause()
# clause2.add_literal(True, 'Q', ['a','b'])
# clause1.add_literal(False, 'B', ['x','y'])
# # 使用提供的例子进行测试
# resolved_clause = resolve(clause1, clause2)
# if resolved_clause is not None:
#     print(f"归结结果: {resolved_clause}")
# else:
#     print("这两个子句无法归结。")
# c1 = tuple_to_clause(("GradStudent(sue)",))
# c2 = tuple_to_clause(("~GradStudent(x)","Student(x)","Student(x)"))  
# resolved_clause = resolve(c1, c2)
# if resolved_clause is not None:
#     print(f"归结结果: {resolved_clause}")
# else:
#     print("这两个子句无法归结。")
# KB = {("GradStudent(sue)",),("~GradStudent(x)","Student(x)"),("~Student(x)","HardWorker(x)"),("~HardWorker(sue)",)}
KB = {("FirstGrade",), ("~FirstGrade","Child"), ("~Child",)}
ResolutionFOL_result = ResolutionFOL(KB)
print("\n".join(ResolutionFOL_result))
# KB = {("A(tony)",),("A(mike)",),("A(john)",),("L(tony,rain)",),("L(tony,snow)",),("~A(x)","S(x)","C(x)"),("~C(y)","~L(y,rain)"),("L(z,snow)","~S(z)"),("~L(tony,u)","~L(mike,u)"),("L(tony,v)","L(mike,v)"),("~A(w)","~C(w)","S(w)")}

# ResolutionFOL_result = ResolutionFOL(KB)
# print("\n".join(ResolutionFOL_result))
# KB = {("On(tony,mike)",),("On(mike,john)",),("Green(tony)",),("~Green(john)",),("~On(xx,yy)","~Green(xx)","Green(yy)")}

# ResolutionFOL_result = ResolutionFOL(KB)
# print("\n".join(ResolutionFOL_result))
