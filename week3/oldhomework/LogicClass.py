# 文字
class Literal:
    def __init__(self, negated, predicate, terms):
        self.negated = negated  # 是否否定，布尔值
        self.predicate = predicate  # 谓词符号，字符串
        self.terms = terms  # 参数列表，列表形式

    def __repr__(self):
        neg = "~" if self.negated else ""
        return f"{neg}{self.predicate}({', '.join(self.terms)})"

    def __eq__(self, other):
        return (self.negated == other.negated and
                self.predicate == other.predicate and
                self.terms == other.terms)

    def __hash__(self):
        return hash((self.negated, self.predicate, tuple(self.terms)))
# 子句，包含多个文字
class Clause:
    def __init__(self, literals=None):
        self.literals = set(literals) if literals else set()

    def add_literal(self, literal):
        self.literals.add(literal)
        
    def add_literal(self, negated, predicate, terms):
        self.literals.add(Literal(negated, predicate, terms))

    def __repr__(self):
        if not self.literals:
            return "[]"
        return "{" + ", ".join(map(str, self.literals)) + "}"
