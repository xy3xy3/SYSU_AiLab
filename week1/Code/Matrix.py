def matrix_add(A, B):
    n = len(A)
    #初始化矩阵
    C = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C

def MatrixMul(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[1, 2], [3, 4]]
    print(matrix_add(A, B))
    print(MatrixMul(A, B))