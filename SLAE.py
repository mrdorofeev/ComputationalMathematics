import numpy as np

eps = 10e-9


def norm(A):
    """
    Норма бесконечности матрицы A
    """
    return np.max(np.sum(np.absolute(A)))


def eq(a, b) -> bool:
    """
    Функция эквивалентности
    """
    if abs(a-b) < eps:
        return True
    return False


def find_det(LU, P, Q):
    """
    LU - LU разложение матрицы
    P - перестановка
    """
    det = np.float64(1)
    for i in range(LU.shape[0]):
        det *= LU[i][i]
    return det * P.even * Q.even


class Permutation(object):
    """
     Класс, описывающий перестановку строк в матрице
     
     P - массив перестановки
     even - чётность перстановки
         1  - чётна
         -1 - нечётна
    """
    def __init__(self, n):
        self.P = np.arange(n)
        self.even = 1
        self.__inverse = None
        
    def __getitem__(self, i):
        return self.P[i]
    
    def swap(self, i, j):
        """
        Поменять строки i и j местами
        """
        if i != j:
            self.P[i], self.P[j] = self.P[j], self.P[i]
            self.even *= -1
            self.__inverse = None
            
    def size(self):
        return self.P.size
    
    def __str__(self):
        return self.P.__str__()
    
    def inverse(self):
        """Обратная перестановка"""
        if self.__inverse is None:
            self.__inverse = np.arange(self.size())
            for i in range(self.__inverse.size):
                self.__inverse[self.P[i]] = i
        return self.__inverse


def LU_decomposition(A):
    """
    input:
        A - матрица
        
    output:
        (LU, P)
        LU - матрица разложения
        P - перестановка строк
        Q - перестановка столбцов
    """
    actions = 0
    
    size = A.shape # размер матрицы
    P = Permutation(size[0]) # перестановка строк
    Q = Permutation(size[1]) # перестановка столбцов
    
    LU = A.copy()
    
    for i in range(min(*size)):
        max_idx = np.argmax(LU[i:, i:])
        if eq(LU[i:, i:].flat[max_idx], 0):
            break
            
        row = i + max_idx // (size[1]-i)
        col = i + max_idx % (size[1]-i)
        P.swap(i, row)
        Q.swap(i, col)
        LU[[i, row]] = LU[[row, i]]
        LU[:, [i, col]] = LU[:, [col, i]]
    
        if i + 1 < min(*LU.shape):
            for r in range(i+1, LU.shape[0]):
                LU[r, i] = LU[r, i] / LU[i, i]
                actions += 1
                LU[r, i+1:] = LU[r, i+1:] - LU[i, i+1:] * LU[r, i]
                actions += 2 * (LU.shape[0] - i - 1)

    return LU, P, Q, actions


def SLAE_triangle(A, B, diagonal_is_ones = False):
    """
    Решение СЛАУ, у которых матрица А имеет вид верхней правой треугольной матрицы
    
    diagonal_is_ones = True, если матрица унитреугольная
    """
    actions = 0
    b = B.copy()
    n = A.shape[1]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i] / (A[i][i] if not diagonal_is_ones else 1)
        actions += 1
        if i > 0:
            # TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            b = b - b[i]*(A[:, i] / (A[i][i] if not diagonal_is_ones else 1))
            actions += 2 * b.size
    return x, actions


def SLAE_by_LU(LU, P, Q, b):
    """
    Решение СЛАУ для невырожденных матриц
    """
    b = b[P.P]

    actions = 0
    
    # если размерность столбца b не соотносится с числом строк
    if LU.shape[0] != b.size:
        b = b[:LU.shape[0]]
       
    if eq(find_det(LU, P, Q), 0):
        raise Exception("det(A) = 0")

    y, acs = SLAE_triangle(LU[::-1, ::-1], b[::-1], True) # Ly = Pb
    actions += acs
    y = y[::-1]

    # actions += acs

    x, acs = SLAE_triangle(LU, y) # Ux = y
    actions += acs
    X = np.zeros(Q.size())
    X[:x.size] = x
    return X[Q.inverse()], actions


def SLAE(A, b):
    """
    Решение СЛАУ для невырожденных матриц
    """
    LU, P, Q, actions = LU_decomposition(A)

    x, acs = SLAE_by_LU(LU, P, Q, b)
    return x, actions + acs