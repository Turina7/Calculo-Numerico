import numpy as np
from gaussian_elimination import gauss_elimination
import matplotlib.pyplot as plt

# Habilita a validação do código de splines
teste_splines = False

if teste_splines:

    h = 0.1
    t0 = 0
    n_pontos = 100
    t = np.arange(t0, n_pontos * h, h)

    def f3grau(t, a, b, c, d):
        return a*t**3 + b*t**2 + c*t + d

    y = f3grau(t, 1, 2, 3, 4)


# Example
# x = np.array([1,3,5,8])
# y = np.array([2,3,9,10])

# Solution
# [[  0.1995614   -0.59868421   0.3004386    2.09868421]
#  [ -0.37280702   4.55263158 -15.15350877  17.55263158]
#  [  0.11549708  -2.77192982  21.46929825 -43.48538012]]



########### GET EQUATIONS #############
def get_equations(x, y):
    # Number of splines
    n = len(x) - 1

    # Equations
    equations = np.zeros((4*n,4*n + 1),dtype=float)
    # equations = [[equation0], [equation1], ..., [euqationN-1]]
    # equationN = [a0, b0, c0, d0, a1, b1, c1, d1, ..., an-1, bn-1, cn-1, dn-1, r]
    # r is the constant


    # Pn(xn) = yn
    for i in range(n):
        # Pn(xn) = yn
        # equation = [x[i]**3, x[i]**2, x[i], y[i]]
        equations[2*i][i*4]     = x[i]**3
        equations[2*i][i*4 + 1] = x[i]**2
        equations[2*i][i*4 + 2] = x[i]
        equations[2*i][i*4 + 3] = 1
        equations[2*i][4*n] = y[i]

        # Pn(xn+1) = yn+1
        equations[2*i+1][i*4]     = x[i+1]**3
        equations[2*i+1][i*4 + 1] = x[i+1]**2
        equations[2*i+1][i*4 + 2] = x[i+1]
        equations[2*i+1][i*4 + 3] = 1
        equations[2*i+1][4*n] = y[i+1]

    for i in range(n-1):
        # P'n(xn+1) = P'n+1(xn+1)
        equations[2*n+i][i*4]     = 3 * x[i+1]**2
        equations[2*n+i][i*4 + 1] = 2 * x[i+1]
        equations[2*n+i][i*4 + 2] = 1
        equations[2*n+i][i*4 + 3] = 0

        equations[2*n+i][i*4 + 4] = -3 * x[i+1]**2
        equations[2*n+i][i*4 + 5] = -2 * x[i+1]
        equations[2*n+i][i*4 + 6] = -1
        equations[2*n+i][i*4 + 7] = 0

        # P"n(xn+1) = P"n+1(xn+1)
        equations[3*n+i-1][i*4]     = 6 * x[i+1]
        equations[3*n+i-1][i*4 + 1] = 2 
        equations[3*n+i-1][i*4 + 2] = 0
        equations[3*n+i-1][i*4 + 3] = 0

        equations[3*n+i-1][i*4 + 4] = -6 * x[i+1]
        equations[3*n+i-1][i*4 + 5] = -2 
        equations[3*n+i-1][i*4 + 6] = 0
        equations[3*n+i-1][i*4 + 7] = 0

    ## Natural splines
    # P"0(x0) = 0
    equations[4*n-2][0] = 6 * x[0]
    equations[4*n-2][1] = 2
        
    # P"n-1(xn) = 0
    equations[4*n-1][4*n-4] = 6 * x[n]
    equations[4*n-1][4*n-3] = 2

    return equations

def get_splines(x,y):
    # N splines 
    n = len(x) - 1
    
    equations = get_equations(x,y)
    variable_matrix = equations[:, :4*n]
    constant_matrix = equations[:, 4*n].reshape(4*n,1)

    # print(variable_matrix)
    # print(constant_matrix)

    # variable_matrix = np.array([[1,1,3],[0,1,3],[-1,3,0]])
    # constant_matrix = np.array([[1],[3],[5]])

    # Import gaussian elimination and solve
    solution = gauss_elimination(variable_matrix, constant_matrix)

    # Splines
    # a0*x³ + b0*x² + c0*x + d0 -> [a0, b0, c0, d0]
    splines = solution.reshape(n, 4)

    return splines

# print(get_splines(t,y))



def spline_aproximation(x, splines, n=10):
    n_splines = len(splines)
    t = np.arange(x[0], x[-1], (x[-1] - x[0])/(n*n_splines))
    y = np.zeros(n_splines*n + 1)
    for i in range(n_splines):
        a = splines[i][0]
        b = splines[i][1]
        c = splines[i][2]
        d = splines[i][3]

        for j in range(n):
            index = i*n + j
            y[index] = a*t[index]**3 + b*t[index]**2 + c*t[index] + d

    t = np.append(t, x[-1])
    y[-1] = a*x[-1]**3 + b*x[-1]**2 + c*x[-1] + d
    return y, t

if teste_splines:
    y_aproximado, t_splines = spline_aproximation(t, get_splines(t,y))

    # Visualizar solução analítica e aproximada
    plt.figure(figsize=(8, 6))
    plt.plot(t, y, label='Solução Analítica')
    plt.plot(t_splines, y_aproximado, label='Aproximação por Splines Cúbicas', linestyle='--')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title('Comparação entre Solução Analítica e aproximação por Splines Cúbicas de função de 3° grau')
    plt.legend()
    plt.grid(True)
    plt.show()


