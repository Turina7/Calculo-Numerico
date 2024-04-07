import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  

# Definir qual dos modelos irá simular
# "pendulo", "mhs" ou null
MODELO = "pendulo"

########### RUNGE KUTTA 3a ORDEM ###########
def RK3(f, tk, yk, h):
    k1 = h*f(tk, yk)
    k2 = h*f(tk + h/2, yk+ k1/2)
    k3 = h*f(tk + 3*h/4, yk + 3*k2/4)

    yk_1 = yk + (2*k1 + 3*k2 + 4*k3)/9
    return yk_1

# --------------------- PENDULO DUPLO ---------------------

############ CONDIÇÕES INICIAIS ###########
THETA1_0 = np.radians(180)
THETA2_0 = np.radians(60)


THETA1P_0 = 0
THETA2P_0 = 0

############## PARÂMETROS ################
M1 = 0.2
M2 = 0.2
L1 = 1
L2 = 1
G = 9.8

############# FUNÇÃO DERIVADA ###########
def f(t, y):
#   print(y)
  theta1, theta2, theta1p, theta2p = y
  # Retorna a derivada de cada componente de y 
  # [theta1, that2, theta1p, theta2p] -> [theta1p, theta2p, theta1pp, theta2pp]
  return np.array([theta1p,
                   theta2p,
                   ( -G*(2*M1 + M2)*np.sin(theta1) -M2*G*np.sin(theta1 - 2*theta2) -2*np.sin(theta1 - theta2) * M2*(theta2p**2*L2 + theta1p**2*L1 * np.cos(theta1-theta2)))
                     / (L1 * (2*M1 + M2 -M2*np.cos(2*theta1 - 2*theta2))),
                   (2*np.sin(theta1 - theta2) * (theta1p**2*L1*(M1+M2) + G*(M1+M2)*np.cos(theta1) + theta2p**2*L2 *M2*np.cos(theta1-theta2)))
                     / (L2 * (2*M1 + M2 -M2*np.cos(2*theta1 - 2*theta2)))])

if MODELO == "pendulo":
    ####### APROXIMAÇÃO PENDULO DUPLO ########
    t0 = 0
    y0 = np.array([THETA1_0, THETA2_0, THETA1P_0, THETA2P_0])
    h = 0.1
    n = 200

    # Iterar para calcular a posição em vários passos de tempo usando RK3
    t_values_rk = [t0]
    y1_values_rk = [y0[0]] # Valores de theta1
    y2_values_rk = [y0[1]] # Valores de theta2
    num_steps = n

    y = y0
    t = t0

    ##### LOOP #####
    for _ in range(num_steps):
        y = RK3(f, t, y, h)
        t += h
        t_values_rk.append(t)
        y1_values_rk.append(y[0])
        y2_values_rk.append(y[1])

    ####### APROXIMAÇÃO PENDULO DUPLO ########
    t0 = 0
    y0 = np.array([np.radians(180.018), np.radians(60.006), THETA1P_0, THETA2P_0])
    h = 0.1
    

    # Iterar para calcular a posição em vários passos de tempo usando RK3
    t_values_rk = [t0]
    y1_values_rk_dif = [y0[0]] # Valores de theta1
    y2_values_rk_dif = [y0[1]] # Valores de theta2
    num_steps = n

    y = y0
    t = t0

    ##### LOOP #####
    for _ in range(num_steps):
        y = RK3(f, t, y, h)
        t += h
        t_values_rk.append(t)
        y1_values_rk_dif.append(y[0])
        y2_values_rk_dif.append(y[1])

    

    # Criar uma tabela com os dados
    # x_table_data = []
    # for t, theta1_rk, theta2_rk in zip(t_values_rk, y1_values_rk, y2_values_rk):
    #     x_table_data.append([t, theta1_rk, theta2_rk])

    # # Imprimir a tabela formatada no terminal
    # headers = ["Tempo", " Theta 1 - RK3", "Theta 2 - RK3"]
    # print(tabulate(x_table_data, headers=headers))

    # Plotar theta1 e theta2 em função do temp
    plt.figure(figsize=(8, 6))
    plt.plot(t_values_rk, y1_values_rk, label='Aproximação de Runge-Kutta')
    plt.plot(t_values_rk, y1_values_rk_dif, label='Aproximação de Runge-Kutta dif', linestyle='--')
    plt.xlabel('Tempo')
    plt.ylabel('Ângulo')
    plt.title('Ângulos theta1 e theta2 - Aproximação de Runge-Kutta para o Movimento de Pendulo Duplo')
    plt.legend()
    plt.grid(True)
    plt.show()



    
# ------------------- OSCILADOR HARMÔNICO SIMPLES -------------------

############ CONDIÇÕES INICIAIS MHS ###########
X_0  = 10
XP_0 = 0

############# PARÂMETROS MHS ##############
M = 1
K = 1
W = np.sqrt(K/M)
PHI = np.arctan2(-XP_0,(W * X_0))
A = np.sqrt(X_0**2 + XP_0**2)

############# FUNÇÃO DERIVADA MHS ################
def f_mhs(t, y):
    # Retorna a derivada de cada componente de y 
    # [[x,xp] -> [xp, x2p]
    x, xp = y
    return np.array([xp, -(K/M) * x])

############## SOLUÇÃO ANALÍTICA ############
def x_mhs(A, W, phi, t):
    return A * np.cos(W * t + phi)

def xp_mhs(A, W, phi, t):
    return -A * W * np.sin(W * t + phi)




if MODELO == "mhs":

    ####### APROXIMAÇÃO MHS ########
    t0_mhs = 0
    y0_mhs = np.array([X_0, XP_0])
    h_mhs = 0.01
    n_mhs = 1000

    # Iterar para calcular a posição em vários passos de tempo usando RK3
    t_values_rk = [t0_mhs]
    y_values_rk = [y0_mhs[0]]
    yp_values_rk = [y0_mhs[1]]
    num_steps = n_mhs

    y_mhs = y0_mhs
    t_mhs = t0_mhs

    ##### LOOP PARA MHS #####
    for _ in range(num_steps):
        y_mhs = RK3(f_mhs, t_mhs, y_mhs, h_mhs)
        t_mhs += h_mhs
        t_values_rk.append(t_mhs)
        y_values_rk.append(y_mhs[0])
        yp_values_rk.append(y_mhs[1])


    # Calcular a solução analítica
    t_values_analytical = np.linspace(t0_mhs, num_steps * h_mhs, 1000)
    y_values_analytical = x_mhs(A, W, PHI, t_values_analytical)
    yp_values_analytical = xp_mhs(A, W, PHI, t_values_analytical)

    # Criar uma tabela com os dados
    x_table_data = []
    for t, y_analytical, y_rk in zip(t_values_rk, y_values_analytical, y_values_rk):
        x_table_data.append([t, y_analytical, y_rk])

    # Imprimir a tabela formatada no terminal
    headers = ["Tempo", " X da Solução Analítica", "X da Aproximação de Runge-Kutta"]
    print(tabulate(x_table_data, headers=headers))

    # Tabela para xp
    xp_table_data = []
    for t, yp_analytical, yp_rk in zip(t_values_rk, yp_values_analytical, yp_values_rk):
        xp_table_data.append([t, yp_analytical, yp_rk])

    # Imprimir a tabela formatada no terminal
    headers = ["Tempo", "X\' da Solução Analítica", "X\' da Aproximação de Runge-Kutta"]
    print(tabulate(xp_table_data, headers=headers))

    # Plotar ambas as soluções
    plt.figure(figsize=(8, 6))
    plt.plot(t_values_analytical, y_values_analytical, label='Solução Analítica')
    plt.plot(t_values_rk, y_values_rk, label='Aproximação de Runge-Kutta', linestyle='--')
    plt.xlabel('Tempo')
    plt.ylabel('Posição')
    plt.title('Posição - Comparação entre Solução Analítica e de Runge-Kutta para o Movimento de Oscilador Harmônico')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotar ambas as soluções
    plt.figure(figsize=(8, 6))
    plt.plot(t_values_analytical, yp_values_analytical, label='Solução Analítica')
    plt.plot(t_values_rk, yp_values_rk, label='Aproximação de Runge-Kutta', linestyle='--')
    plt.xlabel('Tempo')
    plt.ylabel('Velocidade')
    plt.title('Velocidade - Comparação entre Solução Analítica e de Runge-Kutta para o Movimento de Oscilador Harmônico')
    plt.legend()
    plt.grid(True)
    plt.show()

