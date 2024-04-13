import numpy as np
import math
import matplotlib.pyplot as plt
from tabulate import tabulate  

# Definir qual dos modelos irá simular
# "pendulo", "mhs" ou null
MODELO = "mhs"

########### RUNGE KUTTA 3a ORDEM ###########
def RK3(f, tk, yk, h):
    k1 = h*f(tk, yk)
    k2 = h*f(tk + h/2, yk+ k1/2)
    k3 = h*f(tk + 3*h/4, yk + 3*k2/4)

    yk_1 = yk + (2*k1 + 3*k2 + 4*k3)/9
    return yk_1

# --------------------- PENDULO DUPLO ---------------------

############ CONDIÇÕES INICIAIS ###########
angle1 = 90
angle2 = 60

THETA1_0 = np.radians(angle1)
THETA2_0 = np.radians(angle2)


THETA1P_0 = 0
THETA2P_0 = 0

############## PARÂMETROS ###############
M1 = 0.2
M2 = 0.2
L1 = 0.84
L2 = 0.84
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
    h = 0.01
    n = 3000

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

    ####### APROXIMAÇÃO PENDULO DUPLO DIF ########
    # Pequena mudança nas condicoes iniciais
    t0 = 0
    dif = 1.001 # Multiplicador (+0.1%)
    THETA1_0_dif = THETA1_0 * dif
    THETA2_0_dif = THETA2_0 * dif

    # Velocidade inciial permance nula
    THETA1P_0_dif = THETA1P_0
    THETA2P_0_dif = THETA2P_0

    y0 = np.array([THETA1_0_dif, THETA2_0_dif, THETA1P_0_dif, THETA2P_0_dif])
    

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

    # Plotar theta1 com e sem variacao das condicoes iniciais
    plt.figure(figsize=(8, 6))
    plt.plot(t_values_rk, y1_values_rk, label= f'theta1 = {angle1}° e theta2 = {angle2}°')
    plt.plot(t_values_rk, y1_values_rk_dif, label=f'theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°', linestyle='--')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (rad)')
    plt.title('Ângulos da massa 1 - Análise de pequenos ângulos')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotar theta1 com e sem variacao das condicoes iniciais
    plt.figure(figsize=(8, 6))
    plt.plot(t_values_rk, y2_values_rk, label= f'theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°')
    plt.plot(t_values_rk, y2_values_rk_dif, label=f'theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°', linestyle='--')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (rad)')
    plt.title('Ângulos da massa 2 - Análise de pequenos ângulos')
    plt.legend()
    plt.grid(True)
    plt.show()

    ### plotagem no plano x, y do caso 1
    def update_line_position(x0, y0, length, angle_radians):
        # Calculate new end coordinates
        x = x0 + length * math.sin(angle_radians)
        y = y0 - length * math.cos(angle_radians)  # Negative sign for Y due to inverted Y-axis in matplotlib
        return x, y

    x1_values = []
    y1_values = []
    x2_values = []
    y2_values = []

    # Calculate x and y values for each y1 value using the update_line_position function
    for i in y1_values_rk:
        x, y = update_line_position(0, 0, L1, i)
        x1_values.append(x)
        y1_values.append(y)

    count = 0
    for j in y2_values_rk:
        x, y = update_line_position(x1_values[count], y1_values[count], L2, j)
        x2_values.append(x)
        y2_values.append(y)
        count += 1

    plt.figure(figsize=(8, 6))
    plt.plot(x1_values, y1_values, label='Massa 1')
    plt.plot(x2_values, y2_values, label='Massa 2', linestyle='--')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Movimento do pendulo duplo (theta1 = {angle1}° e theta2 = {angle2}°)')
    plt.legend()
    plt.grid(True)
    plt.show()

    #No caso dif
    x1_values_dif = []
    y1_values_dif = []
    x2_values_dif = []
    y2_values_dif = []

    # Calculate x and y values for each y1 value using the update_line_position function
    for i in y1_values_rk_dif:
        x, y = update_line_position(0, 0, L1, i)
        x1_values_dif.append(x)
        y1_values_dif.append(y)

    count = 0
    for j in y2_values_rk_dif:
        x, y = update_line_position(x1_values_dif[count], y1_values_dif[count], L2, j)
        x2_values_dif.append(x)
        y2_values_dif.append(y)
        count += 1

    plt.figure(figsize=(8, 6))
    plt.plot(x1_values_dif, y1_values_dif, label='Massa 1')
    plt.plot(x2_values_dif, y2_values_dif, label='Massa 2', linestyle='--')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Movimento do pendulo duplo (theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°)')
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
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição (m)')
    plt.title('Comparação entre Solução Analítica e de Runge-Kutta para posição do MHS')
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
