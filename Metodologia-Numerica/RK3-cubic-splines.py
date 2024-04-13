'''
Neste código é feita a aproximação dos ângulos
do pêndulo duplo com o método RK3, que é comparada
com a interpolação por splines cubicas de pontos 
selecionados dessa aproximação.
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from tabulate import tabulate  
import cubic_splines as sp
# Altere as condições iniciais e parametros como desejar
# Busque no código por "Condições iniciais"


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
angle1 = 90
angle2 = 60

THETA1_0 = np.radians(angle1)
THETA2_0 = np.radians(angle2)


THETA1P_0 = 0
THETA2P_0 = 0

############## PARÂMETROS ################
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
 
        
    # Aproximação para os 1000 primeiros pontos, utilizando 20 pontos para interpolação
    n_points = 50

    y1_splines = sp.get_splines(t_values_rk[:1000:n_points], y1_values_rk[:1000:n_points])
    y1_values_splines, t_splines = sp.spline_aproximation(t_values_rk[:1000:n_points], y1_splines, n_points)
    # Plotar theta1 com e sem variacao das condicoes iniciais
    plt.figure(figsize=(8, 6))
    plt.plot(t_values_rk[:1000], y1_values_rk[:1000], label= f'RK3 sem interpolação (1000 pontos)')
    plt.plot(t_splines, y1_values_splines, label=f'RK3 com interpolaçao por splines de 20 pontos', linestyle='--')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (rad)')
    plt.title('Interpolação do ângulo theta1 por splines cubicas utilizando 20 pontos')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Aproximação para os 1000 primeiros pontos, utilizando 100 pontos para interpolação
    n_points = 10

    y1_splines = sp.get_splines(t_values_rk[:1000:n_points], y1_values_rk[:1000:n_points])
    y1_values_splines, t_splines = sp.spline_aproximation(t_values_rk[:1000:n_points], y1_splines, n_points)
    # Plotar theta1 com e sem variacao das condicoes iniciais
    plt.figure(figsize=(8, 6))
    plt.plot(t_values_rk[:1000], y1_values_rk[:1000], label= f'RK3 sem interpolação (1000 pontos)')
    plt.plot(t_splines, y1_values_splines, label=f'RK3 com interpolaçao por splines de 100 pontos', linestyle='--')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (rad)')
    plt.title('Interpolação do ângulo theta1 por splines cubicas utilizando 100 pontos')
    plt.legend()
    plt.grid(True)
    plt.show()
