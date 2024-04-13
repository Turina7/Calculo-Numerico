import numpy as np
import math
import matplotlib.pyplot as plt
import cubic_splines as sp
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
angle1 = 90
angle2 = 60

THETA1_0 = np.radians(angle1)
THETA2_0 = np.radians(angle2)


THETA1P_0 = 0
THETA2P_0 = 100

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

def estimar_ordem_convergencia(n_estimativas, t_fixo, h_inicial, num_steps_inicial, t0, y0):
    ordens_convg = []
    
    for i in range(n_estimativas):
        h = h_inicial/(2**i)
        num_steps = num_steps_inicial*(2**i)
        y = y0
        t = t0

        # h
        for _ in range(num_steps):
            y = RK3(f, t, y, h)
            t += h
            if abs(t-t_fixo) < h*2:
                print("entrei")
                y_h = y
                break
            # print("saí")
        
        # h/2
        h = h/2
        num_steps = num_steps*2
        y = y0
        t = t0
        for _ in range(num_steps):
            y = RK3(f, t, y, h)
            t += h
            if abs(t-t_fixo) < h:
                y_h2 = y
                break
        
        # 2h
        h = h*4
        num_steps = int(num_steps/4)
        y = y0
        t = t0
        for _ in range(num_steps):
            y = RK3(f, t, y, h)
            t += h
            if abs(t-t_fixo) < h:
                y_2h = y
                break

        print("Valores para t = ", t_fixo)
        print("h = ", h)
        print("y_h = ", y_h)
        print("y_h2 = ", y_h2)
        print("y_2h = ", y_2h)

        # Calcular estimativa da ordem de convergencia
        ord_conv = np.log2(abs((y_2h - y_h)/(y_h - y_h2)))
        ordens_convg.append(ord_conv)
    
    print("Valores de convergência: ")
    for i in ordens_convg:
        print(f"y1 = {i[0]}, y2 = {i[1]}")
    
    ordem1 = 0
    ordem2 = 0
    for i in ordens_convg:
        ordem1 += i[0]
        ordem2 += i[1]
    return ordem1/n_estimativas, ordem2/n_estimativas


            




if MODELO == "pendulo":
    ####### APROXIMAÇÃO PENDULO DUPLO ########
    t0 = 0
    y0 = np.array([THETA1_0, THETA2_0, THETA1P_0, THETA2P_0])
    h = 0.05
    n = 1000

    # Iterar para calcular a posição em vários passos de tempo usando RK3
    t_values_rk = [t0]
    y1_values_rk = [y0[0]] # Valores de theta1
    y2_values_rk = [y0[1]] # Valores de theta2
    num_steps = n
    y1_values_rk_2t = [y0[0]] # Valores de theta1
    y2_values_rk_2t = [y0[1]] # Valores de theta2
    y1_values_rk_ts2 = [y0[0]] # Valores de theta1
    y2_values_rk_ts2 = [y0[1]] # Valores de theta2
    t_values_rk_2t = [t0]
    t_values_rk_ts2 = [t0]

    y = y0
    t = t0

    # Calcular ordem do erro de convergencia
    n_estimativas = 10
    t_fixo = 2
    h_inicial = 0.005
    num_steps_inicial = 3000
    ordem = estimar_ordem_convergencia(n_estimativas, t_fixo, h_inicial, num_steps_inicial, t0, y0)
    print("Ordem de convergência: ", ordem)
  
    # Criar uma tabela com os dados
    # x_table_data = []
    # for t, theta1_rk, theta2_rk in zip(t_values_rk, y1_values_rk, y2_values_rk):
    #     x_table_data.append([t, theta1_rk, theta2_rk])

    # # Imprimir a tabela formatada no terminal
    # headers = ["Tempo", " Theta 1 - RK3", "Theta 2 - RK3"]
    # print(tabulate(x_table_data, headers=headers))
        
    # y1_splines = sp.get_splines(t_values_rk[:1000:10], y1_values_rk[:1000:10])
    # y2_splines = sp.get_splines(t_values_rk[:1000:10], y2_values_rk[:1000:10])

    # y1_splines_dif = sp.get_splines(t_values_rk, y1_values_rk_dif)
    # y2_splines_dif = sp.get_splines(t_values_rk, y2_values_rk_dif)

    # y1_values_spline, t_splines = sp.spline_aproximation(t_values_rk, y1_splines)
    # y2_values_spline, t_splines = sp.spline_aproximation(t_values_rk, y2_splines)

    # y1_values_spline_dif, t_splines = sp.spline_aproximation(t_values_rk, y1_splines_dif)
    # y2_values_spline_dif, t_splines = sp.spline_aproximation(t_values_rk, y2_splines_dif)

    # # # Plotar erro com e sem variacao das condicoes iniciais
    # plt.figure(figsize=(8, 6))
    # # plt.plot(t_splines, y1_values_spline, label= f'Ordem do erro do  ângulo 1')
    # plt.plot(t_values_rk, y1_values_rk, label= f'Ordem do erro do ângulo 2', linestyle='--')
    # plt.xlabel('Tempo (s)')
    # plt.ylabel('Ordem do erro de convergência')
    # plt.title('')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


            
    # # # Plotar erro com e sem variacao das condicoes iniciais
    # plt.figure(figsize=(8, 6))
    # plt.plot(n_x, n_1, label= f'Ordem do erro do  ângulo 1')
    # plt.plot(n_x, n_2, label= f'Ordem do erro do ângulo 2', linestyle='--')
    # plt.xlabel('Tempo (s)')
    # plt.ylabel('Ordem do erro de convergência')
    # plt.title('')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
            
    # Plotar erro com e sem variacao das condicoes iniciais
    # plt.figure(figsize=(8, 6))
    # plt.plot(t_values_rk[:3000:2], y1_values_rk[:3000:2], label= f'theta1 = {angle1}° e theta2 = {angle2}°')
    # plt.plot(t_values_rk_2t[:1500], y1_values_rk_2t[:1500], label=f'theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°')
    # plt.plot(t_values_rk_ts2[:6000:4], y1_values_rk_ts2[:6000:4], label=f'theta1 = {round(angle1*dif,2)}° e a = {round(angle2*dif,2)}°')
    # plt.xlabel('Tempo (s)')
    # plt.ylabel('trem (rad)')
    # plt.title('Ângulos da massa 1 - Extremo de baixa energia')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
            
            
    

    # Plotar theta1 com e sem variacao das condicoes iniciais
    # plt.figure(figsize=(8, 6))
    # plt.plot(t_values_rk, y1_values_rk, label= f'theta1 = {angle1}° e theta2 = {angle2}°')
    # # plt.plot(t_values_rk, y1_values_rk_dif, label=f'theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°', linestyle='--')
    # plt.xlabel('Tempo (s)')
    # plt.ylabel('Ângulo (rad)')
    # plt.title('Ângulos da massa 1 - Extremo de baixa energia')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plotar theta2 com e sem variacao das condicoes iniciais
    # plt.figure(figsize=(8, 6))
    # plt.plot(t_values_rk, y2_values_rk, label= f'theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°')
    # plt.plot(t_values_rk, y2_values_rk_dif, label=f'theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°', linestyle='--')
    # plt.xlabel('Tempo (s)')
    # plt.ylabel('Ângulo (rad)')
    # plt.title('Ângulos da massa 2 - Extremo de baixa energia')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # ### plotagem no plano x, y do caso 1
    # def update_line_position(x0, y0, length, angle_radians):
    #     # Calculate new end coordinates
    #     x = x0 + length * math.sin(angle_radians)
    #     y = y0 - length * math.cos(angle_radians)  # Negative sign for Y due to inverted Y-axis in matplotlib
    #     return x, y

    # x1_values = []
    # y1_values = []
    # x2_values = []
    # y2_values = []

    # # Calculate x and y values for each y1 value using the update_line_position function
    # for i in y1_values_rk:
    #     x, y = update_line_position(0, 0, L1, i)
    #     x1_values.append(x)
    #     y1_values.append(y)

    # count = 0
    # for j in y2_values_rk:
    #     x, y = update_line_position(x1_values[count], y1_values[count], L2, j)
    #     x2_values.append(x)
    #     y2_values.append(y)
    #     count += 1

    # plt.figure(figsize=(8, 6))
    # plt.plot(x1_values, y1_values, label='Massa 1')
    # plt.plot(x2_values, y2_values, label='Massa 2', linestyle='--')
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.title(f'Movimento do pendulo duplo (theta1 = {angle1}° e theta2 = {angle2}°)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    #No caso dif
    # x1_values_dif = []
    # y1_values_dif = []
    # x2_values_dif = []
    # y2_values_dif = []

    # Calculate x and y values for each y1 value using the update_line_position function
    # for i in y1_values_rk_dif:
    #     x, y = update_line_position(0, 0, L1, i)
    #     x1_values_dif.append(x)
    #     y1_values_dif.append(y)

    # count = 0
    # for j in y2_values_rk_dif:
    #     x, y = update_line_position(x1_values_dif[count], y1_values_dif[count], L2, j)
    #     x2_values_dif.append(x)
    #     y2_values_dif.append(y)
    #     count += 1

    # plt.figure(figsize=(8, 6))
    # plt.plot(x1_values_dif, y1_values_dif, label='Massa 1')
    # plt.plot(x2_values_dif, y2_values_dif, label='Massa 2', linestyle='--')
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.title(f'Movimento do pendulo duplo (theta1 = {round(angle1*dif,2)}° e theta2 = {round(angle2*dif,2)}°)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
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
    headers = ["Tempo", "X\' da Solugrama para ∆t =ção Analítica", "X\' da Aproximação de Runge-Kutta"]
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

