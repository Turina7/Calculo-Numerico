import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # Importar a função tabulate



# --------------------- PENDULO DUPLO ---------------------

############ CONDIÇÕES INICIAIS ###########
THETA1_0 = np.pi/2
THETA2_0 = np.pi/4

THETA1P_0 = 0
THETA2P_0 = 0

############## PARÂMETROS ################
M1 = 1
M2 = 1
L1 = 1
L2 = 1
G = 9.8

############# FUNÇÃO DERIVADA ###########
def f(t, y):
  theta1, theta2, theta1p, theta2p = y
  # Retorna a derivada de cada componente de y 
  # [theta1, that2, theta1p, theta2p] -> [theta1p, theta2p, theta1pp, theta2pp]
  return np.array([theta1p,
                   theta2p,
                   -( G*(2*M1 + M2)*np.sin(theta1) -M2*G*np.sin(theta1 - 2*theta2) -2*np.sin(theta1 - theta2) * M2*(theta2p**2*L2 + theta1p**2*L1 * np.cos(theta1-theta2))) / (L1 * (2*M1 + M2 -M2*np.cos(2*theta1 - 2*theta2))),
                   (2*np.sin(theta1 - theta2) * (theta1p**2*L1*(M1+M2)) + G*(M1+M2)*np.cos(theta1) + theta2p**2*L2 *M2*np.cos(theta1-theta2)) / (L2 * (2*M1 + M2 -M2*np.cos(2*theta1 - 2*theta2)))])


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


########### RUNGE KUTTA 3a ORDEM ###########
def RK3(f, tk, yk, h):
    k1 = h*f(tk, yk)
    k2 = h*f(tk + h/2, yk+ k1/2)
    k3 = h*f(tk + 3*h/4, yk + 3*k2/4)

    yk_1 = yk + (2*k1 + 3*k2 + 4*k3)/9
    return yk_1


####### APROXIMAÇÃO PENDULO DUPLO ########
t0 = 0
y0 = np.array([THETA1_0, THETA2_0, THETA1P_0, THETA2P_0])
h = 0.01
n = 10000

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


# Solucionar EDOs
# t = np.linspace(t0, t0 + n * h, n)
# y = rk3(f, t0, y0, h, n)

# # Extrair resultados
# theta1 = y[:, 0]
# theta2 = y[:, 1]

# theta1, theta2, theta1p, theta2p = RK3(f, 1, y0, h)

# print(f" theta1: {theta1}\n theta2: {theta2}\n theta1p: {theta1p}\n theta2p: {theta2p}")

# import matplotlib.pyplot as plt

# plt.plot(t, theta1, label="θ₁")
# plt.plot(t, theta2, label="θ₂")
# plt.xlabel("Tempo (t)")
# plt.ylabel("Ângulo (θ)")
# plt.legend()
# plt.show()

# def RK3p(f, tk, yk, h):
#     k1 = f(tk, yk)
#     k2 = f(tk + h/2, yk+ k1/2)
#     k3 = f(tk + 3*h/4, yk + 3*k2/4)

#     yk_1 = yk + h*(2*k1 + 3*k2 + 4*k3)/9
    # return yk_1

'''
## Calcular theta1p e theta2p para tk = 1
theta1, theta2, theta1p, theta2p = RK3(f, 1, [theta1, theta2, theta1p, theta2p], h)
theta1_1, theta2_1, theta1_p_1, theta2_p_1 = RK3()
'''