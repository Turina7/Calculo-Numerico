import numpy as np

############## PARÂMETROS ################
M1 = 1
M2 = 1
L1 = 1
L2 = 1
G = 9.8

############ CONDIÇÕES INCIIAIS ###########
THETA1_0 = np.pi/2
THETA2_0 = np.pi/4

THETA1P_0 = 0
THETA2P_0 = 0

############# THETA1'' E THETA2'' ###########
def f(t, y):
  theta1, theta2, theta1p, theta2p = y
  # Retorna a derivada de cada componente de y -> [theta1p, theta2p, theta1pp, theta2pp]
  return np.array([theta1p,
                   theta2p,
                   -( G*(2*M1 + M2)*np.sin(theta1) -M2*G*np.sin(theta1 - 2*theta2) -2*np.sin(theta1 - theta2) * M2*(theta2p**2*L2 + theta1p**2*L1 * np.cos(theta1-theta2))) / (L1 * (2*M1 + M2 -M2*np.cos(2*theta1 - 2*theta2))),
                   (2*np.sin(theta1 - theta2) * (theta1p**2*L1*(M1+M2)) + G*(M1+M2)*np.cos(theta1) + theta2p**2*L2 *M2*np.cos(theta1-theta2)) / (L2 * (2*M1 + M2 -M2*np.cos(2*theta1 - 2*theta2)))])
   
########### RUNGE KUTTA 3a ORDEM ###########
def RK3(f, tk, yk, h):
    k1 = f(tk, yk)
    k2 = f(tk + h/2, yk+ k1/2)
    k3 = f(tk + 3*h/4, yk + 3*k2/4)

    yk_1 = yk + h*(2*k1 + 3*k2 + 4*k3)/9
    return yk_1


# Definir parâmetros
t0 = 0
y0 = np.array([THETA1_0, THETA2_0, THETA1P_0, THETA2P_0])
h = 0.01
n = 1000

# Solucionar EDOs
# t = np.linspace(t0, t0 + n * h, n)
# y = rk3(f, t0, y0, h, n)

# # Extrair resultados
# theta1 = y[:, 0]
# theta2 = y[:, 1]

theta1, theta2, theta1p, theta2p = RK3(f, 1, y0, h)

print(f" theta1: {theta1}\n theta2: {theta2}\n theta1p: {theta1p}\n theta2p: {theta2p}")

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