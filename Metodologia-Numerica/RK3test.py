import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # Importar a função tabulate

# Definição dos parâmetros
A = 10
W = 2 * np.pi  # Frequência angular (2*pi*f onde f é a frequência)
fi = 0  # Ângulo de fase

# Cálculo da posição do MHS
def f(t, y):
    return -W * A * np.sin(W * t + fi)

# RUNGE KUTTA 3ª ORDEM
def RK3(f, tk, yk, h):
    k1 = f(tk, yk)
    k2 = f(tk + h / 2, yk + k1 * h / 2)
    k3 = f(tk + h, yk - k1 * h + 2 * k2 * h)

    yk_1 = yk + h * (k1 + 4 * k2 + k3) / 6
    return yk_1

# Solução analítica
def harmonic_motion(amplitude, angular_frequency, phase_angle, time):
    return amplitude * np.cos(angular_frequency * time + phase_angle)

# Condições iniciais
t0 = 0
y0 = 10
h = 0.1  # Passo de tempo

# Iterar para calcular a posição em vários passos de tempo usando RK3
t_values_rk = [t0]
y_values_rk = [y0]
num_steps = 50  # Número de passos
for _ in range(num_steps):
    y_next = RK3(f, t_values_rk[-1], y_values_rk[-1], h)
    t_values_rk.append(t_values_rk[-1] + h)
    y_values_rk.append(y_next)

# Calcular a solução analítica
t_values_analytical = np.linspace(t0, num_steps * h, 1000)
y_values_analytical = harmonic_motion(A, W, fi, t_values_analytical)

# Criar uma tabela com os dados
table_data = []
for t, y_analytical, y_rk in zip(t_values_rk, y_values_analytical, y_values_rk):
    table_data.append([t, y_analytical, y_rk])

# Imprimir a tabela formatada no terminal
headers = ["Tempo", "Solução Analítica", "Aproximação de Runge-Kutta"]
print(tabulate(table_data, headers=headers))

# Plotar ambas as soluções
plt.figure(figsize=(8, 6))
plt.plot(t_values_analytical, y_values_analytical, label='Solução Analítica')
plt.plot(t_values_rk, y_values_rk, label='Aproximação de Runge-Kutta', linestyle='--')
plt.xlabel('Tempo')
plt.ylabel('Posição')
plt.title('Comparação entre Solução Analítica e de Runge-Kutta para o Movimento de Oscilador Harmônico')
plt.legend()
plt.grid(True)
plt.show()
