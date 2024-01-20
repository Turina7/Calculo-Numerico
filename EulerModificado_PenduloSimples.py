import numpy as np
import matplotlib.pyplot as plt

def solve_pendulum_euler_modified(theta0, omega0, g, L, dt, num_steps):
    # Inicialização de arrays para armazenar os resultados
    theta = np.zeros(num_steps)
    omega = np.zeros(num_steps)
    time = np.zeros(num_steps)

    # Condições iniciais
    theta[0] = theta0
    omega[0] = omega0

    # Método de Euler Modificado (Heun)
    for i in range(1, num_steps):
        # Predição
        theta_pred = theta[i-1] + omega[i-1] * dt
        omega_pred = omega[i-1] - (g / L) * np.sin(theta[i-1]) * dt

        # Correção
        theta[i] = theta[i-1] + 0.5 * (omega[i-1] + omega_pred) * dt
        omega[i] = omega[i-1] - 0.5 * ((g / L) * (np.sin(theta[i-1]) + np.sin(theta_pred))) * dt
        time[i] = time[i-1] + dt

    return time, theta

# Parâmetros do pêndulo
theta0 = np.radians(45)  # Ângulo inicial em graus
omega0 = 0.0  # Velocidade angular inicial
g = 9.8  # Aceleração devido à gravidade (m/s^2)
L = 1.0  # Comprimento da corda (metros)

# Parâmetros de simulação
dt = 0.01  # Passo de tempo
num_steps = 1000  # Número de passos

# Resolvendo a EDO usando o Método de Euler Modificado
time, theta = solve_pendulum_euler_modified(theta0, omega0, g, L, dt, num_steps)

# Imprimir valores em 5 momentos específicos do tempo entre 0 e 2 segundos
for t in np.linspace(0, 2, 5):
    idx = int(t / (time[-1] / num_steps))
    print("Tempo = {:.2f} s || Angulo(Graus) = {:.2f} || Aceleração angular(Rad/s^2) = {:.4f}".format(
        time[idx], np.degrees(theta[idx]), -((g / L) * np.sin(theta[idx]))))

# Plotando o resultado
plt.plot(time, np.degrees(theta))
plt.title('Pêndulo Simples - Método de Euler Modificado')
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (graus)')
plt.show()
