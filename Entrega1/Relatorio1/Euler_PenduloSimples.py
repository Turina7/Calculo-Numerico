import numpy as np
import matplotlib.pyplot as plt

def solve_pendulum_euler(theta0, omega0, g, L, dt, num_steps):
    # Inicialização de arrays para armazenar os resultados
    theta = np.zeros(num_steps)
    omega = np.zeros(num_steps)
    time = np.zeros(num_steps)

    # Condições iniciais
    theta[0] = theta0
    omega[0] = omega0

    # Método de Euler
    for i in range(1, num_steps):
        # Atualização das variáveis
        theta[i] = theta[i-1] + omega[i-1] * dt
        omega[i] = omega[i-1] - (g / L) * np.sin(theta[i-1]) * dt
        time[i] = time[i-1] + dt

    # Imprimir valores em 5 momentos específicos do tempo entre 0 e 2 segundos
    for t in np.linspace(0, 2, 5):
        idx = int(t / (time[-1] / num_steps))
        print("Tempo = {:.2f} s || Angulo(Graus) = {:.2f} || Aceleração angular(Rad/s^2) = {:.4f}".format(
            time[idx], np.degrees(theta[idx]), omega[idx]))

    return time, theta

# Parâmetros do pêndulo
theta0 = np.radians(45)  # Ângulo inicial em graus
omega0 = 0.0  # Velocidade angular inicial
g = 9.8  # Aceleração devido à gravidade (m/s^2)
L = 1.0  # Comprimento da corda (metros)

# Parâmetros de simulação
dt = 0.001  # Passo de tempo
num_steps = 10000  # Número de passos

# Resolvendo a EDO usando o método de Euler
time, theta = solve_pendulum_euler(theta0, omega0, g, L, dt, num_steps)

# Plotando o resultado
plt.plot(time, np.degrees(theta))
plt.title('Pêndulo Simples - Método de Euler')
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (graus)')
plt.show()
