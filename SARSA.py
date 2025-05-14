import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

# Tamaño de la grilla
filas, columnas = 5, 5

# Coordenadas válidas
inicio = (0, 0)
residuos_iniciales = {(2, 3), (3, 4), (3, 1)}
zonas_toxicas = {(1, 4), (4, 4), (1, 2)}

# Visualización
def visualizar_entorno(residuos_presentes):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(columnas))
    ax.set_yticks(np.arange(filas))
    ax.grid(True)

    for i in range(filas):
        for j in range(columnas):
            if (i, j) == inicio:
                ax.text(j, i, 'S', ha='center', va='center', fontsize=12, color='blue')
            elif (i, j) in residuos_presentes:
                ax.text(j, i, '♻️', ha='center', va='center', fontsize=12)
            elif (i, j) in zonas_toxicas:
                ax.text(j, i, '☠️', ha='center', va='center', fontsize=12)
    ax.invert_yaxis()
    plt.title("Mapa con residuos y zonas tóxicas")
    plt.tight_layout()
    return fig

# Función para mover el robot
movimientos = ['UP', 'DOWN', 'LEFT', 'RIGHT']

def mover(posicion, movimiento, residuos_presentes):
    x, y = posicion
    nueva_pos = posicion
    if movimiento == 'UP':
        nueva_pos = (max(x - 1, 0), y)
    elif movimiento == 'DOWN':
        nueva_pos = (min(x + 1, filas - 1), y)
    elif movimiento == 'LEFT':
        nueva_pos = (x, max(y - 1, 0))
    elif movimiento == 'RIGHT':
        nueva_pos = (x, min(y + 1, columnas - 1))

    recompensa = -0.1 # Ligeramente negativo para fomentar la eficiencia
    done = False
    residuo_recogido = None

    if nueva_pos in zonas_toxicas:
        recompensa = -5
        done = True
    elif nueva_pos in residuos_presentes:
        recompensa = 5
        residuo_recogido = nueva_pos
        residuos_presentes.remove(nueva_pos) # Eliminar residuo recogido

    return nueva_pos, recompensa, done, residuos_presentes, residuo_recogido

def epsilon_greedy(Q, estado, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(movimientos)
    else:
        q_values = Q[estado]
        max_q = np.max(q_values)
        best_actions = [movimientos[i] for i, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)

# --- SARSA ---

def entrenar_sarsa(num_episodios, alpha, gamma, epsilon_inicial, epsilon_final = 0.01, epsilon_decay_rate = 0.99):
    Q = np.zeros((filas, columnas, len(movimientos)))
    recompensas_por_episodio = []
    recompensa_acumulada = 0
    promedios_recompensa = []
    tiempo_inicio = time.time()

    for episodio in range(num_episodios):
        estado = inicio
        epsilon = max(epsilon_final, epsilon_inicial * (epsilon_decay_rate**episodio))
        accion = epsilon_greedy(Q, estado, epsilon)
        recompensa_episodio = 0
        done = False
        residuos_presentes = set(residuos_iniciales)

        while not done:
            nuevo_estado, recompensa, done, residuos_presentes, _ = mover(estado, accion, set(residuos_presentes)) # Pasar copia para no modificar original
            nuevo_accion = epsilon_greedy(Q, nuevo_estado, epsilon)
            estado_idx = estado[0], estado[1]
            nuevo_estado_idx = nuevo_estado[0], nuevo_estado[1]
            accion_idx = movimientos.index(accion)
            nuevo_accion_idx = movimientos.index(nuevo_accion)

            td_target = recompensa + gamma * Q[nuevo_estado_idx][nuevo_accion_idx]
            td_error = td_target - Q[estado_idx][accion_idx]
            Q[estado_idx][accion_idx] += alpha * td_error

            estado = nuevo_estado
            accion = nuevo_accion
            recompensa_episodio += recompensa
            if len(residuos_presentes) == 0 and not done: # Condición de fin si todos los residuos se recogen
                done = True

        recompensas_por_episodio.append(recompensa_episodio)
        recompensa_acumulada += recompensa_episodio
        promedios_recompensa.append(recompensa_acumulada / (episodio + 1))

    tiempo_final = time.time() - tiempo_inicio
    return Q, recompensas_por_episodio, tiempo_final

def obtener_trayectoria_sarsa(Q):
    estado = inicio
    trayectoria = [estado]
    recompensas = [0]
    residuos_recogidos_en_trayectoria = set()
    residuos_presentes = set(residuos_iniciales)

    for _ in range(200): # Aumentar límite de pasos para la trayectoria final
        if len(residuos_presentes) == 0:
            break
        accion_idx = np.argmax(Q[estado])
        accion = movimientos[accion_idx]
        nuevo_estado, recompensa, done, residuos_presentes, residuo_recogido = mover(estado, accion, residuos_presentes)

        trayectoria.append(nuevo_estado)
        recompensas.append(recompensa)
        estado = nuevo_estado

    return trayectoria, recompensas

def visualizar_trayectoria_sarsa(trayectoria, recompensas):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(columnas))
    ax.set_yticks(np.arange(filas))
    ax.grid(True)

    residuos_dibujados = set()
    for i in range(filas):
        for j in range(columnas):
            if (i, j) == inicio:
                ax.text(j, i, 'S', ha='center', va='center', fontsize=12, color='blue')
            elif (i, j) in residuos_iniciales:
                ax.text(j, i, '♻️', ha='center', va='center', fontsize=12)
                residuos_dibujados.add((i,j))
            elif (i, j) in zonas_toxicas:
                ax.text(j, i, '☠️', ha='center', va='center', fontsize=12)

    for idx in range(1, len(trayectoria)):
        y_prev, x_prev = trayectoria[idx-1]
        y_curr, x_curr = trayectoria[idx]

        ax.annotate("",
                    xy=(x_curr, y_curr),
                    xytext=(x_prev, y_prev),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

        mid_x = (x_prev + x_curr) / 2
        mid_y = (y_prev + y_curr) / 2
        ax.text(mid_x, mid_y, f"{recompensas[idx]:.1f}",
                fontsize=8, color='darkred', ha='center', va='center')

    ax.invert_yaxis()
    plt.title("Trayectoria aprendida (SARSA)")
    plt.tight_layout()
    return fig

def visualizar_metricas_sarsa(recompensas, tiempo_entrenamiento, Q):
    promedio = np.mean(recompensas)
    varianza = np.var(recompensas)
    ventana = 20
    recompensa_media = [np.mean(recompensas[max(0, i - ventana):i+1]) for i in range(len(recompensas))]

    fig = plt.figure(figsize=(10, 4))
    plt.plot(recompensas, label="Recompensa por episodio", alpha=0.5)
    plt.plot(recompensa_media, label="Promedio móvil (20)", color="purple")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Evolución de la recompensa (SARSA)")
    plt.legend()
    plt.tight_layout()

    trayectoria, recompensas_camino = obtener_trayectoria_sarsa(Q)
    recompensa_total = sum(recompensas_camino)

    st.write(f"Recompensa promedio: {promedio:.2f}")
    st.write(f"Varianza (estabilidad): {varianza:.2f}")
    st.write(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} s")
    st.write(f"Recompensa total del recorrido final: {recompensa_total:.2f}")


    return fig

# --- INTERFAZ STREAMLIT PARA SARSA ---
st.title("Simulación SARSA: Recogida de Residuos")
st.markdown("Este entorno simula un agente que aprende a recoger residuos evitando zonas tóxicas usando **SARSA**.")

# Mostrar entorno inicial
st.subheader("Entorno de aprendizaje")
fig_entorno = visualizar_entorno(residuos_iniciales)
st.pyplot(fig_entorno)

# Parámetros configurables
st.sidebar.header("Parámetros de entrenamiento")
neon_episodios = st.sidebar.slider("Episodios", 500, 10000, 2000, step=100) # Aumentar episodios por defecto
neon_alpha = st.sidebar.slider("Tasa de aprendizaje (α)", 0.01, 0.5, 0.1, step=0.01) # Ajustar rango de alpha
neon_gamma = st.sidebar.slider("Factor de descuento (γ)", 0.5, 0.99, 0.9, step=0.01) # Ajustar rango de gamma
epsilon_inicial = st.sidebar.slider("Exploración inicial (ε)", 0.0, 1.0, 0.8, step=0.05) # Epsilon inicial alto

if st.button("Entrenar agente SARSA"):
    with st.spinner('Entrenando con SARSA...'):
        Q, recompensas_por_episodio, tiempo_final = entrenar_sarsa(
            num_episodios=neon_episodios,
            alpha=neon_alpha,
            gamma=neon_gamma,
            epsilon_inicial=epsilon_inicial,
        )

    trayectoria, recompensas_trayectoria = obtener_trayectoria_sarsa(Q)

    st.subheader("Trayectoria aprendida por SARSA")
    fig1 = visualizar_trayectoria_sarsa(trayectoria, recompensas_trayectoria)
    st.pyplot(fig1)

    st.subheader("Métricas del entrenamiento SARSA")
    fig2 = visualizar_metricas_sarsa(recompensas_por_episodio, tiempo_final, Q)
    st.pyplot(fig2)