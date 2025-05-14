import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


filas, columnas = 5, 5
inicio = (0, 0)
residuos = [(2, 3), (3, 4), (3, 1)]
zonas_toxicas = [(1, 4), (4, 4), (1, 2)]
acciones = ['arriba', 'abajo', 'izquierda', 'derecha']
n_acciones = len(acciones)


def visualizar_entorno():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(columnas + 1))  # +1 para incluir el último tick
    ax.set_yticks(np.arange(filas + 1))    # +1 para incluir el último tick
    ax.set_xlim(0, columnas)
    ax.set_ylim(0, filas)
    ax.grid(True)

    for i in range(filas):
        for j in range(columnas):
            if (i, j) == inicio:
                ax.text(j + 0.5, i + 0.5, 'Inicio', ha='center', va='center', fontsize=12, color='red')
            elif (i, j) in residuos:
                ax.text(j + 0.5, i + 0.5, '♻️', ha='center', va='center', fontsize=16)
            elif (i, j) in zonas_toxicas:
                ax.text(j + 0.5, i + 0.5, '☠️', ha='center', va='center', fontsize=16)

    ax.invert_yaxis()
    plt.title("Mapa con residuos y zonas tóxicas")
    plt.tight_layout()
    return fig


def obtener_recompensa(estado):
    if estado in residuos:
        return 5
    elif estado in zonas_toxicas:
        return -5
    else:
        return -1


def mover(estado, accion):
    i, j = estado
    if accion == 0 and i > 0:             i -= 1  #arriba
    elif accion == 1 and i < filas - 1:   i += 1  #abajo
    elif accion == 2 and j > 0:           j -= 1  #izquierda
    elif accion == 3 and j < columnas - 1: j += 1  #derecha
    return (i, j)


def entrenar_q_learning(episodios=1000, alpha=0.1, gamma=0.9, epsilon=0.2):
    Q = np.zeros((filas, columnas, n_acciones))
    recompensas_totales = []
    tiempo_inicio = time.time()

    for ep in range(episodios):
        estado = inicio
        total = 0
        recogidos = set()

        for _ in range(100):
            # Selección de acción (epsilon-greedy)
            if np.random.rand() < epsilon:
                accion = np.random.randint(n_acciones)
            else:
                accion = np.argmax(Q[estado])

            nuevo_estado = mover(estado, accion)
            recompensa = obtener_recompensa(nuevo_estado)

            
            if nuevo_estado in residuos and nuevo_estado not in recogidos:
                recogidos.add(nuevo_estado)
            elif nuevo_estado in residuos:
                recompensa = -1  

            total += recompensa

            # Actualización Q-learning
            Q[estado][accion] += alpha * (recompensa + gamma * np.max(Q[nuevo_estado]) - Q[estado][accion])
            estado = nuevo_estado

            if len(recogidos) == len(residuos):
                break  # terminó al recoger todos los residuos

        recompensas_totales.append(total)

    tiempo_entrenamiento = time.time() - tiempo_inicio
    return Q, recompensas_totales, tiempo_entrenamiento

def obtener_trayectoria(Q):
    estado = inicio
    trayectoria = [estado]
    recompensas = [obtener_recompensa(estado)]
    recogidos = set()

    for _ in range(100):
        accion = np.argmax(Q[estado])
        nuevo_estado = mover(estado, accion)
        recompensa = obtener_recompensa(nuevo_estado)

        if nuevo_estado in residuos and nuevo_estado not in recogidos:
            recogidos.add(nuevo_estado)
        elif nuevo_estado in residuos:
            recompensa = -1

        trayectoria.append(nuevo_estado)
        recompensas.append(recompensa)
        estado = nuevo_estado

        if len(recogidos) == len(residuos):
            break

    return trayectoria, recompensas


def visualizar_trayectoria(trayectoria, recompensas):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(columnas + 1))  
    ax.set_yticks(np.arange(filas + 1))    
    ax.set_xlim(0, columnas)
    ax.set_ylim(0, filas)
    ax.grid(True)

    for i in range(filas):
        for j in range(columnas):
            if (i, j) == inicio:
                ax.text(j + 0.5, i + 0.5, 'Inicio', ha='center', va='center', fontsize=12, color='red')
            elif (i, j) in residuos:
                ax.text(j + 0.5, i + 0.5, '♻️', ha='center', va='center', fontsize=16)
            elif (i, j) in zonas_toxicas:
                ax.text(j + 0.5, i + 0.5, '☠️', ha='center', va='center', fontsize=16)

    for idx in range(1, len(trayectoria)):
        y1, x1 = trayectoria[idx - 1]
        y2, x2 = trayectoria[idx]
        dx, dy = x2 - x1, y2 - y1
        ax.arrow(x1 + 0.5, y1 + 0.5, dx * 0.8, dy * 0.8, head_width=0.2, head_length=0.2, fc='green', ec='green')
        ax.text(x2 + 0.5, y2 + 0.5, f"{recompensas[idx]:.1f}", fontsize=8, color='darkgreen')

    ax.invert_yaxis()
    plt.title("Trayectoria óptima recogiendo residuos")
    plt.tight_layout()
    return fig


def visualizar_metricas(recompensas, tiempo_entrenamiento):
    promedio = np.mean(recompensas)
    varianza = np.var(recompensas)
    ventana = 20
    recompensa_media = [np.mean(recompensas[max(0, i - ventana):i+1]) for i in range(len(recompensas))]

    fig = plt.figure(figsize=(10, 4))
    plt.plot(recompensas, label="Recompensa por episodio", alpha=0.5)
    plt.plot(recompensa_media, label="Promedio móvil (20)", color="red")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Evolución de la recompensa")
    plt.legend()
    plt.tight_layout()

    trayectoria, recompensas_camino = obtener_trayectoria(Q)
    recompensa_total = sum(recompensas_camino)

    st.write(f"Recompensa promedio: {promedio:.2f}")
    st.write(f"Varianza (estabilidad): {varianza:.2f}")
    st.write(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} s")
    st.write(f"Recompensa total del recorrido final: {recompensa_total:.2f}")

    return fig


st.title("Simulación Q-learning: Recogida de Residuos")


#Mostrar mapa inicial
st.subheader("Entorno de aprendizaje")
fig_entorno = visualizar_entorno()
st.pyplot(fig_entorno)


st.sidebar.header("Parámetros de entrenamiento")
episodios = st.sidebar.slider("Episodios", 100, 500, 1000, step=100)
alpha = st.sidebar.slider("Tasa de aprendizaje", 0.01, 1.0, 0.1, step=0.01)
gamma = st.sidebar.slider("Factor de descuento", 0.1, 1.0, 0.9, step=0.05)
epsilon = st.sidebar.slider("Exploración", 0.0, 1.0, 0.2, step=0.05)

if st.button("Entrenar agente"):
    with st.spinner('Entrenando al agente...'):
        Q, recompensas_totales, tiempo_entrenamiento = entrenar_q_learning(episodios, alpha, gamma, epsilon)
    
    trayectoria, recompensas_camino = obtener_trayectoria(Q)

    st.subheader("Trayectoria óptima aprendida")
    fig1 = visualizar_trayectoria(trayectoria, recompensas_camino)
    st.pyplot(fig1)

    st.subheader("Métricas del entrenamiento")
    fig2 = visualizar_metricas(recompensas_totales, tiempo_entrenamiento)
    st.pyplot(fig2)