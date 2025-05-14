import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- PARÁMETROS DEL ENTORNO ---
filas, columnas = 5, 5
inicio = (0, 0)
residuos = [(2, 3), (3, 4), (3, 1)]
zonas_toxicas = [(1, 4), (4, 4), (1, 2)]
acciones = ['arriba', 'abajo', 'izquierda', 'derecha']
n_acciones = len(acciones)

# --- FUNCIONES DEL ENTORNO ---
def obtener_recompensa(estado):
    if estado in residuos:
        return 5
    elif estado in zonas_toxicas:
        return -5
    else:
        return -1

def mover(estado, accion):
    i, j = estado
    if accion == 0 and i > 0:             i -= 1
    elif accion == 1 and i < filas - 1:   i += 1
    elif accion == 2 and j > 0:           j -= 1
    elif accion == 3 and j < columnas - 1:j += 1
    return (i, j)

def visualizar_entorno():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(columnas))
    ax.set_yticks(np.arange(filas))
    ax.grid(True)

    for i in range(filas):
        for j in range(columnas):
            if (i, j) == inicio:
                ax.text(j, i, 'Inicio', ha='center', va='center', fontsize=12, color='red')
            elif (i, j) in residuos:
                ax.text(j, i, '♻️', ha='center', va='center', fontsize=16)
            elif (i, j) in zonas_toxicas:
                ax.text(j, i, '☠️', ha='center', va='center', fontsize=16)

    ax.invert_yaxis()
    plt.title("Mapa con residuos y zonas tóxicas")
    plt.tight_layout()
    return fig

# --- MONTE CARLO ---
def generar_episodio(Q, epsilon):
    estado = inicio
    episodio = []
    recogidos = set()

    for _ in range(100):
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

        episodio.append((estado, accion, recompensa))
        estado = nuevo_estado

        if len(recogidos) == len(residuos):
            break

    return episodio

def entrenar_monte_carlo(episodios=1000, gamma=0.9, epsilon=0.2):
    Q = np.zeros((filas, columnas, n_acciones))
    visitas = [[[0]*n_acciones for _ in range(columnas)] for _ in range(filas)]
    recompensas_totales = []
    tiempo_inicio = time.time()

    for _ in range(episodios):
        episodio = generar_episodio(Q, epsilon)
        G = 0
        visitados = set()
        total_recompensa = 0

        for t in reversed(range(len(episodio))):
            estado, accion, recompensa = episodio[t]
            total_recompensa += recompensa
            G = gamma * G + recompensa
            if (estado, accion) not in visitados:
                visitas[estado[0]][estado[1]][accion] += 1
                alpha = 1 / visitas[estado[0]][estado[1]][accion]
                Q[estado][accion] += alpha * (G - Q[estado][accion])
                visitados.add((estado, accion))

        recompensas_totales.append(total_recompensa)

    tiempo_entrenamiento = time.time() - tiempo_inicio
    return Q, recompensas_totales, tiempo_entrenamiento

# --- VISUALIZACIONES ---
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
    ax.set_xticks(np.arange(columnas))
    ax.set_yticks(np.arange(filas))
    ax.grid(True)

    for i in range(filas):
        for j in range(columnas):
            if (i, j) == inicio:
                ax.text(j, i, 'Inicio', ha='center', va='center', fontsize=12, color='red')
            elif (i, j) in residuos:
                ax.text(j, i, '♻️', ha='center', va='center', fontsize=16)
            elif (i, j) in zonas_toxicas:
                ax.text(j, i, '☠️', ha='center', va='center', fontsize=16)

    for idx in range(1, len(trayectoria)):
        y1, x1 = trayectoria[idx - 1]
        y2, x2 = trayectoria[idx]
        dx, dy = x2 - x1, y2 - y1
        ax.arrow(x1, y1, dx * 0.8, dy * 0.8, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        ax.text(x2, y2, f"{recompensas[idx]:.1f}", fontsize=8, color='darkblue')

    ax.invert_yaxis()
    plt.title("Trayectoria óptima (Monte Carlo)")
    plt.tight_layout()
    return fig

def visualizar_metricas(recompensas, tiempo_entrenamiento, Q):
    promedio = np.mean(recompensas)
    varianza = np.var(recompensas)
    ventana = 20
    recompensa_media = [np.mean(recompensas[max(0, i - ventana):i+1]) for i in range(len(recompensas))]

    fig = plt.figure(figsize=(10, 4))
    plt.plot(recompensas, label="Recompensa por episodio", alpha=0.5)
    plt.plot(recompensa_media, label="Promedio móvil (20)", color="purple")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Evolución de la recompensa (Monte Carlo)")
    plt.legend()
    plt.tight_layout()

    trayectoria, recompensas_camino = obtener_trayectoria(Q)
    recompensa_total = sum(recompensas_camino)

    st.write(f"Recompensa promedio: {promedio:.2f}")
    st.write(f"Varianza (estabilidad): {varianza:.2f}")
    st.write(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} s")
    st.write(f"Recompensa total del recorrido final: {recompensa_total:.2f}")

    return fig

# --- INTERFAZ STREAMLIT ---
st.title("Simulación Monte Carlo: Recogida de Residuos")
st.markdown("Este entorno simula un agente que aprende a recoger residuos evitando zonas tóxicas usando **Monte Carlo**.")

# Mostrar entorno inicial
st.subheader("Entorno de aprendizaje")
fig_entorno = visualizar_entorno()
st.pyplot(fig_entorno)

# Parámetros configurables
st.sidebar.header("Parámetros de entrenamiento")
episodios = st.sidebar.slider("Episodios", 100, 5000, 1000, step=100)
gamma = st.sidebar.slider("Factor de descuento (γ)", 0.1, 1.0, 0.9, step=0.05)
epsilon = st.sidebar.slider("Exploración (ε)", 0.0, 1.0, 0.2, step=0.05)

if st.button("Entrenar agente"):
    with st.spinner('Entrenando con Monte Carlo...'):
        Q, recompensas_totales, tiempo_entrenamiento = entrenar_monte_carlo(episodios, gamma, epsilon)

    trayectoria, recompensas_trayectoria = obtener_trayectoria(Q)

    st.subheader("Trayectoria óptima aprendida")
    fig1 = visualizar_trayectoria(trayectoria, recompensas_trayectoria)
    st.pyplot(fig1)

    st.subheader("Métricas del entrenamiento")
    fig2 = visualizar_metricas(recompensas_totales, tiempo_entrenamiento, Q)
    st.pyplot(fig2)
