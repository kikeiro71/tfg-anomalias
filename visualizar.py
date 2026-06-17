# visualizar.py — Visualización de datos biométricos y anomalías
# ---------------------------------------------------------------
# Genera gráficas con matplotlib para incluir en la memoria del TFG.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import SENALES

DIR_GRAFICAS = Path("datos/graficas")
DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)


def grafica_senales_usuario(df: pd.DataFrame, usuario: str):
    """
    Genera un gráfico con subplots de todas las señales de un usuario,
    marcando las anomalías en rojo.
    """
    datos = df[df["usuario"] == usuario].copy()
    datos = datos.reset_index(drop=True)

    normales = datos[datos["es_anomalia"] == 0]
    anomalas = datos[datos["es_anomalia"] == 1]

    senales = list(SENALES.keys())
    fig, axes = plt.subplots(len(senales), 1, figsize=(14, 3 * len(senales)), sharex=True)
    fig.suptitle(f"Señales biométricas — {usuario}", fontsize=14, fontweight="bold")

    for i, senal in enumerate(senales):
        ax = axes[i]
        info = SENALES[senal]

        # Datos normales en azul
        ax.plot(
            normales.index, normales[senal],
            ".", color="#3498db", markersize=4, alpha=0.6, label="Normal",
        )
        # Anomalías en rojo
        if not anomalas.empty:
            ax.scatter(
                anomalas.index, anomalas[senal],
                color="#e74c3c", s=20, zorder=5, label="Anomalía",
            )
        # Bandas de rango normal
        ax.axhspan(info["min"], info["max"], alpha=0.1, color="green", label="Rango normal")

        ax.set_ylabel(f"{senal}\n({info['unidad']})", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Registro")
    plt.tight_layout()

    ruta = DIR_GRAFICAS / f"senales_{usuario}.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada: {ruta}")


def grafica_distribucion_scores(df_scores: pd.DataFrame):
    """
    Histograma de los scores del Isolation Forest, coloreado por nivel de alerta.
    df_scores debe tener columnas: score, nivel_alerta
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colores = {
        "normal": "#2ecc71",
        "leve": "#f39c12",
        "moderada": "#e67e22",
        "grave": "#e74c3c",
    }

    for nivel, color in colores.items():
        subset = df_scores[df_scores["nivel_alerta"] == nivel]
        if not subset.empty:
            ax.hist(subset["score"], bins=30, alpha=0.6, color=color, label=nivel)

    ax.set_xlabel("Score Isolation Forest")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de scores por nivel de alerta")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = DIR_GRAFICAS / "distribucion_scores.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada: {ruta}")


def grafica_comparativa_real_vs_anomalo(df: pd.DataFrame):
    """
    Boxplots comparando datos normales vs anómalos para cada señal.
    Útil para la memoria del TFG.
    """
    senales = list(SENALES.keys())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, senal in enumerate(senales):
        ax = axes[i]
        datos_normal = df[df["es_anomalia"] == 0][senal]
        datos_anomalo = df[df["es_anomalia"] == 1][senal]

        ax.boxplot(
            [datos_normal, datos_anomalo],
            labels=["Normal", "Anomalía"],
            patch_artist=True,
            boxprops=[
                dict(facecolor="#3498db", alpha=0.5),
                dict(facecolor="#e74c3c", alpha=0.5),
            ][0:1] * 2,  # Simplificado
        )
        ax.set_title(senal.replace("_", " ").title(), fontsize=9)
        ax.set_ylabel(SENALES[senal]["unidad"], fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Comparativa: datos normales vs anomalías", fontsize=13, fontweight="bold")
    plt.tight_layout()

    ruta = DIR_GRAFICAS / "comparativa_normal_vs_anomalo.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada: {ruta}")


# ── Punto de entrada ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from modelo_anomalias import DetectorAnomalias, COLS_SENALES
    from pathlib import Path as P

    # Cargar el dataset disponible
    ruta_real = "datos/dataset_procesado.csv"
    ruta_simulado = "datos/biometricos_simulados.csv"

    if P(ruta_real).exists():
        ruta = ruta_real
    elif P(ruta_simulado).exists():
        ruta = ruta_simulado
    else:
        print("ERROR: No hay dataset. Ejecuta 'python cargar_kaggle.py' o 'python generador_datos.py'")
        exit(1)

    print(f"Cargando datos desde '{ruta}'...")
    df = pd.read_csv(ruta)

    # Gráficas de señales (primeros 3 usuarios)
    print("\nGenerando gráficas de señales:")
    usuarios = df["usuario"].unique()[:3]
    for usuario in usuarios:
        grafica_senales_usuario(df, usuario)

    # Gráfica comparativa
    print("\nGenerando gráfica comparativa:")
    grafica_comparativa_real_vs_anomalo(df)

    # Gráfica de distribución de scores
    print("\nCalculando scores para todos los registros...")
    detector = DetectorAnomalias()
    detector.cargar()

    registros_scores = []
    for _, fila in df.iterrows():
        medida = {col: fila[col] for col in COLS_SENALES}
        resultado = detector.predecir(fila["usuario"], medida)
        if "error" not in resultado:
            registros_scores.append({
                "score": resultado["score"],
                "nivel_alerta": resultado["nivel_alerta"],
            })

    df_scores = pd.DataFrame(registros_scores)
    grafica_distribucion_scores(df_scores)
    print("\nVisualización completada.")
