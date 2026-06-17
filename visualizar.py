# visualizar.py — Visualización de datos biométricos y anomalías
# ---------------------------------------------------------------
# Genera gráficas con matplotlib para incluir en la memoria del TFG.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    datos["timestamp"] = pd.to_datetime(datos["timestamp"])

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
            normales["timestamp"], normales[senal],
            ".", color="#3498db", markersize=2, alpha=0.5, label="Normal",
        )
        # Anomalías en rojo
        ax.scatter(
            anomalas["timestamp"], anomalas[senal],
            color="#e74c3c", s=15, zorder=5, label="Anomalía",
        )
        # Bandas de rango normal
        ax.axhspan(info["min"], info["max"], alpha=0.1, color="green", label="Rango normal")

        ax.set_ylabel(f"{senal}\n({info['unidad']})", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axes[-1].set_xlabel("Fecha")
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


# ── Punto de entrada ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from modelo_anomalias import DetectorAnomalias, COLS_SENALES

    print("Cargando datos...")
    df = pd.read_csv("datos/biometricos_simulados.csv")

    # Generar gráficas de señales por usuario
    print("\nGenerando gráficas de señales:")
    for usuario in df["usuario"].unique():
        grafica_senales_usuario(df, usuario)

    # Generar gráfica de distribución de scores
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
