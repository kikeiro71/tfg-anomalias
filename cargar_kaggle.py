# cargar_kaggle.py — Carga y preprocesa el dataset real de Kaggle
# ----------------------------------------------------------------
# Dataset: "Sleep Health and Lifestyle Dataset"
# https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
#
# Este script transforma el CSV de Kaggle al formato que espera
# nuestro modelo, y lo enriquece con anomalías sintéticas para
# poder evaluar la detección.

import pandas as pd
import numpy as np
from pathlib import Path

from config import MAPEO_COLUMNAS_KAGGLE, SENALES

# Ruta donde el usuario debe colocar el CSV descargado de Kaggle
RUTA_CSV_KAGGLE = Path("datos/Sleep_health_and_lifestyle_dataset.csv")
RUTA_CSV_SALIDA = Path("datos/dataset_procesado.csv")

PORCENTAJE_ANOMALIAS_SINTETICAS = 5  # % de filas anómalas a inyectar

np.random.seed(42)


def cargar_y_transformar() -> pd.DataFrame:
    """
    Lee el CSV de Kaggle, renombra columnas y extrae la presión arterial
    (viene como "130/85" y la separamos en sistólica y diastólica).
    """
    if not RUTA_CSV_KAGGLE.exists():
        print(f"ERROR: No se encontró el archivo '{RUTA_CSV_KAGGLE}'")
        print("Descárgalo desde:")
        print("  https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")
        print(f"y colócalo en la carpeta 'datos/' con el nombre:")
        print(f"  {RUTA_CSV_KAGGLE.name}")
        raise FileNotFoundError(RUTA_CSV_KAGGLE)

    df = pd.read_csv(RUTA_CSV_KAGGLE)
    print(f"CSV de Kaggle cargado: {len(df)} registros, {len(df.columns)} columnas")

    # Separar la presión arterial "sistólica/diastólica"
    if "Blood Pressure" in df.columns:
        bp = df["Blood Pressure"].str.split("/", expand=True)
        df["presion_sistolica"] = pd.to_numeric(bp[0], errors="coerce")
        df["presion_diastolica"] = pd.to_numeric(bp[1], errors="coerce")

    # Renombrar columnas según nuestro mapeo
    df = df.rename(columns=MAPEO_COLUMNAS_KAGGLE)

    # Convertir el Person ID a formato "user_X"
    df["usuario"] = "user_" + df["usuario"].astype(str)

    # Seleccionar solo las columnas que necesitamos
    columnas_modelo = list(SENALES.keys())
    columnas_finales = ["usuario"] + columnas_modelo
    df = df[columnas_finales].copy()

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Marcar todos los datos reales como normales (no son anomalías)
    df["es_anomalia"] = 0

    print(f"Dataset transformado: {len(df)} registros, {len(columnas_modelo)} señales")
    print(f"Usuarios únicos: {df['usuario'].nunique()}")
    return df


def inyectar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inyecta anomalías sintéticas para poder evaluar el modelo.
    Toma filas reales y desplaza sus valores fuera de rango.
    Esto es necesario porque el dataset original no tiene
    anomalías etiquetadas.
    """
    n_anomalias = int(len(df) * PORCENTAJE_ANOMALIAS_SINTETICAS / 100)

    # Seleccionar filas aleatorias como base para las anomalías
    indices = np.random.choice(df.index, size=n_anomalias, replace=False)
    anomalias = df.loc[indices].copy()
    anomalias["es_anomalia"] = 1

    # Alterar entre 1 y 3 señales por fila
    for idx in anomalias.index:
        senales_a_alterar = np.random.choice(
            list(SENALES.keys()),
            size=np.random.randint(1, 4),
            replace=False,
        )
        for senal in senales_a_alterar:
            rango = SENALES[senal]
            valor_actual = anomalias.loc[idx, senal]
            # Desplazar por encima o por debajo del rango
            if np.random.random() < 0.5:
                anomalias.loc[idx, senal] = rango["max"] + abs(valor_actual) * 0.3
            else:
                anomalias.loc[idx, senal] = max(0, rango["min"] - abs(valor_actual) * 0.2)

    # Combinar datos originales + anomalías inyectadas
    df_final = pd.concat([df, anomalias], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_final


# ── Punto de entrada ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  Carga del dataset de Kaggle")
    print("═" * 60)

    # Paso 1: Cargar y transformar
    df = cargar_y_transformar()

    # Paso 2: Inyectar anomalías sintéticas
    print(f"\nInyectando {PORCENTAJE_ANOMALIAS_SINTETICAS}% de anomalías sintéticas...")
    df = inyectar_anomalias(df)

    # Paso 3: Guardar
    df.to_csv(RUTA_CSV_SALIDA, index=False)

    total = len(df)
    anomalas = df["es_anomalia"].sum()
    print(f"\nDataset final guardado en '{RUTA_CSV_SALIDA}'")
    print(f"  Total registros : {total}")
    print(f"  Datos reales    : {total - anomalas}")
    print(f"  Anomalías       : {anomalas} ({anomalas/total*100:.1f}%)")
    print(f"  Usuarios        : {df['usuario'].nunique()}")
    print(f"\nPrimeras filas:")
    print(df.head(10).to_string(index=False))
