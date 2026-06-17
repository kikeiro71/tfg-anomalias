# generador_datos.py — Genera datos biométricos simulados con anomalías controladas
# ---------------------------------------------------------------------------------
# Este script crea un CSV con medidas simuladas compatibles con las señales
# del dataset de Kaggle. Sirve como alternativa cuando no se dispone del
# dataset real, y para ampliar los datos con más usuarios y registros.

import numpy as np
import pandas as pd

from config import SENALES

# ── Parámetros de generación ─────────────────────────────────────────────

NUM_USUARIOS = 20          # Usuarios simulados
REGISTROS_POR_USUARIO = 50 # Registros diarios por usuario
PORCENTAJE_ANOMALIAS = 5   # % de filas que serán anómalas

np.random.seed(42)


def _linea_base_usuario(user_id: int) -> dict:
    """
    Genera parámetros de línea base individuales para un usuario.
    Cada persona tiene valores centrales ligeramente diferentes.
    """
    d = (user_id - 1) * 0.3  # pequeña variación entre usuarios
    return {
        "frecuencia_cardiaca": 72 + d,
        "pasos_diarios":       6000 + d * 200,
        "horas_sueno":         7.0 - d * 0.05,
        "calidad_sueno":       7 - d * 0.1,
        "nivel_estres":        4 + d * 0.1,
        "actividad_fisica":    45 + d * 2,
        "presion_sistolica":   120 + d,
        "presion_diastolica":  78 + d * 0.5,
    }


def _generar_muestra_normal(base: dict) -> dict:
    """Genera una muestra normal con ruido gaussiano alrededor de la línea base."""
    return {
        "frecuencia_cardiaca": np.clip(np.random.normal(base["frecuencia_cardiaca"], 6), 40, 130),
        "pasos_diarios":       np.clip(np.random.normal(base["pasos_diarios"], 1500), 0, 20000),
        "horas_sueno":         np.clip(np.random.normal(base["horas_sueno"], 0.8), 2, 12),
        "calidad_sueno":       np.clip(np.random.normal(base["calidad_sueno"], 1.0), 1, 10),
        "nivel_estres":        np.clip(np.random.normal(base["nivel_estres"], 1.5), 1, 10),
        "actividad_fisica":    np.clip(np.random.normal(base["actividad_fisica"], 12), 0, 120),
        "presion_sistolica":   np.clip(np.random.normal(base["presion_sistolica"], 8), 80, 200),
        "presion_diastolica":  np.clip(np.random.normal(base["presion_diastolica"], 5), 50, 120),
    }


def _generar_muestra_anomala(base: dict) -> dict:
    """Genera una muestra anómala desplazando valores fuera del rango normal."""
    muestra = _generar_muestra_normal(base)

    senales_a_alterar = np.random.choice(
        list(SENALES.keys()),
        size=np.random.randint(1, 4),
        replace=False,
    )

    for senal in senales_a_alterar:
        rango = SENALES[senal]
        if np.random.random() < 0.5:
            muestra[senal] = rango["max"] + np.random.uniform(
                rango["max"] * 0.1, rango["max"] * 0.3
            )
        else:
            muestra[senal] = max(0, rango["min"] - np.random.uniform(
                rango["min"] * 0.05, rango["min"] * 0.15
            ))

    return muestra


def generar_dataset() -> pd.DataFrame:
    """
    Genera el dataset completo con datos normales y anomalías inyectadas.
    """
    registros = []

    for uid in range(1, NUM_USUARIOS + 1):
        base = _linea_base_usuario(uid)

        for _ in range(REGISTROS_POR_USUARIO):
            es_anomalia = np.random.random() < (PORCENTAJE_ANOMALIAS / 100)

            if es_anomalia:
                muestra = _generar_muestra_anomala(base)
            else:
                muestra = _generar_muestra_normal(base)

            registros.append({
                "usuario": f"user_{uid}",
                **muestra,
                "es_anomalia": int(es_anomalia),
            })

    df = pd.DataFrame(registros)

    # Redondear valores
    df["frecuencia_cardiaca"] = df["frecuencia_cardiaca"].round(0).astype(int)
    df["pasos_diarios"] = df["pasos_diarios"].round(0).astype(int)
    df["horas_sueno"] = df["horas_sueno"].round(1)
    df["calidad_sueno"] = df["calidad_sueno"].round(0).astype(int)
    df["nivel_estres"] = df["nivel_estres"].round(0).astype(int)
    df["actividad_fisica"] = df["actividad_fisica"].round(0).astype(int)
    df["presion_sistolica"] = df["presion_sistolica"].round(0).astype(int)
    df["presion_diastolica"] = df["presion_diastolica"].round(0).astype(int)

    return df


# ── Punto de entrada ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generando datos simulados...")
    df = generar_dataset()

    ruta = "datos/biometricos_simulados.csv"
    df.to_csv(ruta, index=False)

    total = len(df)
    anomalas = df["es_anomalia"].sum()
    print(f"Dataset guardado en '{ruta}'")
    print(f"  Total registros : {total}")
    print(f"  Anomalías       : {anomalas} ({anomalas/total*100:.1f}%)")
    print(f"  Usuarios        : {df['usuario'].nunique()}")
    print(f"\nPrimeras filas:")
    print(df.head(10).to_string(index=False))
