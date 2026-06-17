# generador_datos.py — Genera datos biométricos simulados con anomalías controladas
# ---------------------------------------------------------------------------------
# Este script crea un CSV con medidas de varios usuarios ficticios.
# Cada usuario tiene una "línea base" ligeramente distinta (variabilidad
# inter-individual), y se inyectan anomalías en un porcentaje configurable
# de las filas para poder evaluar el modelo.

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import SENALES

# ── Parámetros de generación ─────────────────────────────────────────────

NUM_USUARIOS = 5           # Usuarios simulados (user_1 … user_5)
DIAS = 30                  # Días de datos por usuario
MUESTRAS_POR_DIA = 24      # Una muestra por hora
PORCENTAJE_ANOMALIAS = 5   # % de filas que serán anómalas

np.random.seed(42)


def _linea_base_usuario(user_id: int) -> dict:
    """
    Genera parámetros de línea base individuales para un usuario.
    Cada persona tiene valores centrales ligeramente diferentes.
    """
    desplazamiento = (user_id - 1) * 0.5  # pequeña variación entre usuarios
    return {
        "frecuencia_cardiaca": 72 + desplazamiento,
        "saturacion_oxigeno":  97 - desplazamiento * 0.2,
        "temperatura":         36.5 + desplazamiento * 0.1,
        "pasos_hora":          200 + desplazamiento * 20,
        "horas_sueno":         7.0 - desplazamiento * 0.1,
    }


def _generar_muestra_normal(base: dict) -> dict:
    """Genera una muestra normal con ruido gaussiano alrededor de la línea base."""
    return {
        "frecuencia_cardiaca": np.clip(
            np.random.normal(base["frecuencia_cardiaca"], 8), 40, 180
        ),
        "saturacion_oxigeno": np.clip(
            np.random.normal(base["saturacion_oxigeno"], 1.5), 70, 100
        ),
        "temperatura": np.clip(
            np.random.normal(base["temperatura"], 0.3), 34.0, 41.0
        ),
        "pasos_hora": np.clip(
            np.random.normal(base["pasos_hora"], 100), 0, 2000
        ),
        "horas_sueno": np.clip(
            np.random.normal(base["horas_sueno"], 1.0), 0, 14
        ),
    }


def _generar_muestra_anomala(base: dict) -> dict:
    """
    Genera una muestra anómala desplazando uno o más valores fuera del
    rango normal.  Simula situaciones como fiebre, taquicardia, desaturación, etc.
    """
    muestra = _generar_muestra_normal(base)

    # Elegir aleatoriamente qué señales alterar (al menos una)
    senales_a_alterar = np.random.choice(
        list(SENALES.keys()),
        size=np.random.randint(1, 4),
        replace=False,
    )

    for senal in senales_a_alterar:
        rango = SENALES[senal]
        # Desplazar el valor muy por encima o por debajo del rango normal
        if np.random.random() < 0.5:
            muestra[senal] = rango["max"] + np.random.uniform(
                rango["max"] * 0.1, rango["max"] * 0.3
            )
        else:
            muestra[senal] = rango["min"] - np.random.uniform(
                rango["min"] * 0.05, rango["min"] * 0.15
            )

    return muestra


def generar_dataset() -> pd.DataFrame:
    """
    Genera el dataset completo con datos normales y anomalías inyectadas.
    Devuelve un DataFrame con columnas:
        usuario, timestamp, frecuencia_cardiaca, saturacion_oxigeno,
        temperatura, pasos_hora, horas_sueno, es_anomalia
    """
    registros = []
    fecha_inicio = datetime(2026, 5, 1)

    for uid in range(1, NUM_USUARIOS + 1):
        base = _linea_base_usuario(uid)
        total_muestras = DIAS * MUESTRAS_POR_DIA

        for i in range(total_muestras):
            timestamp = fecha_inicio + timedelta(hours=i)
            es_anomalia = np.random.random() < (PORCENTAJE_ANOMALIAS / 100)

            if es_anomalia:
                muestra = _generar_muestra_anomala(base)
            else:
                muestra = _generar_muestra_normal(base)

            registros.append({
                "usuario": f"user_{uid}",
                "timestamp": timestamp.isoformat(),
                **muestra,
                "es_anomalia": int(es_anomalia),
            })

    df = pd.DataFrame(registros)

    # Redondear valores para mayor legibilidad
    df["frecuencia_cardiaca"] = df["frecuencia_cardiaca"].round(1)
    df["saturacion_oxigeno"] = df["saturacion_oxigeno"].round(1)
    df["temperatura"] = df["temperatura"].round(2)
    df["pasos_hora"] = df["pasos_hora"].round(0).astype(int)
    df["horas_sueno"] = df["horas_sueno"].round(1)

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
