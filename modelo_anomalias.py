# modelo_anomalias.py — Modelo de detección de anomalías con Isolation Forest
# ---------------------------------------------------------------------------
# Entrena un modelo por usuario (línea base individual) y permite predecir
# si una nueva medida es anómala y con qué nivel de gravedad.

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import MODELO_CONFIG, UMBRALES_ALERTA, SENALES

# Columnas de señales biométricas que usa el modelo
COLS_SENALES = list(SENALES.keys())

# Directorio donde se guardan los modelos entrenados
DIR_MODELOS = Path("datos/modelos")
DIR_MODELOS.mkdir(parents=True, exist_ok=True)


class DetectorAnomalias:
    """
    Detector de anomalías basado en Isolation Forest.
    Se entrena un modelo independiente por usuario para respetar
    la variabilidad inter-individual (línea base personal).
    """

    def __init__(self):
        # Diccionarios: usuario -> modelo / scaler
        self.modelos: dict[str, IsolationForest] = {}
        self.scalers: dict[str, StandardScaler] = {}

    # ── Entrenamiento ────────────────────────────────────────────────

    def entrenar(self, df: pd.DataFrame) -> dict:
        """
        Entrena un modelo por cada usuario presente en el DataFrame.
        Solo usa las filas NO marcadas como anomalía para aprender
        el comportamiento normal de cada persona.

        Retorna un diccionario con métricas básicas por usuario.
        """
        resultados = {}

        for usuario in df["usuario"].unique():
            # Filtrar datos normales del usuario
            datos_usuario = df[
                (df["usuario"] == usuario) & (df["es_anomalia"] == 0)
            ][COLS_SENALES]

            if datos_usuario.empty:
                continue

            # Escalar los datos (el Isolation Forest funciona mejor con datos normalizados)
            scaler = StandardScaler()
            X = scaler.fit_transform(datos_usuario)

            # Entrenar el modelo
            modelo = IsolationForest(**MODELO_CONFIG)
            modelo.fit(X)

            self.modelos[usuario] = modelo
            self.scalers[usuario] = scaler

            resultados[usuario] = {
                "muestras_entrenamiento": len(datos_usuario),
                "estado": "entrenado",
            }

        return resultados

    # ── Predicción ───────────────────────────────────────────────────

    def predecir(self, usuario: str, medida: dict) -> dict:
        """
        Evalúa una medida individual y devuelve:
        - es_anomalia: True/False
        - score: puntuación del modelo (más negativo = más anómalo)
        - nivel_alerta: "normal", "leve", "moderada" o "grave"
        - detalles: señales fuera de rango
        """
        if usuario not in self.modelos:
            return {"error": f"No hay modelo entrenado para '{usuario}'"}

        modelo = self.modelos[usuario]
        scaler = self.scalers[usuario]

        # Preparar la medida como array
        valores = np.array([[medida.get(col, 0) for col in COLS_SENALES]])
        X = scaler.transform(valores)

        # Obtener predicción y score
        prediccion = modelo.predict(X)[0]        # 1 = normal, -1 = anomalía
        score = modelo.score_samples(X)[0]        # Score continuo

        # Clasificar el nivel de alerta según los umbrales
        nivel = self._clasificar_nivel(score)

        # Detectar qué señales están fuera de rango
        detalles = self._analizar_senales(medida)

        return {
            "es_anomalia": prediccion == -1 or nivel != "normal",
            "score": round(float(score), 4),
            "nivel_alerta": nivel,
            "senales_fuera_de_rango": detalles,
        }

    def _clasificar_nivel(self, score: float) -> str:
        """Clasifica el score en un nivel de alerta."""
        if score < UMBRALES_ALERTA["grave"]:
            return "grave"
        elif score < UMBRALES_ALERTA["moderada"]:
            return "moderada"
        elif score < UMBRALES_ALERTA["leve"]:
            return "leve"
        return "normal"

    def _analizar_senales(self, medida: dict) -> list[dict]:
        """Identifica qué señales están fuera de su rango fisiológico normal."""
        fuera_de_rango = []
        for senal, rango in SENALES.items():
            valor = medida.get(senal)
            if valor is None:
                continue
            if valor < rango["min"]:
                fuera_de_rango.append({
                    "senal": senal,
                    "valor": valor,
                    "rango_normal": f"{rango['min']}–{rango['max']} {rango['unidad']}",
                    "tipo": "por_debajo",
                })
            elif valor > rango["max"]:
                fuera_de_rango.append({
                    "senal": senal,
                    "valor": valor,
                    "rango_normal": f"{rango['min']}–{rango['max']} {rango['unidad']}",
                    "tipo": "por_encima",
                })
        return fuera_de_rango

    # ── Persistencia ─────────────────────────────────────────────────

    def guardar(self, ruta: Path = DIR_MODELOS):
        """Guarda todos los modelos y scalers en disco."""
        for usuario in self.modelos:
            with open(ruta / f"{usuario}_modelo.pkl", "wb") as f:
                pickle.dump(self.modelos[usuario], f)
            with open(ruta / f"{usuario}_scaler.pkl", "wb") as f:
                pickle.dump(self.scalers[usuario], f)
        print(f"Modelos guardados en '{ruta}/' ({len(self.modelos)} usuarios)")

    def cargar(self, ruta: Path = DIR_MODELOS):
        """Carga modelos previamente guardados."""
        for archivo in ruta.glob("*_modelo.pkl"):
            usuario = archivo.stem.replace("_modelo", "")
            with open(archivo, "rb") as f:
                self.modelos[usuario] = pickle.load(f)
            with open(ruta / f"{usuario}_scaler.pkl", "rb") as f:
                self.scalers[usuario] = pickle.load(f)
        print(f"Modelos cargados: {list(self.modelos.keys())}")


# ── Punto de entrada: entrenar y guardar ─────────────────────────────────

if __name__ == "__main__":
    # Cargar datos simulados
    ruta_csv = "datos/biometricos_simulados.csv"
    print(f"Cargando datos desde '{ruta_csv}'...")
    df = pd.read_csv(ruta_csv)

    # Entrenar
    detector = DetectorAnomalias()
    resultados = detector.entrenar(df)

    for usuario, info in resultados.items():
        print(f"  {usuario}: {info['muestras_entrenamiento']} muestras → {info['estado']}")

    # Guardar modelos
    detector.guardar()

    # Prueba rápida con una medida anómala
    print("\n── Prueba de predicción ──")
    medida_anomala = {
        "frecuencia_cardiaca": 160,   # Taquicardia
        "saturacion_oxigeno": 85,     # Desaturación
        "temperatura": 39.5,          # Fiebre
        "pasos_hora": 0,
        "horas_sueno": 2,             # Poco sueño
    }
    resultado = detector.predecir("user_1", medida_anomala)
    print(f"  Medida: {medida_anomala}")
    print(f"  Resultado: {resultado}")
