# modelo_anomalias.py — Modelo de deteccion de anomalias con Isolation Forest
# ---------------------------------------------------------------------------
# Entrena un modelo global como respaldo y modelos individuales por usuario
# cuando hay suficientes muestras. Calibrado empirico de umbrales.

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import MODELO_CONFIG, UMBRALES_ALERTA, SENALES, MIN_MUESTRAS_USUARIO

# Columnas de senales biometricas que usa el modelo
COLS_SENALES = list(SENALES.keys())

# Directorio donde se guardan los modelos entrenados
DIR_MODELOS = Path("datos/modelos")
DIR_MODELOS.mkdir(parents=True, exist_ok=True)


class DetectorAnomalias:
    """
    Detector de anomalias basado en Isolation Forest.

    Estrategia de entrenamiento:
    1. Siempre se entrena un modelo GLOBAL con todos los datos normales.
    2. Para usuarios con >= MIN_MUESTRAS_USUARIO registros normales,
       se entrena ademas un modelo individual (linea base personal).
    3. En prediccion, se usa el modelo individual si existe;
       si no, se recurre al modelo global.
    """

    def __init__(self):
        self.modelos: dict[str, IsolationForest] = {}
        self.scalers: dict[str, StandardScaler] = {}
        # Modelo global de respaldo
        self.modelo_global: IsolationForest | None = None
        self.scaler_global: StandardScaler | None = None

    # ── Entrenamiento ────────────────────────────────────────────────

    def entrenar(self, df: pd.DataFrame) -> dict:
        """
        Entrena el modelo global y modelos individuales por usuario.
        Solo usa filas NO marcadas como anomalia.
        """
        resultados = {}

        # 1. Modelo global con todos los datos normales
        todos_normales = df[df["es_anomalia"] == 0][COLS_SENALES]
        self.scaler_global = StandardScaler()
        X_global = self.scaler_global.fit_transform(todos_normales)
        self.modelo_global = IsolationForest(**MODELO_CONFIG)
        self.modelo_global.fit(X_global)

        resultados["__global__"] = {
            "muestras_entrenamiento": len(todos_normales),
            "estado": "entrenado",
        }
        print(f"  Modelo global: {len(todos_normales)} muestras -> entrenado")

        # 2. Modelos individuales por usuario (si hay suficientes muestras)
        for usuario in df["usuario"].unique():
            datos_usuario = df[
                (df["usuario"] == usuario) & (df["es_anomalia"] == 0)
            ][COLS_SENALES]

            if len(datos_usuario) < MIN_MUESTRAS_USUARIO:
                resultados[usuario] = {
                    "muestras_entrenamiento": len(datos_usuario),
                    "estado": f"insuficiente (< {MIN_MUESTRAS_USUARIO}), usa global",
                }
                continue

            scaler = StandardScaler()
            X = scaler.fit_transform(datos_usuario)

            modelo = IsolationForest(**MODELO_CONFIG)
            modelo.fit(X)

            self.modelos[usuario] = modelo
            self.scalers[usuario] = scaler

            resultados[usuario] = {
                "muestras_entrenamiento": len(datos_usuario),
                "estado": "entrenado (individual)",
            }

        return resultados

    # ── Prediccion ───────────────────────────────────────────────────

    def predecir(self, usuario: str, medida: dict) -> dict:
        """
        Evalua una medida individual y devuelve:
        - es_anomalia: True/False
        - score: puntuacion del modelo (mas negativo = mas anomalo)
        - nivel_alerta: "normal", "leve", "moderada" o "grave"
        - modelo_usado: "individual" o "global"
        - senales_fuera_de_rango: detalle de senales fuera de rango
        """
        # Elegir modelo: individual si existe, global como respaldo
        if usuario in self.modelos:
            modelo = self.modelos[usuario]
            scaler = self.scalers[usuario]
            modelo_usado = "individual"
        elif self.modelo_global is not None:
            modelo = self.modelo_global
            scaler = self.scaler_global
            modelo_usado = "global"
        else:
            return {"error": "No hay ningun modelo entrenado"}

        # Preparar la medida como DataFrame (evita warning de feature names)
        valores_df = pd.DataFrame([medida], columns=COLS_SENALES)
        X = scaler.transform(valores_df)

        # Obtener score continuo
        score = float(modelo.score_samples(X)[0])

        # Clasificar nivel de alerta SOLO por score (sin usar predict)
        nivel = self._clasificar_nivel(score)

        # Detectar senales fuera de rango fisiologico
        detalles = self._analizar_senales(medida)

        return {
            "es_anomalia": bool(nivel != "normal"),
            "score": round(score, 4),
            "nivel_alerta": nivel,
            "modelo_usado": modelo_usado,
            "senales_fuera_de_rango": detalles,
        }

    def _clasificar_nivel(self, score: float) -> str:
        """Clasifica el score en un nivel de alerta basado en umbrales calibrados."""
        if score < UMBRALES_ALERTA["grave"]:
            return "grave"
        elif score < UMBRALES_ALERTA["moderada"]:
            return "moderada"
        elif score < UMBRALES_ALERTA["leve"]:
            return "leve"
        return "normal"

    def _analizar_senales(self, medida: dict) -> list[dict]:
        """Identifica que senales estan fuera de su rango fisiologico normal."""
        fuera_de_rango = []
        for senal, rango in SENALES.items():
            valor = medida.get(senal)
            if valor is None:
                continue
            if valor < rango["min"]:
                fuera_de_rango.append({
                    "senal": senal,
                    "valor": valor,
                    "rango_normal": f"{rango['min']}-{rango['max']} {rango['unidad']}",
                    "tipo": "por_debajo",
                })
            elif valor > rango["max"]:
                fuera_de_rango.append({
                    "senal": senal,
                    "valor": valor,
                    "rango_normal": f"{rango['min']}-{rango['max']} {rango['unidad']}",
                    "tipo": "por_encima",
                })
        return fuera_de_rango

    # ── Persistencia ─────────────────────────────────────────────────

    def guardar(self, ruta: Path = DIR_MODELOS):
        """Guarda todos los modelos y scalers en disco."""
        # Guardar modelo global
        if self.modelo_global:
            with open(ruta / "global_modelo.pkl", "wb") as f:
                pickle.dump(self.modelo_global, f)
            with open(ruta / "global_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler_global, f)

        # Guardar modelos individuales
        for usuario in self.modelos:
            with open(ruta / f"{usuario}_modelo.pkl", "wb") as f:
                pickle.dump(self.modelos[usuario], f)
            with open(ruta / f"{usuario}_scaler.pkl", "wb") as f:
                pickle.dump(self.scalers[usuario], f)

        n_individuales = len(self.modelos)
        print(f"Modelos guardados en '{ruta}/' (1 global + {n_individuales} individuales)")

    def cargar(self, ruta: Path = DIR_MODELOS):
        """Carga modelos previamente guardados."""
        # Cargar modelo global
        global_modelo_path = ruta / "global_modelo.pkl"
        if global_modelo_path.exists():
            with open(global_modelo_path, "rb") as f:
                self.modelo_global = pickle.load(f)
            with open(ruta / "global_scaler.pkl", "rb") as f:
                self.scaler_global = pickle.load(f)

        # Cargar modelos individuales
        for archivo in ruta.glob("*_modelo.pkl"):
            if archivo.stem == "global_modelo":
                continue
            usuario = archivo.stem.replace("_modelo", "")
            with open(archivo, "rb") as f:
                self.modelos[usuario] = pickle.load(f)
            with open(ruta / f"{usuario}_scaler.pkl", "rb") as f:
                self.scalers[usuario] = pickle.load(f)

        n_individuales = len(self.modelos)
        global_ok = "si" if self.modelo_global else "no"
        print(f"Modelos cargados: global={global_ok}, individuales={n_individuales}")
        if self.modelos:
            print(f"  Usuarios con modelo propio: {list(self.modelos.keys())}")


# ── Punto de entrada: entrenar, evaluar y guardar ────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import classification_report, roc_auc_score

    # Cargar dataset disponible
    ruta_real = Path("datos/dataset_procesado.csv")
    ruta_simulado = Path("datos/biometricos_simulados.csv")

    if ruta_real.exists():
        ruta_csv = str(ruta_real)
        print(f"Usando dataset real (Kaggle): '{ruta_csv}'")
    elif ruta_simulado.exists():
        ruta_csv = str(ruta_simulado)
        print(f"Usando dataset simulado: '{ruta_csv}'")
    else:
        print("ERROR: No se encontro ningun dataset.")
        print("Ejecuta primero 'python cargar_kaggle.py' o 'python generador_datos.py'")
        exit(1)

    df = pd.read_csv(ruta_csv)

    # Limpiar modelos anteriores
    for f in DIR_MODELOS.glob("*.pkl"):
        f.unlink()

    # Entrenar
    print("\n=== ENTRENAMIENTO ===")
    detector = DetectorAnomalias()
    resultados = detector.entrenar(df)

    for usuario, info in resultados.items():
        if usuario != "__global__":
            print(f"  {usuario}: {info['muestras_entrenamiento']} muestras -> {info['estado']}")

    # Guardar modelos
    detector.guardar()

    # Evaluacion sobre el dataset completo
    print("\n=== EVALUACION ===")
    y_real = df["es_anomalia"].values
    scores = []
    predicciones = []

    for _, fila in df.iterrows():
        medida = {col: fila[col] for col in COLS_SENALES}
        resultado = detector.predecir(fila["usuario"], medida)
        scores.append(resultado["score"])
        predicciones.append(1 if resultado["es_anomalia"] else 0)

    scores = np.array(scores)
    predicciones = np.array(predicciones)

    # Metricas
    auc = roc_auc_score(y_real, -scores)
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_real, predicciones, target_names=["Normal", "Anomalia"]))

    # Distribucion por nivel
    print("=== DISTRIBUCION POR NIVEL ===")
    niveles = {"normal": 0, "leve": 0, "moderada": 0, "grave": 0}
    niveles_anom = {"normal": 0, "leve": 0, "moderada": 0, "grave": 0}

    for _, fila in df.iterrows():
        medida = {col: fila[col] for col in COLS_SENALES}
        resultado = detector.predecir(fila["usuario"], medida)
        nivel = resultado["nivel_alerta"]
        niveles[nivel] += 1
        if fila["es_anomalia"] == 1:
            niveles_anom[nivel] += 1

    for nivel in ["normal", "leve", "moderada", "grave"]:
        total = niveles[nivel]
        anom = niveles_anom[nivel]
        norm = total - anom
        print(f"  {nivel:10s}: {total:4d} registros ({anom} anomalias, {norm} normales)")

    # Prueba rapida
    print("\n-- Prueba de prediccion --")
    medida_anomala = {
        "frecuencia_cardiaca": 140,
        "pasos_diarios": 200,
        "horas_sueno": 2,
        "calidad_sueno": 1,
        "nivel_estres": 10,
        "actividad_fisica": 5,
        "presion_sistolica": 180,
        "presion_diastolica": 110,
    }
    primer_usuario = list(detector.modelos.keys())[0] if detector.modelos else "user_1"
    resultado = detector.predecir(primer_usuario, medida_anomala)
    print(f"  Usuario: {primer_usuario}")
    print(f"  Resultado: anomalia={resultado['es_anomalia']}, "
          f"score={resultado['score']}, nivel={resultado['nivel_alerta']}, "
          f"modelo={resultado['modelo_usado']}")
