# config.py — Configuracion central del sistema de deteccion de anomalias
# -----------------------------------------------------------------------
# Aqui se definen los rangos fisiologicos normales para cada senal
# biometrica y los umbrales que determinan el nivel de alerta.

# ── Senales biometricas monitorizadas ────────────────────────────────────
# Adaptadas al dataset "Sleep Health and Lifestyle" de Kaggle
# https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

SENALES = {
    "frecuencia_cardiaca":   {"min": 50, "max": 100, "unidad": "bpm"},
    "pasos_diarios":         {"min": 2000, "max": 12000, "unidad": "pasos"},
    "horas_sueno":           {"min": 4, "max": 10, "unidad": "horas"},
    "calidad_sueno":         {"min": 4, "max": 10, "unidad": "1-10"},
    "nivel_estres":          {"min": 1, "max": 10, "unidad": "1-10"},
    "actividad_fisica":      {"min": 20, "max": 90, "unidad": "min/dia"},
    "presion_sistolica":     {"min": 90, "max": 140, "unidad": "mmHg"},
    "presion_diastolica":    {"min": 60, "max": 90, "unidad": "mmHg"},
}

# Mapeo de columnas del CSV original de Kaggle a nombres internos
MAPEO_COLUMNAS_KAGGLE = {
    "Heart Rate":              "frecuencia_cardiaca",
    "Daily Steps":             "pasos_diarios",
    "Sleep Duration":          "horas_sueno",
    "Quality of Sleep":        "calidad_sueno",
    "Stress Level":            "nivel_estres",
    "Physical Activity Level": "actividad_fisica",
    "Person ID":               "usuario",
}

# Umbrales para clasificar la gravedad de una anomalia.
# Calibrados empiricamente a partir de los percentiles de scores
# del Isolation Forest sobre datos normales:
#   - leve:     percentil 5  de los scores normales
#   - moderada: percentil 2  de los scores normales
#   - grave:    percentil 1  de los scores normales
# Un score por debajo de estos valores indica una medida cada vez
# mas alejada del comportamiento normal aprendido.
UMBRALES_ALERTA = {
    "leve":     -0.52,   # p5 — el 5% mas extremo de lo normal
    "moderada": -0.54,   # p2 — solo el 2% de normales llega aqui
    "grave":    -0.56,   # p1 — practicamente nunca se ve en normales
}

# Parametros del modelo Isolation Forest
MODELO_CONFIG = {
    "n_estimators": 150,         # Numero de arboles en el bosque
    "contamination": 0.05,       # Proporcion esperada de anomalias (5%)
    "random_state": 42,          # Semilla para reproducibilidad
}

# Minimo de muestras normales necesarias para entrenar un modelo por usuario.
# Si un usuario tiene menos muestras, se usa el modelo global.
MIN_MUESTRAS_USUARIO = 20
