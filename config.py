# config.py — Configuración central del sistema de detección de anomalías
# -----------------------------------------------------------------------
# Aquí se definen los rangos fisiológicos normales para cada señal
# biométrica y los umbrales que determinan el nivel de alerta.

# ── Señales biométricas monitorizadas ────────────────────────────────────
# Adaptadas al dataset "Sleep Health and Lifestyle" de Kaggle
# https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

SENALES = {
    "frecuencia_cardiaca":   {"min": 50, "max": 100, "unidad": "bpm"},
    "pasos_diarios":         {"min": 2000, "max": 12000, "unidad": "pasos"},
    "horas_sueno":           {"min": 4, "max": 10, "unidad": "horas"},
    "calidad_sueno":         {"min": 4, "max": 10, "unidad": "1-10"},
    "nivel_estres":          {"min": 1, "max": 10, "unidad": "1-10"},
    "actividad_fisica":      {"min": 20, "max": 90, "unidad": "min/día"},
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

# Umbrales para clasificar la gravedad de una anomalía.
# El score del Isolation Forest va de -1 (muy anómalo) a +1 (normal).
# Cuanto más negativo, más anómala es la medida.
UMBRALES_ALERTA = {
    "leve":     -0.3,   # Score entre -0.3 y -0.5
    "moderada": -0.5,   # Score entre -0.5 y -0.7
    "grave":    -0.7,   # Score por debajo de -0.7
}

# Parámetros del modelo Isolation Forest
MODELO_CONFIG = {
    "n_estimators": 150,         # Número de árboles en el bosque
    "contamination": 0.05,       # Proporción esperada de anomalías (5%)
    "random_state": 42,          # Semilla para reproducibilidad
}
