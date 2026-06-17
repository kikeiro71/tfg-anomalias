# config.py — Configuración central del sistema de detección de anomalías
# -----------------------------------------------------------------------
# Aquí se definen los rangos fisiológicos normales para cada señal
# biométrica y los umbrales que determinan el nivel de alerta.

# Señales biométricas monitorizadas y sus rangos normales de referencia
SENALES = {
    "frecuencia_cardiaca": {"min": 50, "max": 120, "unidad": "bpm"},
    "saturacion_oxigeno":  {"min": 90, "max": 100, "unidad": "%"},
    "temperatura":         {"min": 35.5, "max": 37.5, "unidad": "°C"},
    "pasos_hora":          {"min": 0, "max": 800, "unidad": "pasos/h"},
    "horas_sueno":         {"min": 4, "max": 10, "unidad": "horas"},
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
