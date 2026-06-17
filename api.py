# api.py — API REST mínima en Flask para detección de anomalías
# --------------------------------------------------------------
# Endpoints:
#   POST /api/medida       → Envía una medida y recibe la clasificación
#   GET  /api/estado       → Comprueba que la API y los modelos están listos
#   GET  /api/umbrales     → Devuelve los umbrales de alerta configurados

from flask import Flask, request, jsonify
from modelo_anomalias import DetectorAnomalias, COLS_SENALES
from config import UMBRALES_ALERTA, SENALES

app = Flask(__name__)

# Cargar modelos al arrancar la API
detector = DetectorAnomalias()
detector.cargar()


# ── POST /api/medida ─────────────────────────────────────────────────────

@app.route("/api/medida", methods=["POST"])
def recibir_medida():
    """
    Recibe una medida biométrica y devuelve el resultado del análisis.

    Body JSON esperado:
    {
        "usuario": "user_1",
        "frecuencia_cardiaca": 130,
        "saturacion_oxigeno": 88,
        "temperatura": 39.2,
        "pasos_hora": 50,
        "horas_sueno": 3
    }

    Respuesta:
    {
        "usuario": "user_1",
        "es_anomalia": true,
        "score": -0.4523,
        "nivel_alerta": "moderada",
        "senales_fuera_de_rango": [...],
        "accion_recomendada": "Notificar al profesional sanitario"
    }
    """
    datos = request.get_json()

    if not datos:
        return jsonify({"error": "Se requiere un body JSON"}), 400

    usuario = datos.get("usuario")
    if not usuario:
        return jsonify({"error": "El campo 'usuario' es obligatorio"}), 400

    # Extraer las señales del body
    medida = {}
    campos_faltantes = []
    for col in COLS_SENALES:
        if col in datos:
            medida[col] = float(datos[col])
        else:
            campos_faltantes.append(col)

    if campos_faltantes:
        return jsonify({
            "error": f"Faltan campos obligatorios: {campos_faltantes}"
        }), 400

    # Ejecutar predicción
    resultado = detector.predecir(usuario, medida)

    if "error" in resultado:
        return jsonify(resultado), 404

    # Añadir acción recomendada según el nivel
    acciones = {
        "normal":   "Sin acción necesaria",
        "leve":     "Registrar y vigilar en las próximas horas",
        "moderada": "Notificar al profesional sanitario",
        "grave":    "Alerta urgente — contactar servicios de emergencia",
    }

    respuesta = {
        "usuario": usuario,
        **resultado,
        "accion_recomendada": acciones.get(resultado["nivel_alerta"], ""),
    }

    return jsonify(respuesta)


# ── GET /api/estado ──────────────────────────────────────────────────────

@app.route("/api/estado", methods=["GET"])
def estado():
    """Comprueba que la API está activa y los modelos cargados."""
    return jsonify({
        "estado": "activo",
        "modelos_cargados": list(detector.modelos.keys()),
        "senales_monitorizadas": list(SENALES.keys()),
    })


# ── GET /api/umbrales ───────────────────────────────────────────────────

@app.route("/api/umbrales", methods=["GET"])
def umbrales():
    """Devuelve los umbrales de alerta configurados."""
    return jsonify(UMBRALES_ALERTA)


# ── Arranque ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Iniciando API de detección de anomalías...")
    print(f"Modelos disponibles: {list(detector.modelos.keys())}")
    print("Endpoints:")
    print("  POST /api/medida   — Evaluar una medida")
    print("  GET  /api/estado   — Estado de la API")
    print("  GET  /api/umbrales — Umbrales de alerta")
    app.run(host="0.0.0.0", port=5000, debug=True)
