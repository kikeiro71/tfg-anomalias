# Sistema de Detección de Anomalías en Señales Biométricas

Proyecto de Trabajo de Fin de Grado (TFG) que implementa un sistema de detección de anomalías en señales biométricas de dispositivos wearable, utilizando Isolation Forest con línea base individual por usuario.

## Señales monitorizadas

| Señal | Rango normal | Unidad |
|-------|-------------|--------|
| Frecuencia cardíaca | 50 – 120 | bpm |
| Saturación de oxígeno | 90 – 100 | % |
| Temperatura corporal | 35.5 – 37.5 | °C |
| Pasos por hora | 0 – 800 | pasos/h |
| Horas de sueño | 4 – 10 | horas |

## Niveles de alerta

- **Leve** — Registro para vigilancia posterior
- **Moderada** — Notificación al profesional sanitario
- **Grave** — Alerta urgente multicanal (email + Telegram + SMS)

## Estructura del proyecto

```
├── config.py              # Configuración (rangos, umbrales, parámetros del modelo)
├── generador_datos.py     # Generador de datos simulados con anomalías controladas
├── modelo_anomalias.py    # Modelo Isolation Forest (entrenamiento y predicción)
├── visualizar.py          # Gráficas con matplotlib
├── api.py                 # API REST en Flask
├── flujo_n8n.md           # Documentación del flujo de notificaciones en n8n
└── requirements.txt       # Dependencias
```

## Instalación

```bash
git clone https://github.com/tu-usuario/tfg-anomalias.git
cd tfg-anomalias
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Uso

```bash
# 1. Generar datos simulados
python generador_datos.py

# 2. Entrenar modelos (uno por usuario)
python modelo_anomalias.py

# 3. Generar gráficas
python visualizar.py

# 4. Arrancar la API
python api.py
```

## API

### `POST /api/medida`

Envía una medida biométrica y recibe la clasificación.

```bash
curl -X POST http://localhost:5000/api/medida \
  -H "Content-Type: application/json" \
  -d '{
    "usuario": "user_1",
    "frecuencia_cardiaca": 160,
    "saturacion_oxigeno": 85,
    "temperatura": 39.5,
    "pasos_hora": 0,
    "horas_sueno": 2
  }'
```

Respuesta:

```json
{
  "usuario": "user_1",
  "es_anomalia": true,
  "score": -0.6821,
  "nivel_alerta": "moderada",
  "senales_fuera_de_rango": [
    {"senal": "frecuencia_cardiaca", "valor": 160, "tipo": "por_encima"},
    {"senal": "saturacion_oxigeno", "valor": 85, "tipo": "por_debajo"}
  ],
  "accion_recomendada": "Notificar al profesional sanitario"
}
```

### `GET /api/estado`

Comprueba que la API y los modelos están activos.

### `GET /api/umbrales`

Devuelve los umbrales de alerta configurados.

## Tecnologías

- **Python 3.10+**
- **scikit-learn** — Isolation Forest para detección de anomalías
- **pandas** — Manipulación de datos
- **matplotlib** — Visualización
- **Flask** — API REST
- **n8n** — Automatización de notificaciones

## Flujo de notificaciones

El flujo de notificaciones en n8n está documentado en [`flujo_n8n.md`](flujo_n8n.md), incluyendo arquitectura, nodos, configuración y diagramas.

## Licencia

Este proyecto forma parte de un Trabajo de Fin de Grado.
