# Sistema de Detección de Anomalías en Señales Biométricas

Proyecto de Trabajo de Fin de Grado (TFG) que implementa un sistema de detección de anomalías en señales biométricas de dispositivos wearable, utilizando Isolation Forest con línea base individual por usuario.

## Dataset

Este proyecto utiliza el dataset público **[Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)** de Kaggle (400 registros, 13 variables, datos reales de salud y estilo de vida).

También incluye un generador de datos simulados para ampliar el volumen y demostrar el flujo completo.

## Señales monitorizadas

| Señal | Rango normal | Unidad |
|-------|-------------|--------|
| Frecuencia cardíaca | 50 – 100 | bpm |
| Pasos diarios | 2,000 – 12,000 | pasos |
| Horas de sueño | 4 – 10 | horas |
| Calidad de sueño | 4 – 10 | 1-10 |
| Nivel de estrés | 1 – 10 | 1-10 |
| Actividad física | 20 – 90 | min/día |
| Presión sistólica | 90 – 140 | mmHg |
| Presión diastólica | 60 – 90 | mmHg |

## Niveles de alerta

- **Leve** — Registro para vigilancia posterior
- **Moderada** — Notificación al profesional sanitario
- **Grave** — Alerta urgente multicanal (email + Telegram + SMS)

## Estructura del proyecto

```
├── config.py              # Configuración (rangos, umbrales, parámetros del modelo)
├── cargar_kaggle.py       # Carga y preprocesa el dataset real de Kaggle
├── generador_datos.py     # Generador de datos simulados (alternativa)
├── modelo_anomalias.py    # Modelo Isolation Forest (entrenamiento y predicción)
├── visualizar.py          # Gráficas con matplotlib
├── api.py                 # API REST en Flask
├── flujo_n8n.md           # Documentación del flujo de notificaciones en n8n
└── requirements.txt       # Dependencias
```

## Instalación

```bash
git clone https://github.com/kikeiro71/tfg-anomalias.git
cd tfg-anomalias
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Uso con datos reales (Kaggle)

1. Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
2. Coloca el archivo `Sleep_health_and_lifestyle_dataset.csv` en la carpeta `datos/`
3. Ejecuta:

```bash
# Cargar y preprocesar datos reales
python cargar_kaggle.py

# Entrenar modelos
python modelo_anomalias.py

# Generar gráficas
python visualizar.py

# Arrancar la API
python api.py
```

## Uso con datos simulados (sin Kaggle)

```bash
python generador_datos.py
python modelo_anomalias.py
python visualizar.py
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
    "frecuencia_cardiaca": 140,
    "pasos_diarios": 200,
    "horas_sueno": 2,
    "calidad_sueno": 1,
    "nivel_estres": 10,
    "actividad_fisica": 5,
    "presion_sistolica": 180,
    "presion_diastolica": 110
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
    {"senal": "frecuencia_cardiaca", "valor": 140, "tipo": "por_encima"},
    {"senal": "presion_sistolica", "valor": 180, "tipo": "por_encima"}
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
