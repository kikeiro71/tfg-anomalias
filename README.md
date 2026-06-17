# Sistema de Deteccion de Anomalias en Senales Biometricas

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Trabajo de Fin de Grado (TFG) que implementa un sistema de deteccion de anomalias en senales biometricas de dispositivos wearable, utilizando **Isolation Forest** con linea base individual por usuario y un modelo global de respaldo.

---

## Arquitectura

```
+-------------+         +-----------------+         +-----------+
|  Wearable / |  POST   |                 |  POST   |           |
|  Simulador  +-------->+   API Flask     +-------->+   n8n     |
|             |         |   :5000         |         |   :5678   |
+-------------+         +-------+---------+         +-----+-----+
                                |                         |
                         +------v--------+          +-----v-----+
                         |               |          |  Email    |
                         |  Isolation    |          |  Telegram |
                         |  Forest       |          |  SMS      |
                         |               |          +-----------+
                         |  - Global     |
                         |  - Individual |
                         +------+--------+
                                |
                         +------v--------+
                         |  Clasificar   |
                         |  nivel alerta |
                         |               |
                         |  Normal       |
                         |  Leve         |
                         |  Moderada     |
                         |  Grave        |
                         +---------------+
```

## Dataset

Este proyecto soporta dos fuentes de datos:

| Fuente | Registros | Descripcion |
|--------|-----------|-------------|
| [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) (Kaggle) | 400 | Datos reales de salud y estilo de vida |
| Generador simulado (`generador_datos.py`) | 1,000 | 20 usuarios con anomalias controladas al 5% |

## Senales monitorizadas

| Senal | Rango normal | Unidad |
|-------|-------------|--------|
| Frecuencia cardiaca | 50 - 100 | bpm |
| Pasos diarios | 2,000 - 12,000 | pasos |
| Horas de sueno | 4 - 10 | horas |
| Calidad de sueno | 4 - 10 | 1-10 |
| Nivel de estres | 1 - 10 | 1-10 |
| Actividad fisica | 20 - 90 | min/dia |
| Presion sistolica | 90 - 140 | mmHg |
| Presion diastolica | 60 - 90 | mmHg |

## Resultados del modelo

Metricas obtenidas tras calibrado empirico de umbrales (basados en percentiles p5/p2/p1 de los scores normales):

| Metrica | Modelo global | Modelos individuales |
|---------|--------------|---------------------|
| **ROC-AUC** | 0.95 | 0.91 |
| Precision (normal) | 98% | 98% |
| Recall (anomalias) | 72% | 74% |
| Tasa falsos positivos | 5.5% | 11.9% |

### Distribucion por nivel de alerta

| Nivel | Total | Anomalias reales | Normales |
|-------|-------|-------------------|----------|
| Normal | 847 | 14 | 833 |
| Leve | 64 | 7 | 57 |
| Moderada | 44 | 13 | 31 |
| **Grave** | **45** | **20** | 25 |

Los umbrales se calibran automaticamente a partir de la distribucion real de scores, no con valores arbitrarios.

## Niveles de alerta

| Nivel | Umbral (score) | Accion |
|-------|---------------|--------|
| Normal | > -0.52 | Sin accion necesaria |
| Leve | -0.52 a -0.54 | Registrar y vigilar |
| Moderada | -0.54 a -0.56 | Notificar al profesional sanitario |
| Grave | < -0.56 | Alerta urgente multicanal |

## Estructura del proyecto

```
tfg-anomalias/
├── config.py              # Configuracion (rangos, umbrales calibrados, parametros)
├── cargar_kaggle.py       # Carga y preprocesa el dataset real de Kaggle
├── generador_datos.py     # Generador de datos simulados (alternativa)
├── modelo_anomalias.py    # Isolation Forest: modelo global + individuales
├── visualizar.py          # Graficas con matplotlib
├── api.py                 # API REST en Flask + webhook n8n
├── flujo_n8n.md           # Documentacion del flujo de notificaciones
├── requirements.txt       # Dependencias
└── LICENSE
```

## Instalacion

```bash
git clone https://github.com/kikeiro71/tfg-anomalias.git
cd tfg-anomalias
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Uso rapido

```bash
# 1. Generar datos simulados (o usar cargar_kaggle.py con datos reales)
python generador_datos.py

# 2. Entrenar modelos y ver evaluacion
python modelo_anomalias.py

# 3. Generar graficas para la memoria
python visualizar.py

# 4. Arrancar la API
python api.py
```

## API

### `POST /api/medida`

Envia una medida biometrica y recibe la clasificacion.

**Request:**

```json
{
  "usuario": "user_1",
  "frecuencia_cardiaca": 145,
  "pasos_diarios": 300,
  "horas_sueno": 3,
  "calidad_sueno": 2,
  "nivel_estres": 9,
  "actividad_fisica": 5,
  "presion_sistolica": 185,
  "presion_diastolica": 115
}
```

**Response:**

```json
{
  "usuario": "user_1",
  "es_anomalia": true,
  "score": -0.6764,
  "nivel_alerta": "grave",
  "modelo_usado": "individual",
  "senales_fuera_de_rango": [
    {"senal": "frecuencia_cardiaca", "valor": 145.0, "tipo": "por_encima"},
    {"senal": "presion_sistolica", "valor": 185.0, "tipo": "por_encima"}
  ],
  "accion_recomendada": "Alerta urgente - contactar servicios de emergencia"
}
```

### `GET /api/estado`

Comprueba que la API y los modelos estan activos.

### `GET /api/umbrales`

Devuelve los umbrales de alerta configurados.

## Flujo de notificaciones

El sistema se integra con [n8n](https://n8n.io/) para automatizar notificaciones segun el nivel de alerta. La documentacion completa esta en [`flujo_n8n.md`](flujo_n8n.md).

```
Alerta leve      -->  Registro en base de datos
Alerta moderada  -->  Email al profesional sanitario
Alerta grave     -->  Email + Telegram + SMS
```

## Tecnologias

| Tecnologia | Uso |
|-----------|-----|
| Python 3.12 | Lenguaje principal |
| scikit-learn | Isolation Forest |
| pandas | Manipulacion de datos |
| matplotlib | Visualizacion |
| Flask | API REST |
| n8n | Automatizacion de notificaciones |

## Licencia

[MIT](LICENSE)
