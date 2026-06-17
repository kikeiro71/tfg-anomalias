# Flujo de notificaciones en n8n

## Descripción general

El sistema de notificaciones conecta la API Flask con los canales de
comunicación (email, Telegram, etc.) a través de n8n, una herramienta
de automatización visual de flujos de trabajo.

## Arquitectura del flujo

```
┌──────────────┐     POST /api/medida     ┌───────────────┐
│  Dispositivo │ ──────────────────────►   │   API Flask    │
│  wearable    │                           │  (puerto 5000) │
└──────────────┘                           └───────┬───────┘
                                                   │
                                            JSON con resultado
                                                   │
                                                   ▼
                                           ┌───────────────┐
                                           │  Webhook n8n   │
                                           │  /webhook/alerta│
                                           └───────┬───────┘
                                                   │
                                          ┌────────┼────────┐
                                          │        │        │
                                          ▼        ▼        ▼
                                       Leve    Moderada   Grave
                                          │        │        │
                                          ▼        ▼        ▼
                                       Log DB   Email    Telegram
                                                         + Email
                                                         + SMS
```

## Nodos del flujo en n8n

### 1. Webhook de entrada (Trigger)

- **Tipo**: Webhook
- **Método**: POST
- **Ruta**: `/webhook/alerta`
- **Body esperado** (JSON que envía la API Flask):

```json
{
  "usuario": "user_1",
  "es_anomalia": true,
  "score": -0.5821,
  "nivel_alerta": "moderada",
  "senales_fuera_de_rango": [
    {
      "senal": "frecuencia_cardiaca",
      "valor": 145,
      "rango_normal": "50–120 bpm",
      "tipo": "por_encima"
    }
  ],
  "accion_recomendada": "Notificar al profesional sanitario",
  "timestamp": "2026-06-17T10:30:00"
}
```

### 2. Nodo Switch — Clasificar por nivel

- **Tipo**: Switch
- **Campo evaluado**: `{{ $json.nivel_alerta }}`
- **Ramas**:
  - `"leve"` → Rama 1
  - `"moderada"` → Rama 2
  - `"grave"` → Rama 3

### 3. Rama LEVE — Registrar en base de datos

- **Nodo**: Google Sheets / Postgres / MySQL (según la infraestructura)
- **Acción**: Insertar fila con los datos de la alerta
- **Campos**: usuario, timestamp, nivel_alerta, score, señales afectadas
- **Motivo**: Las alertas leves se registran para análisis posterior
  pero no generan notificación inmediata.

### 4. Rama MODERADA — Email al profesional

- **Nodo 1**: Registrar en base de datos (igual que rama leve)
- **Nodo 2**: Send Email (Gmail / SMTP)
  - **Destinatario**: Profesional sanitario asignado al usuario
  - **Asunto**: `⚠️ Alerta moderada — {{ $json.usuario }}`
  - **Cuerpo**:

```
Se ha detectado una anomalía moderada en las señales biométricas.

Usuario: {{ $json.usuario }}
Hora: {{ $json.timestamp }}
Score: {{ $json.score }}

Señales fuera de rango:
{{ $json.senales_fuera_de_rango }}

Acción recomendada: {{ $json.accion_recomendada }}
```

### 5. Rama GRAVE — Notificación urgente multicanal

- **Nodo 1**: Registrar en base de datos
- **Nodo 2**: Send Email (prioridad alta)
  - **Asunto**: `🚨 ALERTA GRAVE — {{ $json.usuario }}`
- **Nodo 3**: Telegram Bot
  - **Chat ID**: Grupo de emergencias o chat del profesional
  - **Mensaje**: Resumen de la alerta con enlace al panel
- **Nodo 4** (opcional): SMS vía Twilio
  - Solo para alertas graves donde se requiera confirmación de lectura

## Cómo conectar la API Flask con n8n

En la API Flask, tras detectar una anomalía, se envía el resultado al
webhook de n8n. Este código se añadiría en `api.py` dentro del
endpoint `/api/medida`, después de obtener el resultado:

```python
import requests

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/alerta"

def notificar_n8n(usuario, resultado, timestamp):
    """Envía la alerta a n8n si es una anomalía."""
    if resultado["es_anomalia"]:
        payload = {
            "usuario": usuario,
            **resultado,
            "timestamp": timestamp,
        }
        try:
            requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
        except requests.RequestException as e:
            print(f"Error al notificar a n8n: {e}")
```

## Configuración de n8n

### Instalación rápida

```bash
# Instalar n8n globalmente con npm
npm install -g n8n

# Arrancar n8n
n8n start
# Acceder a http://localhost:5678
```

### Importar el flujo

1. Abrir n8n en el navegador (`http://localhost:5678`)
2. Crear un nuevo flujo de trabajo
3. Añadir los nodos en el orden descrito arriba
4. Configurar las credenciales de email/Telegram en n8n
5. Activar el flujo

### Credenciales necesarias

| Servicio  | Tipo de credencial         | Dónde configurar                |
|-----------|----------------------------|---------------------------------|
| Gmail     | OAuth2 o App Password      | n8n → Credentials → Gmail       |
| Telegram  | Bot Token (via BotFather)  | n8n → Credentials → Telegram    |
| Twilio    | Account SID + Auth Token   | n8n → Credentials → Twilio      |
| BD        | Conexión Postgres/MySQL    | n8n → Credentials → según motor |

## Diagrama de secuencia

```
Wearable          API Flask           n8n               Email/Telegram
   │                  │                 │                     │
   │  POST /medida    │                 │                     │
   │─────────────────►│                 │                     │
   │                  │  analizar()     │                     │
   │                  │────────┐        │                     │
   │                  │◄───────┘        │                     │
   │                  │                 │                     │
   │                  │  POST /webhook  │                     │
   │                  │────────────────►│                     │
   │                  │                 │  switch(nivel)      │
   │                  │                 │────────┐            │
   │                  │                 │◄───────┘            │
   │                  │                 │                     │
   │                  │                 │  enviar alerta      │
   │  respuesta JSON  │                 │────────────────────►│
   │◄─────────────────│                 │                     │
   │                  │                 │                     │
```

## Notas para la memoria del TFG

- n8n es open-source y self-hosted, lo que cumple con los requisitos
  de privacidad de datos biométricos (no se envían a terceros).
- El flujo es extensible: se pueden añadir nodos para dashboards,
  almacenamiento en InfluxDB, o integración con sistemas hospitalarios.
- La separación API ↔ n8n permite modificar las reglas de notificación
  sin tocar el código del modelo de detección.
