# System Catalog

## Smart Driver & Vehicle Proactive Safety System — living reference

**Status:** Reflects the code as of 2026-08-19 (post direct-to-Firebase
pivot). This is a reference, not a narrative — for the story of *why* things
changed, see [`../smart-driver-docs/iot-connection-gap-analysis.md`](../smart-driver-docs/iot-connection-gap-analysis.md);
for how to run the whole chain end to end, see [`full-walkthrough.md`](full-walkthrough.md)
(same folder); for standing up the Firebase project from zero, see
[`firebase-setup-guide.md`](firebase-setup-guide.md) (same folder). See
[`docs-index.md`](docs-index.md) for the full document map.

---

## 1. Architecture

```
ESP32-CAM ──JPEG/WebSocket (local hotspot)──> AI server (webserver/Backend/server.py)
                                                          │ verdict JSON
                                                          ▼
                                                     ESP32-CAM
                                                          │
                                                     ESP-NOW (2.4GHz, no AP)
                                                          ▼
                                                  ESP32 Main Hub
                                        (GPS + ultrasonic + risk fusion + LEDs)
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    │ 4G / EC20 — LIVE PATH                      │ 4G / EC20 — occasional
                                    ▼                                            ▼
                          Firebase RTDB directly                    Backend: POST /iot/device-token
                    (PUT safetyData/live, PATCH status)              (mints the token used on the left)
                                    │
                                    ▼
                              Mobile app (frontend/, "touruni-mobile")
                     QR scan → POST /devices/register → GET /devices/{id}/firebase-token
                     → signInWithCustomToken → onValue(/devices/{id}/safetyData/live, /status)
```

**The backend (`backend/clean_run`) is no longer in the telemetry hot path.**
It still does everything else:
- Device registration/claim (QR flow) — `POST /devices/register`
- Issuing the *mobile app's* Firebase read token — `GET /devices/{id}/firebase-token`
- Issuing the *device's* Firebase write token — `POST /iot/device-token` (new)
- Alert-event history (Mongo) — `POST`/`GET /iot/alert-events`, now called by
  the mobile app itself (`frontend/src/context/IoTContext.tsx`) on every
  tier increase, since nothing server-side sees telemetry to log it from
  anymore
- Trip sessions — `POST /iot/trips/start`, `POST /iot/trips/{id}/end`
- `POST /iot/telemetry` (Option B, backend-relay) still exists and works —
  documented fallback, not called by the Hub by default. Firmware's
  `sendToBackendCellular()` is kept, unused, as the quick way back to it.

The local AI server (`webserver/Backend/server.py`) and the ESP-NOW link
between the two ESP32 boards are unchanged by this pivot — only what
happens after the Main Hub has assembled a telemetry reading changed.

---

## 2. Main Hub — GPIO table

`Arduino/esp 32 main/sketch_may9b_copy_20260509095030/sketch_may9b_copy_20260509095030.ino`

| GPIO | Name | Direction | Role |
|---|---|---|---|
| 2 | `LED_PIN` | OUTPUT | **Red** — alert tier (existing) |
| 4 | `LED2_PIN` | OUTPUT | **Green** — system health (new) |
| 5 | `TRIG_PIN` | OUTPUT | Ultrasonic trigger |
| 16 | `GPS_RX_PIN` | UART2 RX | GPS module |
| 17 | `GPS_TX_PIN` | UART2 TX | GPS module |
| 18 | `ECHO_PIN` | INPUT | Ultrasonic echo |
| 19 | `VIB_MOTOR_PIN` | OUTPUT | Vibration motor (via NPN driver) |
| 23 | `BUZZER_PIN` | OUTPUT | Buzzer |
| 25 | `MODEM_PWRKEY_PIN` | OUTPUT | EC20 power key |
| 26 | `MODEM_TX_PIN` | UART1 TX | EC20 RXD |
| 27 | `MODEM_RX_PIN` | UART1 RX | EC20 TXD |
| 32 | `MODEM_STATUS_PIN` | INPUT | EC20 status monitor |
| 33 | `MODEM_RESET_PIN` | OUTPUT | EC20 hard-reset |

No conflicts, no strapping pins (0/2/12/15) used for anything new, no
input-only pins (34/35/36/39) used for an output.

---

## 3. LED state tables

**LED1 / RED (GPIO2)** — alert tier, driven by `updateAlertPattern()`,
unchanged in meaning from the original firmware:

| Pattern | Tier | Condition |
|---|---|---|
| OFF | 0 | No risk |
| Single ~100ms flash, once/second | 1 (LOW) | `riskScore >= 0.50` or `drowsyLevel >= 2` |
| Triple pulse (150on/100off), repeating | 2 (MEDIUM) | `riskScore >= 0.70` or `drowsyLevel >= 3` |
| Continuous flash (200on/100off) + vibration | 3 (HIGH) | `riskScore >= 0.85` or `drowsyLevel >= 4` |

Tier computed by `computeAlertTier()`, shared verbatim with the value sent
to Firebase as `alertTier` — see §4's payload shape. This is the same table
as `normalizer.py`'s `_TIER_THRESHOLDS`; the two are required to match, per
that file's own comment.

**LED2 / GREEN (GPIO4)** — system health, driven by `updateHealthLed()`,
priority-ordered (worst wins):

| Priority | Pattern | Condition |
|---|---|---|
| 1 (worst) | **OFF** | `!modem.isGprsConnected()` OR last Firebase write failed |
| 2 | **Slow blink** (~150ms pulse/2s) | link+auth OK, but camera/AI data stale (`!driverVisible()`) or no GPS fix (`!gpsFixed`) |
| 3 (best) | **Solid ON** | link+auth OK, camera/AI fresh, GPS fixed |

Before `setup()` evaluates any of these once, GREEN defaults OFF.

---

## 4. Telemetry payload — what the Hub writes to Firebase

`PUT /devices/{deviceId}/safetyData/live.json?auth=<idToken>` (full
overwrite each cycle):

```json
{
  "riskScore": 0.42,
  "alertTier": 1,
  "driver": {
    "drowsyLevel": 2,
    "confidence": 0.63,
    "eyeStatus": "open",
    "yawningStatus": "normal",
    "earScore": 0.0,
    "driverVisible": true
  },
  "vehicle": { "distanceCm": 180.0, "ttcSeconds": 999.0 },
  "gps": { "latitude": 6.9271, "longitude": 79.8612, "speedKmh": 32.0, "satellites": 7, "fixed": true },
  "timestampMs": 1755600000000,
  "deviceUptimeMs": 184532,
  "sequenceNum": 41
}
```

`PATCH /devices/{deviceId}/status.json?auth=<idToken>`:

```json
{ "online": true, "lastSeenAt": { ".sv": "timestamp" }, "linkType": "cellular", "csq": 21 }
```

Field-for-field, this is the same shape `backend/clean_run/iot/normalizer.py`
used to produce server-side (Option B) — the Hub now computes it itself:
`riskScore` is 0–1 (not the internal 0–100 scale), `eyeStatus`/
`yawningStatus` are strings gated on `driverVisible`, `timestampMs` is a
real epoch (sourced from cellular network time — see §6), `alertTier` and
`ttcSeconds` are derived on-device via `computeAlertTier()` /
`computeTtcSeconds()`.

**Not written by the direct path** (documented gap, not silently dropped):
`/safetyData/snapshots` (throttled history) and Firebase-side `/alerts` —
those were Option-B/Mongo-sequence-counter features. Alert *history* itself
still works: the mobile app calls `POST /iot/alert-events` itself on every
tier increase (`frontend/src/context/IoTContext.tsx`).

---

## 5. Backend routes touched by this pipeline

| Route | Auth | Caller | Purpose |
|---|---|---|---|
| `POST /iot/device-token` | `X-Device-Id`/`X-Device-Secret` | Main Hub | Mint a short-lived Firebase custom token for the device |
| `POST /iot/telemetry` | `X-Device-Id`/`X-Device-Secret` | *(dormant — Option B fallback)* | Legacy backend-relay ingest, still functional |
| `POST /devices/register` | Bearer JWT | Mobile app | Claim a device via scanned QR (`device_id`, `label`, `registration_secret`) |
| `GET /devices/{id}/firebase-token` | Bearer JWT | Mobile app | Read-scoped Firebase token — one token covers every device the user owns |
| `POST /iot/alert-events` | Bearer JWT | Mobile app | Persist an alert event on a tier increase (now the *only* thing that does) |
| `GET /iot/alert-events` | Bearer JWT | Mobile app | Paginated alert history |
| `POST/GET /iot/trips/*` | Bearer JWT | Mobile app | Trip session lifecycle |

---

## 6. Firmware auth/time chain (device → Firebase)

1. `fetchDeviceCustomToken()` → `POST /iot/device-token` (device_secret auth)
   → `{firebase_token, expires_in}`.
2. `exchangeForIdToken()` → Google's `POST
   https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=<FIREBASE_API_KEY>`
   → `{idToken, expiresIn}`.
3. `idToken` used as `?auth=` on every RTDB REST call. Refreshed by
   `ensureFirebaseAuth()` whenever it's within 5 minutes of expiry.
4. `syncEpochTime()` — real time-of-day from `AT+CCLK?` (TinyGSM's
   `getNetworkTime()`), not GPS. GPS may never acquire a fix indoors (a
   plausible demo environment), and `firebase_rules.json` rejects any
   `timestampMs` below a real-epoch floor — a GPS-derived clock would
   silently block every write in exactly that environment. Cellular signal
   is already required for this whole path, so network time adds no new
   failure mode. Synced once after registration, not every cycle.

See `firebase-setup-guide.md` for the exact `curl` equivalent of each hop.

---

## 7. `SensorData` — ESP-NOW struct (CAM → Main Hub)

```cpp
typedef struct {
  int drowsyLevel;
  float confidence;
  int eyeStatus;
  int yawningStatus;
  int faceValid;
  unsigned long timestamp;
} SensorData; // 24 bytes, static_assert'd identical on both boards
```

Channel: `ESPNOW_CHANNEL` (Hub, hardcoded) must match the CAM's actual
`WiFi.channel()` (logged at CAM boot as `"WiFi channel: N"`). No automatic
negotiation — the Hub logs a warning if zero packets arrive within 15s of
boot, naming the current `ESPNOW_CHANNEL` as the likely culprit.
