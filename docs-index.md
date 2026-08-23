# Smart Driver Safety — Design Documents

Planning and analysis for the **Smart Driver & Vehicle Proactive Safety System**
(integrated driver drowsiness detection + forward collision warning).

Single source of truth for these documents — they previously existed as duplicate
copies in two other folders.

**Note:** `system-catalog.md`, `full-walkthrough.md`, and
`firebase-setup-guide.md` live in this folder, alongside the AI server code.
The rest live in [`../smart-driver-docs/`](../smart-driver-docs/) — the
links below point there.

| Document | What it covers |
|---|---|
| [system-catalog.md](system-catalog.md) | **Start here.** Living reference — architecture, GPIO table, LED state tables, telemetry payload shape, backend routes, the Firebase auth chain, the ESP-NOW struct. Reflects the code as it stands today. |
| [full-walkthrough.md](full-walkthrough.md) | Step-by-step bench test / demo rehearsal script — the 20-step chain end to end, plus five failure scenarios and expected recovery behavior. |
| [firebase-setup-guide.md](firebase-setup-guide.md) | From-zero Firebase project setup — console steps, security rules, service account, the two values the firmware needs, and a curl-by-curl breakdown of the device auth chain. |
| [iot-connection-gap-analysis.md](../smart-driver-docs/iot-connection-gap-analysis.md) | History — the breaks between firmware, backend, Firebase and the app that were found and fixed in the backend-relay era; see its 2026-08-19 amendment note for what's since changed. |
| [4g-ec20-integration-plan.md](../smart-driver-docs/4g-ec20-integration-plan.md) | Replacing WiFi with the Quectel EC20 4G modem — wiring, power budget, electrical constraints, phased rollout. |
| [iot-device-implementation-plan.md](../smart-driver-docs/iot-device-implementation-plan.md) | Device firmware — sensor fusion, risk scoring, alert tiers, state machine. |
| [mobile-iot-integration-plan.md](../smart-driver-docs/mobile-iot-integration-plan.md) | Mobile app — screens, Firebase listeners, device registration, trip sessions. |
| [eraser-ai-diagram-prompts.md](../smart-driver-docs/eraser-ai-diagram-prompts.md) | Prompts for regenerating the architecture diagrams. |

This folder's own [README.md](README.md) covers the AI server code itself
(models, training scripts, dashboard) — this file is the design-doc index,
kept separate so the two don't get tangled.

## Related repositories

| Repo | Contents |
|---|---|
| `smart-driver-firmware` | ESP32 main hub + ESP32-CAM sketches |
| `smart-driver-ai-server` / `webserver/Backend` | Drowsiness inference server (in-vehicle laptop/Pi) — also where this index and the three reference docs live |
| `tourunipp2` | FastAPI backend — `clean_run/iot/` |
| `touruni-mobile` | Expo/React Native app — `src/` IoT screens |

## System at a glance

```
ESP32-CAM ──JPEG/WebSocket──> AI server ──verdict──> ESP32-CAM
                                                          │
                                                     ESP-NOW
                                                          ▼
                                                  ESP32 Main Hub
                                        (GPS + ultrasonic + risk fusion)
                                                          │
                                                    4G / EC20
                                                          │
                                    ┌─────────────────────┴──────────────┐
                                    ▼                                    ▼
                          Firebase RTDB (direct write —          FastAPI backend
                          the live telemetry path)          (device tokens, QR claim,
                                    │                         alert history, trips)
                                    ▼                                    │
                                Mobile app ◄───────────────────────  MongoDB
```

Inference happens in the vehicle and needs no internet. Only the result — a few
hundred bytes — leaves, over cellular, straight to Firebase. The backend
mints the device's write token and handles everything that isn't telemetry
itself — see `system-catalog.md` for the current architecture in detail.
