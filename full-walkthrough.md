# Full System Walkthrough

## Bench test script and demo rehearsal script — one document, two uses

**Companion to:** [`system-catalog.md`](system-catalog.md) (reference, same
folder), [`../smart-driver-docs/iot-connection-gap-analysis.md`](../smart-driver-docs/iot-connection-gap-analysis.md)
(history), [`firebase-setup-guide.md`](firebase-setup-guide.md) (Firebase
project setup, same folder).

Each step below names what to check and the exact Serial/UI output to expect.
Run it in order once on the bench before a demo; the same sequence works as
a live rehearsal script.

---

## Part 1 — the 20-step chain

**1. ESP32-CAM boots.** Serial: `📡 ESP32-CAM MAC Address: ...` then
`✅ Camera ready!`.

**2. Camera initializes.** No `❌ Camera init failed` line. If it fails,
check the pin table in `system-catalog.md` §2 (camera board, not the Main
Hub) and PSRAM availability.

**3. Local AI server connects.** CAM Serial: `✅ WebSocket connected!` plus
3 fast LED flashes. Server side (`webserver/Backend/server.py`), console
shows a new client on the `:5002` WebSocket.

**4. Camera frame captured.** Every 200ms (`FRAME_INTERVAL_MS`), a JPEG
frame goes out over `webSocket.sendBIN()`. No `❌ Camera capture failed`.

**5. AI model processes the frame.** Server console/preview page
(`http://<server-ip>:5001/preview`) shows an annotated frame with face
bounding boxes.

**6. AI result returned.** Server sends `{"faceValid":1,"reason":"ok",
"drowsyLevel":N,"confidence":0.NN,"eyeStatus":0|1,"yawningStatus":0|1}` as a
WebSocket TEXT frame. `drowsyLevel` should now visibly range 0–4 across a
sleepy→awake test (this is the graded-scoring fix — confirm it's not stuck
at only 0 or 1).

**7. ESP32-CAM receives/parses the result.** CAM Serial:
`📊 face=1 (ok) drowsy=N conf=0.NN`.

**8. ESP32-CAM sends via ESP-NOW.** No explicit success log per-packet
(only failures print), but `sendSuccessCount` climbs — visible in the
periodic `📊 ESP-NOW stats` line every 100 frames.

**9. Main Hub receives the result.** Hub Serial: the `📥 ESP-NOW DATA
RECEIVED!` box, with matching `drowsyLevel`/`confidence`/etc. If nothing
ever arrives, see Part 2 → Test B.

**10. Main Hub reads sensors.** Ultrasonic distance updates every 500ms; if
the sensor is unplugged, watch for `⚠️ Ultrasonic sensor: no echo (timeout)`
— this is the fault flag, not a genuine 400cm clear reading.

**11. Main Hub reads GPS.** `📍 GPS: Lat ..., Lon ..., Speed ..., Sats ...`
once fixed. Indoors, expect `📍 GPS: Waiting for fix...` — this is normal
and, per §6 below, does not block telemetry (network time does that job
instead).

**12. Main Hub connects to EC20.** `🔌 Powering on EC20 modem...` →
`✅ Registered on cellular network` → `✅ Cellular data link up` with a CSQ
value. If it fails: `❌ Network registration failed` — check SIM/APN/antenna
per [`../smart-driver-docs/4g-ec20-integration-plan.md`](../smart-driver-docs/4g-ec20-integration-plan.md) §6.1.

**13. EC20 connects to mobile network.** Same step as 12 — registration and
PDP context are one call (`modemNetworkConnect()`).

**14. Main Hub authenticates to Firebase.** `🔑 Requesting device Firebase
token from backend...` → `🔑 Exchanging custom token for a Firebase ID
token...`. Both must succeed once before any telemetry write — check the
status line: `Firebase auth: YES`.

**15. Main Hub syncs real time.** `🕐 Network time synced: YYYY-MM-DD
HH:MM:SS UTC`. Must show `Epoch synced: YES` in the status line before
telemetry writes succeed — Firebase's rules reject any write without it.

**16. Main Hub writes telemetry directly to Firebase.**
`✅ Firebase accepted live telemetry (HTTP 200)`. If you see `HTTP 401` or
`403`, re-check `firebase_rules.json` deployed to the console matches the
repo copy exactly.

**17. Firebase stores device-specific data.** In the Firebase console →
Realtime Database, watch `/devices/{deviceId}/safetyData/live` update every
~3 seconds, and `/devices/{deviceId}/status/online` flip to `true`.

**18. Mobile app scans QR.** `DeviceRegistrationScreen.tsx` — point the
camera at the QR printed by `provision_device.py`. Confirms
`{v:1, device_id, secret}` parses.

**19. Device ID extracted, device claimed.** App calls `POST
/devices/register`; on success the device appears in the "My Devices" list
(`DeviceListScreen.tsx`) — confirm the online dot reflects reality (this is
the fix from this pass; it used to always show offline).

**20. Dashboard displays live values.** Open the device →
`IoTDashboardScreen.tsx` shows risk score, GPS map (once fixed), driver
state, distance/TTC — all sourced from the same `/safetyData/live` node
written in step 16/17.

---

## Part 2 — failure scenarios

### Test A — 4G disconnected
Power-cycle or physically disconnect the antenna mid-run.
**Expect:** Hub keeps running (GPS/ultrasonic/alert LED unaffected — no
loop() stall). `maintainCellularLink()` retries with exponential backoff
(5s → 10s → ... capped at 60s). After 5 consecutive failures,
`modemHardReset()` fires automatically. LED2/green goes OFF while
disconnected. Once signal returns, telemetry resumes without a manual reset.

### Test B — ESP32-CAM disconnected
Power off the CAM board.
**Expect:** Hub's `driverVisible()` goes false within `CAM_TIMEOUT_MS`
(5s). Status line shows `NOT VISIBLE (drowsy ignored)`. If it's been off
since boot, the 15s ESP-NOW warning fires:
`⚠️ No ESP-NOW packet received in 15s — check ESPNOW_CHANNEL ...`. LED2/green
drops to slow-blink (degraded). Dashboard's `driverVisible` field goes
`false`, `eyeStatus`/`yawningStatus` report `"unknown"`/`"normal"`.

### Test C — AI server offline
Kill the `server.py` process.
**Expect:** CAM's `webSocket` disconnects — `❌ WebSocket disconnected!`,
LED reverts to slow blink (once/3s). It keeps retrying
(`setReconnectInterval(5000)`). No new ESP-NOW packets are sent, so the
Hub's `driverVisible()` goes stale exactly as in Test B once
`CAM_TIMEOUT_MS` elapses — old AI data does not get used for new safety
decisions.

### Test D — GPS has no fix
Run indoors, or shield the antenna.
**Expect:** `gpsFixed` stays/goes `false`. Dashboard shows "GPS not
acquired" instead of stale coordinates (`IoTDashboardScreen.tsx` gates the
map render on `gps.fixed`). Telemetry still sends — GPS fix is not a
prerequisite for the Firebase write, only network time is (§6 in
`system-catalog.md`). If the module goes fully silent (no NMEA bytes at
all, not just no fix), `gpsFixed` still correctly flips false within
`GPS_TIMEOUT_MS` (10s) — this is the staleness fix from this pass.

### Test E — Firebase/backend unavailable
Block the RTDB host, or let the device-token fetch fail (stop `uvicorn` if
testing locally without the deployed backend).
**Expect:** Hub does not crash. If the ID token is still cached and only
`identitytoolkit`/RTDB is unreachable, individual write attempts fail with
a logged HTTP/err code and retry next cycle. If the *backend* is
unreachable and no token has ever been fetched, telemetry stays paused
(`⚠️ No Firebase ID token yet — skipping this telemetry cycle`) until it
recovers — sensors, GPS, and the alert LEDs keep running throughout.
