# Firebase Setup Guide

## From zero to a working Realtime Database for this project

This is a from-scratch setup guide, not just a config reference — follow it
top to bottom if you're standing up a new Firebase project. If the project
already exists, skip to whichever step you need.

**Current project (as of 2026-08-19):** `drive-safe-udl`, RTDB at
`https://drive-safe-udl-default-rtdb.asia-southeast1.firebasedatabase.app`
— this replaced the earlier shared `smartdriversafety-33479` project. Steps
1–6 below are done for it: project created, RTDB enabled, Auth enabled,
service account generated and dropped into
`backend/clean_run/firebase-service-account.json`, and the Web API key +
full web config wired into both the firmware (`FIREBASE_API_KEY`) and
`frontend/.env`. **Still open: confirm `firebase_rules.json` is actually
pasted into this project's console** (step 3) — a freshly created RTDB
starts locked or wide open depending what was picked at creation time, and
the rules file being correct in this repo doesn't mean it's live in the
console yet. Check via §8's curl chain below (hop 3 fails with `401`/`403`
if the rules aren't deployed, or succeeds regardless of ownership if the
database is still in open test mode — either result tells you something).

---

## 1. Create the Firebase project

1. Go to the [Firebase console](https://console.firebase.google.com/) and
   click **Add project**.
2. Name it (e.g. `smart-driver-safety`), accept or decline Google Analytics
   (not used by this system — safe to skip).
3. Wait for provisioning to finish.

## 2. Enable the Realtime Database

1. In the left sidebar: **Build → Realtime Database → Create Database**.
2. Pick a region close to your users/demo venue (affects latency, not
   correctness).
3. Start in **locked mode** — you'll paste real rules in the next step, so
   the default "test mode" (open to everyone) should never go live.

## 3. Paste the security rules

1. Open `backend/clean_run/iot/firebase_rules.json` in this repo.
2. In the console: **Realtime Database → Rules** tab.
3. Paste the file's contents (the console's rules editor accepts the `//`
   comments at the top — a strict JSON parser wouldn't, which is why the
   file keeps them).
4. **Publish.**

What these rules actually do (see the file's own header comment for the
full rationale):
- Reads: only the device's *owner*, checked via `auth.uid` against
  `/devices/{id}/meta/ownerUid`.
- Writes to `/devices/{id}/safetyData/live` and `/devices/{id}/status`:
  only the device itself, via a token carrying `role=='iot_device'` and a
  matching `device_id` claim (minted by `POST /iot/device-token`).
- Every other write (`meta`, `alerts`, `safetyData/snapshots`): backend
  Admin SDK only, which bypasses rules entirely by design.

## 4. Enable Authentication

1. **Build → Authentication → Get started.**
2. Under **Sign-in method**, you don't need to enable any specific provider
   (Email/Password, Google, etc.) for this system — everything here
   authenticates via **custom tokens**, which work as soon as Authentication
   is enabled at all, no specific provider toggle required.
3. If `frontend/`'s own user login uses a specific provider (check
   `frontend/src/services` or however it signs up users), enable that too —
   that's a separate concern from the device/IoT token flow this guide
   covers.

## 5. Generate the Admin SDK service account (for the backend)

1. **Project settings (gear icon) → Service accounts** tab.
2. **Generate new private key** → confirms a JSON file download.
3. Wire it into the backend, per `backend/clean_run/iot/firebase_admin_service.py`'s
   required env vars — pick ONE:
   - `FIREBASE_SERVICE_ACCOUNT_JSON` — the full JSON file contents as a
     single-line env var string (what most PaaS deploy targets want —
     Railway, Render, Docker secrets).
   - `GOOGLE_APPLICATION_CREDENTIALS` — a filesystem path to the JSON file
     (local dev only; don't commit the file).
4. This is what lets the backend mint custom tokens (`POST
   /iot/device-token`, `GET /devices/{id}/firebase-token`) and write with
   Admin privileges (`rtdb_service.py`, the Option-B `/iot/telemetry`
   fallback).

## 6. Two values the firmware needs

The Main Hub's `.ino` needs these as plain `#define`s — neither is secret in
the way a service account key is (they're the same values any Firebase
client SDK embeds openly):

- **Web API key** → Project settings → General tab → "Web API Key". Put it
  in `FIREBASE_API_KEY`.
- **Database URL** → Realtime Database page, shown at the top (e.g.
  `https://your-project-default-rtdb.firebaseio.com`). Put it in
  `FIREBASE_DATABASE_URL`, and the **host only** (strip `https://`) in
  `FIREBASE_DB_HOST` — the two must stay in sync; there's a comment at the
  `#define` reminding of this.

## 7. Set `FIREBASE_DATABASE_URL` in the backend's `.env` too

`rtdb_service.py` needs this independently of the service account —
`backend/clean_run/.env`:
```
FIREBASE_DATABASE_URL=https://your-project-default-rtdb.firebaseio.com
```
Without it, RTDB writes from the backend (device-token issuance's
downstream effects, the Option-B fallback) fail with a clear "Live
telemetry storage is unavailable" error rather than a cryptic SDK failure —
`firebase_admin_service.py` was written to surface this loudly on purpose.

---

## 8. The auth chain, hop by hop

This is what `ensureFirebaseAuth()` does on the Main Hub, broken into
independently testable `curl` calls — if telemetry isn't reaching Firebase,
run these in order to find which hop is broken without touching hardware.

**Hop 1 — device custom token** (`fetchDeviceCustomToken()`):
```bash
curl -i -X POST https://<your-backend-host>/iot/device-token \
  -H "X-Device-Id: ESP32-MAC-<hub-mac-no-colons>" \
  -H "X-Device-Secret: <the device_secret from provision_device.py>"
```
Expect `200` with `{"firebase_token": "<long JWT>", "expires_in": 3600}`.
A `401` here means the device isn't registered, or the secret is wrong —
check `provision_device.py`'s output was pasted into `secrets.h` correctly.

**Hop 2 — exchange for an ID token** (`exchangeForIdToken()`):
```bash
curl -i -X POST "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=<FIREBASE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"token":"<firebase_token from hop 1>","returnSecureToken":true}'
```
Expect `200` with `{"idToken": "<very long JWT>", "expiresIn": "3600", ...}`
— note `expiresIn` is a numeric *string* here, which is why the firmware's
JSON parsing uses `.as<long>()` rather than assuming a JSON number. A `400`
with `INVALID_CUSTOM_TOKEN` usually means the Web API key doesn't match the
project that minted the custom token in hop 1 — double check `FIREBASE_API_KEY`
against the project the backend's service account belongs to.

**Hop 3 — the actual RTDB write** (`sendTelemetryToFirebase()`):
```bash
curl -i -X PUT "https://<FIREBASE_DB_HOST>/devices/ESP32-MAC-<mac>/safetyData/live.json?auth=<idToken from hop 2>" \
  -H "Content-Type: application/json" \
  -d '{"riskScore":0.1,"alertTier":0,"driver":{"drowsyLevel":0,"confidence":0,"eyeStatus":"open","yawningStatus":"normal","earScore":0,"driverVisible":false},"vehicle":{"distanceCm":400,"ttcSeconds":999},"gps":{"latitude":0,"longitude":0,"speedKmh":0,"satellites":0,"fixed":false},"timestampMs":1755600000000,"deviceUptimeMs":1000,"sequenceNum":0}'
```
Expect `200` with the written JSON echoed back. A `401`/`403` here — with a
valid ID token from hop 2 — means the security rules aren't matching:
re-check the `role`/`device_id` claims embedded in the custom token from
hop 1 (`firebase_admin_service.py`'s `issue_device_firebase_token()`)
against what `firebase_rules.json` checks (`auth.token.role`,
`auth.token.device_id`). A `400` with a validation error means the payload
itself violates a `.validate` rule — check `riskScore` is in `[0,1]` and
`timestampMs` is a real epoch (this is exactly the check `syncEpochTime()`
exists to satisfy — if you're testing with a fake timestamp, use something
close to the real current time in milliseconds, not `0` or `millis()`).

**Hop 4 — the mobile app's read side** (unrelated to the write chain above,
included for completeness): the app calls `GET
/devices/{id}/firebase-token` (user JWT auth) → `signInWithCustomToken` →
`onValue()` on the same `/safetyData/live` path. If writes are landing
(confirmed via the console or hop 3 above) but the app shows nothing, the
break is on this read side, not the write side covered by hops 1–3.
