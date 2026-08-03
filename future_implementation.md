# Future Implementation Notes
## Tomato Leaf Disease Advisory Platform
**FYP by Sia Jia Le (22062566) — Sunway University**

This document tracks features that are designed and scaffolded but not yet activated,
and features planned for post-FYP development. Each entry includes the current state,
what is needed to activate it, and the estimated effort.

---

## 1. LLM Chatbot (Anthropic Claude API)

**Current state:** Stub endpoint exists at `POST /api/v1/chat/message`.
Returns `503 Service Unavailable` with message "LLM not configured yet".
DB tables `chat_conversations` and `chat_messages` are created and ready.
`backend/infrastructure/ai/llm_client.py` has the skeleton.

**What is needed:**
- Anthropic API key from https://console.anthropic.com (billing required)
- Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`
- Add to `docker-compose.yml` environment: `ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}`
- Implement `llm_client.py` using `anthropic` Python SDK
- Activate `chat_router.py` to call the LLM use case

**Suggested model:** `claude-haiku-4-5-20251001` (low cost, fast for farming Q&A)

**Packages to add:**
```
anthropic>=0.28.0
```

**Estimated effort:** 1–2 days once API key is available.

---

## 2. WhatsApp OTP (WhatsApp Business API)

**Current state:** OTP is randomly generated and printed to the **backend console log** in dev mode.
The farmer copies it from Docker logs and pastes it into the OTP verification screen.
This works for FYP demonstration but is not suitable for production farmers.

**What is needed:**
- WhatsApp Business Account approved by Meta
- Access to WhatsApp Cloud API (https://developers.facebook.com/docs/whatsapp)
- A pre-approved OTP message template, e.g.:
  `Your TomatoScan verification code is {{1}}. Valid for 5 minutes.`
- Add to `.env`:
  ```
  WHATSAPP_TOKEN=EAAxxxxx
  WHATSAPP_PHONE_NUMBER_ID=1234567890
  ```
- Implement `backend/infrastructure/messaging/whatsapp_client.py`:
  ```python
  POST https://graph.facebook.com/v19.0/{phone_number_id}/messages
  Authorization: Bearer {WHATSAPP_TOKEN}
  Body: { "type": "template", "template": { "name": "otp_verification", ... } }
  ```
- Replace the `print(f"[DEV OTP] {otp}")` line in `otp_cache.py` with a call to `whatsapp_client.send_otp(phone, otp)`

**Alternative (simpler, lower cost):** Twilio Verify API — see section 3 below.

**Estimated effort:** 2–3 days (Meta approval may take 1–2 business days).

---

## 3. SMS OTP via Twilio (Alternative to WhatsApp)

**Current state:** `backend/infrastructure/messaging/sms_gateway_client.py` is a scaffold file
created in Phase 5. It is not wired up to any router or use case.

**What is needed:**
- Twilio account at https://www.twilio.com
- Add to `.env`:
  ```
  TWILIO_ACCOUNT_SID=ACxxxxx
  TWILIO_AUTH_TOKEN=xxxxx
  TWILIO_FROM_NUMBER=+1234567890
  ```
- Use [Twilio Verify](https://www.twilio.com/docs/verify/api) (recommended — handles OTP lifecycle):
  ```python
  client.verify.v2.services(VERIFY_SERVICE_SID).verifications.create(to=phone, channel="sms")
  client.verify.v2.services(VERIFY_SERVICE_SID).verification_checks.create(to=phone, code=otp)
  ```
- Replace Redis OTP cache with Twilio Verify calls (Twilio manages expiry for you)

**Packages to add:**
```
twilio>=8.0.0
```

**Estimated effort:** 1 day once Twilio account is set up.

---

## 4. Scan Reminders via Push Notification / Scheduled Task

**Current state:** Scan reminders are stored in the `scan_reminders` table with `interval_days` and `last_scan_at`.
`GET /api/v1/plots/{id}/reminders` checks if a reminder is due (DB polling, on-demand only).
There is no background scheduler or push notification.

**What is needed (option A — scheduled emails):**
- Celery + Redis Beat for background task scheduling
- Add `celery[redis]` to requirements
- `backend/workers/reminder_worker.py` — periodic task to query overdue reminders and send email via SendGrid/Mailgun

**What is needed (option B — browser push):**
- Web Push API + VAPID keys
- `pywebpush` on the backend to send push notifications
- Service Worker on the frontend to receive them
- `frontend/public/sw.js` + `vite-plugin-pwa`

**Estimated effort:** 3–5 days (Celery option is simpler than Web Push).

---

## 5. Payment Gateway (Billplz — Malaysia)

**Current state:** `backend/infrastructure/payment/payment_gateway_client.py` is a scaffold.
DB tables `subscription_plans`, `farmer_subscriptions` are created in Phase 5 with no data.

**What is needed:**
- Billplz merchant account at https://www.billplz.com
- Add to `.env`:
  ```
  BILLPLZ_API_KEY=xxxxx
  BILLPLZ_COLLECTION_ID=xxxxx
  ```
- Implement bill creation + callback webhook endpoint
- Frontend: subscription upgrade flow (`SubscriptionView.vue`)

**Estimated effort:** 3–4 days.

---

## 6. ML-Based Disease Risk Forecasting

**Current state:** Phase 2 uses a **rule-based** epidemiological model (`domain/services/disease_risk_model.py`).
Thresholds are hardcoded (e.g., humidity > 85% + temp 20–30°C → high fungal risk).
This is accurate enough for demonstration but does not learn from real outbreak data.

**What is needed:**
- Historical outbreak data from the `outbreak_events` table (Phase 4) — needs several months of data
- Train a lightweight time-series classifier (e.g., LightGBM) on weather features → disease incidence
- Export model as ONNX or pickle; add to `ModelRegistry` as `risk_forecaster`
- Replace rule-based thresholds with ML model predictions in `get_weather_risk.py`

**Estimated effort:** 4–6 weeks after sufficient data has been collected.

---

## 7. Voice Interface (STT/TTS)

**Current state:** Not scaffolded. Requires Google Cloud Speech-to-Text + Text-to-Speech billing.

**What is needed:**
- Google Cloud project with STT + TTS APIs enabled
- `google-cloud-speech` and `google-cloud-texttospeech` packages
- Voice-enabled version of DetectView — microphone button to describe symptoms
- TTS readback of diagnosis result

**Target users:** Low-literacy farmers; farmers working with gloves (hands-free).

**Estimated effort:** 1–2 weeks.

---

## 8. SMS/USSD Offline Mode

**Current state:** Not scaffolded. Requires telco partnership in Malaysia (Maxis, Celcom, Digi).

**What is needed:**
- USSD gateway agreement with a Malaysian telco
- Implement a USSD state machine: `*123#` → menu → upload image → receive SMS result
- This is primarily relevant for farmers with feature phones and no smartphone

**Estimated effort:** 4–8 weeks (mostly telco onboarding time).

---

## 9. Marketplace / Input Supplier Integration

**Current state:** `supplier_partners` table scaffolded in Phase 5. No supplier data or matching logic.

**What is needed:**
- Partnership agreements with agro-input suppliers
- Supplier product catalogue API or manual data entry UI
- After diagnosis: show recommended pesticide/fungicide products with links to buy
- `SupplierMatchView.vue` — filterable by disease + region

**Estimated effort:** 2–3 weeks (engineering) + partner onboarding time.

---

---

## 10. Low-End Device Support (Tier 6.2)

**Current state:** The frontend is not optimised for low-end Android devices (budget smartphones).

**What is needed:**
- Client-side image compression before upload (e.g. `browser-image-compression` package) to reduce upload time on slow connections
- Lazy-loading of route components (`() => import(...)`) to reduce initial bundle size
- Bundle size audit with `rollup-plugin-visualizer` — target < 300 kB gzipped
- Consider shipping a quantized ONNX model (INT8) for on-device inference as fallback when network is unavailable

**Target users:** Smallholder farmers using low-end Android devices (2–3 GB RAM, < 100 Mbps).

**Estimated effort:** 3–5 days.

---

## 11. Freemium Subscription Model (Tier 7)

**Current state:** Not scaffolded. All features are currently free.

**What is needed:**
- DB tables: `subscription_plans` (name, price, features JSON), `farmer_subscriptions` (farmer_id, plan_id, start, end)
- Define free vs. premium tiers (e.g., free: 5 scans/day; premium: unlimited + weather risk + cooperative)
- `backend/interface/routers/subscription_router.py` — plan listing + payment initiation
- Gate premium features in relevant routers using a `require_premium_tier` dependency
- Frontend: `SubscriptionView.vue` + upgrade prompt banners

**Revenue model:** RM 9.90/month for individual farmers; RM 49/month for cooperatives.

**Estimated effort:** 1–2 weeks (engineering) + payment gateway integration (see #5).

---

## 12. B2B2C — Agricultural Input Supplier Partnerships (Tier 7)

**Current state:** `supplier_partners` table exists (Phase 5 scaffold). No matching or display logic.

**What is needed:**
- Supplier product catalogue with disease-specific SKU mapping
- After diagnosis: show "Recommended products" section with affiliated links
- Commission tracking table (`affiliate_clicks`, `affiliate_purchases`)
- Partner onboarding dashboard (separate admin UI or Django Admin)

**Estimated effort:** 2–3 weeks (engineering) + supplier partner agreements.

---

## 13. Government / NGO Distribution Channel (Tier 7)

**Current state:** Not scaffolded.

**What is needed:**
- White-label tenant support: `tenants` table linking to custom branding, logos, and extension officer rosters
- Role-based access: extension officer can see flagged cases from their assigned district
- Integration with RISDA/MOA (Malaysia) data export formats (CSV/Excel)
- Custom URL per tenant: `risda.tomatoscan.my`

**Estimated effort:** 3–4 weeks.

---

## 14. Anonymised Data Licensing (Tier 7)

**Current state:** Farmer consent infrastructure is implemented (Phase 5 — `data_sharing_consent` JSONB). No data pipeline or licensing mechanism.

**What is needed:**
- Data export pipeline: nightly job aggregates consented scans (image features + labels only) into an anonymised dataset
- Legal: data processing agreement template for research institutions / agri-companies
- Revenue: negotiate per-dataset fee or revenue share
- Technical: differential privacy noise injection before export (e.g., `opacus` library) to meet academic data sharing standards

**Estimated effort:** 4–6 weeks (legal + engineering).

---

## Summary Table

| Feature | Phase Scaffolded | Blocker | Effort |
|---|---|---|---|
| Anthropic LLM chatbot | Phase 2 | API key | 1–2 days |
| WhatsApp OTP | Phase 3 | Meta Business approval | 2–3 days |
| Twilio SMS OTP | Phase 5 (scaffold) | Twilio account | 1 day |
| Push / email scan reminders | Phase 3 | Celery or Web Push setup | 3–5 days |
| Billplz payments | Phase 5 (scaffold) | Merchant account | 3–4 days |
| ML risk forecasting | Phase 2 (rule-based) | 6+ months outbreak data | 4–6 weeks |
| Voice STT/TTS | Not scaffolded | Google Cloud billing | 1–2 weeks |
| SMS/USSD offline mode | Not scaffolded | Telco partnership | 4–8 weeks |
| Marketplace integration | Phase 5 (scaffold) | Supplier partnerships | 2–3 weeks |
| Low-end device support | Not scaffolded | Engineering only | 3–5 days |
| Freemium model | Not scaffolded | Payment gateway + gating | 1–2 weeks |
| B2B2C supplier partnerships | Partial scaffold | Partner agreements | 2–3 weeks |
| Government/NGO distribution | Not scaffolded | MOA partnership | 3–4 weeks |
| Anonymised data licensing | Consent ready | Legal + pipeline | 4–6 weeks |
