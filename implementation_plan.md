# Implementation Plan v5: Phases 2–5
## Tomato Leaf Disease Advisory Platform
**FYP by Sia Jia Le (22062566) — Sunway University**
**Supersedes implementation_plan.md (v4)**

---

## Current State (Phases 0, 1 & 2 — COMPLETE)

### Phase 0 & 1
- FastAPI DDD backend running on PostgreSQL
- ONNX ResNet34 inference with TTA
- Lesion segmenter (severity estimation)
- Confidence policy (alternatives on low confidence)
- Treatment recommendation engine + seeded `treatment_options` table
- Treatment logs, diagnosis sessions
- Vue 3 frontend: DetectView, HistoryView, ResultCard with severity/treatment/alternatives
- Docker Compose: backend + frontend (Vite dev server) + PostgreSQL

### Phase 2 ✅
- Open-Meteo weather integration (`GET /api/v1/weather-risk?lat={}&lng={}`)
- Rule-based disease risk model (all 10 diseases, two modes: humid_warm / hot_dry)
- Prevention tips with 4-language support (`GET /api/v1/prevention-tips?stage={}&lang={}`)
- 16 prevention tips seeded into Docker PostgreSQL
- LLM chatbot stub (`/api/v1/chat`) returns 503 until Anthropic API key is configured
- WeatherRiskView and PreventionView added to frontend
- `httpx` added to requirements for async HTTP calls

---

## Scope Decisions for Phases 2–5

Some features from the full roadmap are out of scope for FYP implementation:
- **Voice STT/TTS** — requires Google Cloud billing; documented as future work
- **SMS/USSD fallback** — requires telco partnership; documented as future work
- **Marketplace integration** — post-graduation scope
- **Celery/beat scheduler** — replaced with a simpler DB-polling approach for scan reminders
- **Payment gateway** — out of FYP scope; schema scaffolded only
- **Risk forecasting ML model** (Phase 4) — rule-based model from Phase 2 is sufficient for FYP

Everything else is implemented in full.

---

## Phase 2 — Advisory & Decision Support

**External dependencies:**
- Open-Meteo (https://api.open-meteo.com) — free, no API key required
- Anthropic Claude API — DEFERRED (wired up when API key is available; DB tables and stub endpoint created now)

### 2.1 Database Migration (`0003_tier2_advisory`)
New tables:
- `weather_risk_alerts` — stores computed risk assessments per location/disease
- `prevention_tips` — admin-curated content in 4 languages (EN/MS/ZH/TA)
- `chat_conversations` — stub for future LLM chatbot (created now, not activated)
- `chat_messages` — stub for future LLM chatbot (created now, not activated)

### 2.2 Backend — Domain Layer
| File | Action |
|---|---|
| `domain/entities/weather_risk_alert.py` | New dataclass |
| `domain/entities/prevention_tip.py` | New dataclass |
| `domain/entities/chat_conversation.py` | New dataclass (stub) |
| `domain/services/disease_risk_model.py` | New — rule-based epidemiological thresholds for all 10 diseases |

### 2.3 Backend — Infrastructure Layer
| File | Action |
|---|---|
| `infrastructure/external/open_meteo_client.py` | New — wraps Open-Meteo hourly forecast API (no API key needed) |
| `infrastructure/ai/llm_client.py` | New stub — raises `NotImplementedError` until API key is provided |
| `infrastructure/persistence/weather_alert_repo.py` | New |
| `infrastructure/persistence/prevention_tip_repo.py` | New |
| `infrastructure/persistence/chat_repo.py` | New stub |
| `infrastructure/persistence/models.py` | Add WeatherRiskAlert, PreventionTip, ChatConversation, ChatMessage ORM models |

### 2.4 Backend — Application Layer
| File | Action |
|---|---|
| `application/use_cases/get_weather_risk.py` | New — fetches Open-Meteo forecast → runs DiseaseRiskModel → saves alert |
| `application/dtos/weather_dto.py` | New |
| `application/dtos/prevention_dto.py` | New |

### 2.5 Backend — Interface Layer
| File | Action |
|---|---|
| `interface/routers/weather_router.py` | New — `GET /api/v1/weather-risk?lat={}&lng={}` |
| `interface/routers/prevention_router.py` | New — `GET /api/v1/prevention-tips?stage={}&lang={}` |
| `interface/routers/chat_router.py` | New stub — returns `503 Service Unavailable` with message "LLM not configured yet" |
| `main.py` | Register 3 new routers |
| `backend/requirements.txt` | Add `httpx` (for async Open-Meteo HTTP calls) |

### 2.6 Scripts
| File | Action |
|---|---|
| `scripts/seed_prevention_tips.py` | New — 16 tips covering 4 growth stages (seedling/vegetative/flowering/fruiting) × 4 categories (watering/spacing/soil/companion_planting) in EN, MS, ZH, TA |

### 2.7 Frontend
| File | Action |
|---|---|
| `frontend/src/components/weather/RiskAlertBanner.vue` | New — shows disease risk level (low/moderate/high) with color coding |
| `frontend/src/components/prevention/PreventionTipCard.vue` | New — card per tip with growth stage badge |
| `frontend/src/views/WeatherRiskView.vue` | New — location input + risk assessment per disease |
| `frontend/src/views/PreventionView.vue` | New — prevention tips filtered by growth stage + language |
| `frontend/src/stores/weatherStore.js` | New |
| `frontend/src/router/index.js` | Add `/weather-risk` and `/prevention` routes |
| `frontend/src/services/api.js` | Add weather and prevention API calls |
| `frontend/src/components/layout/AppHeader.vue` | Add nav links for new views |

### 2.8 Environment Variables Added
```
# No new env vars needed for Phase 2
# (Open-Meteo requires no API key)
# ANTHROPIC_API_KEY= will be added when chatbot is activated
```

---

## Phase 3 — Farm Management & Tracking

**Status:** In progress (approved 2026-06-30)

**New infrastructure:** Redis added to Docker Compose (OTP cache + JWT denylist)

### OTP Strategy (Dev)
OTP is a randomly generated 6-digit code printed to the **backend console log**.
Copy it from the Docker logs and paste it in the frontend OTP field to verify.
> Future: Replace with WhatsApp Business API — see `future_implementation.md`

### 3.1 New DB Tables (auto-created via `Base.metadata.create_all` on startup)
New tables: `farmers`, `farms`, `plots`, `scan_reminders`, `yield_estimates`
Altered tables: `predictions` (add optional `plot_id`, `farmer_id` FKs), `chat_conversations` (add optional `farmer_id` FK)

### 3.2 Backend — Domain Layer
| File | Action |
|---|---|
| `domain/entities/farmer.py` | New dataclass |
| `domain/entities/farm.py` | New dataclass |
| `domain/entities/plot.py` | New dataclass |
| `domain/entities/scan_reminder.py` | New dataclass |
| `domain/entities/yield_estimate.py` | New dataclass |
| `domain/services/yield_estimator.py` | New — maps severity score → estimated yield loss % per disease |
| `domain/repositories/farmer_repository.py` | New ABC |
| `domain/repositories/farm_repository.py` | New ABC |
| `domain/repositories/plot_repository.py` | New ABC |

### 3.3 Backend — Infrastructure Layer
| File | Action |
|---|---|
| `infrastructure/auth/jwt_handler.py` | New — create/verify JWT (python-jose); 30-day expiry |
| `infrastructure/auth/otp_cache.py` | New — Redis-backed; stores phone→OTP with 5-min TTL; logs OTP to console in dev |
| `infrastructure/persistence/postgres_farmer_repo.py` | New |
| `infrastructure/persistence/postgres_farm_repo.py` | New |
| `infrastructure/persistence/postgres_plot_repo.py` | New |
| `infrastructure/persistence/models.py` | Add FarmerRecord, FarmRecord, PlotRecord, ScanReminderRecord, YieldEstimateRecord ORM models |
| `docker-compose.yml` | Add `redis:7-alpine` service; add `REDIS_URL` + `JWT_SECRET_KEY` env vars to backend |
| `backend/requirements.txt` | Add `python-jose[cryptography]`, `passlib[bcrypt]`, `redis` |

### 3.4 Backend — Application Layer
| File | Action |
|---|---|
| `application/use_cases/request_otp.py` | New — generates OTP, caches in Redis, logs to console |
| `application/use_cases/verify_otp.py` | New — validates OTP, creates/fetches farmer, returns JWT |
| `application/use_cases/manage_farm.py` | New — create/list farms; create/list plots per farm |
| `application/use_cases/get_plot_history.py` | New — paginated scan history for a plot |
| `application/use_cases/get_yield_estimate.py` | New — aggregates prediction severities → estimated loss |
| `application/dtos/farmer_dto.py` | New |
| `application/dtos/farm_dto.py` | New |

### 3.5 Backend — Interface Layer
| File | Action |
|---|---|
| `interface/routers/auth_router.py` | New — `POST /api/v1/auth/request-otp`, `POST /api/v1/auth/verify-otp` |
| `interface/routers/farm_router.py` | New — `GET/POST /api/v1/farms`, `POST /api/v1/farms/{id}/plots`, `GET /api/v1/plots/{id}/history`, `GET /api/v1/plots/{id}/yield-estimate`, `PATCH /api/v1/plots/{id}/reminders` |
| `interface/middleware/auth_middleware.py` | New — FastAPI `Depends` for optional + required JWT |
| `main.py` | Register `auth_router` and `farm_router` |

### 3.6 Infrastructure — Model Registry (Multi-Crop Scaffold)
| File | Action |
|---|---|
| `infrastructure/ml/model_registry.py` | New — `get_inferencer(crop_type)` lookup; only `tomato` registered now |

### 3.7 Frontend
| File | Action |
|---|---|
| `frontend/src/views/auth/PhoneLoginView.vue` | New — phone number input → POST /auth/request-otp |
| `frontend/src/views/auth/OTPVerifyView.vue` | New — 6-digit OTP input → POST /auth/verify-otp → stores JWT |
| `frontend/src/views/FarmListView.vue` | New — list farmer's farms + create farm form |
| `frontend/src/views/PlotDetailView.vue` | New — scan history timeline + yield estimate card |
| `frontend/src/views/PlotSettingsView.vue` | New — scan reminder interval config |
| `frontend/src/components/farm/FarmCard.vue` | New |
| `frontend/src/components/farm/PlotHistoryTimeline.vue` | New |
| `frontend/src/stores/authStore.js` | New — phone, token, farmer profile; persists token to localStorage |
| `frontend/src/stores/farmStore.js` | New — farms, plots, plot history |
| `frontend/src/router/index.js` | Add `/login`, `/verify-otp`, `/farms`, `/farms/:id/plots/:plotId`, `/plots/:id/settings`; navigation guard for auth-required routes |
| `frontend/src/services/api.js` | Add `Authorization: Bearer {token}` injection; auth + farm + plot API functions |
| `frontend/src/components/layout/AppHeader.vue` | Add "My Farms" nav link; show login/logout button based on auth state |

### 3.8 Environment Variables Added
```env
JWT_SECRET_KEY=change-me-in-production-use-random-32-char-string
REDIS_URL=redis://redis:6379/0
```

---

## Phase 4 — Community & Market + Predictive ✅

**New npm package:** Leaflet (outbreak heatmap)

### 4.1 Database Migration (`0005_tier4_community_and_predictive`)
New tables: `outbreak_events`, `forum_posts`, `forum_replies`, `extension_requests`, `cooperatives`, `cooperative_members`, `risk_forecasts`

### 4.2 Backend — Domain Layer
| File | Action |
|---|---|
| `domain/entities/outbreak_event.py` | New |
| `domain/entities/forum_post.py` | New |
| `domain/entities/extension_request.py` | New |
| `domain/entities/cooperative.py` | New |
| `domain/entities/risk_forecast.py` | New |
| `domain/services/location_privacy.py` | New — 5 km grid coarsening |
| `domain/services/outbreak_aggregator.py` | New |
| `domain/repositories/outbreak_repository.py` | New ABC |
| `domain/repositories/forum_repository.py` | New ABC |
| `domain/repositories/extension_repository.py` | New ABC |
| `domain/repositories/cooperative_repository.py` | New ABC |

### 4.3 Backend — Infrastructure Layer
| File | Action |
|---|---|
| `infrastructure/persistence/postgres_outbreak_repo.py` | New — PostGIS heatmap query |
| `infrastructure/persistence/postgres_forum_repo.py` | New |
| `infrastructure/persistence/postgres_extension_repo.py` | New |
| `infrastructure/persistence/postgres_cooperative_repo.py` | New |
| `infrastructure/persistence/models.py` | Add 6 new ORM models |

### 4.4 Backend — Application Layer
| File | Action |
|---|---|
| `application/use_cases/publish_outbreak_event.py` | New — called after every prediction |
| `application/use_cases/get_regional_heatmap.py` | New |
| `application/use_cases/request_extension_officer.py` | New |
| `application/use_cases/manage_forum.py` | New — post + reply |
| `application/use_cases/manage_cooperative.py` | New |
| `application/use_cases/predict_disease.py` | Update — call publish_outbreak_event after save |

### 4.5 Backend — Interface Layer
| File | Action |
|---|---|
| `interface/routers/heatmap_router.py` | New — `GET /api/v1/heatmap` |
| `interface/routers/forum_router.py` | New — posts + replies |
| `interface/routers/extension_router.py` | New — escalation requests |
| `interface/routers/cooperative_router.py` | New — member management |
| `main.py` | Register 4 new routers |

### 4.6 Frontend
| File | Action |
|---|---|
| `frontend/src/views/HeatmapView.vue` | New — Leaflet map + outbreak layer |
| `frontend/src/views/forum/ForumListView.vue` | New |
| `frontend/src/views/forum/ForumPostDetailView.vue` | New |
| `frontend/src/views/cooperative/CooperativeDashboardView.vue` | New |
| `frontend/src/components/heatmap/OutbreakMapLayer.vue` | New |
| `frontend/src/components/extension/ExtensionRequestForm.vue` | New |
| `frontend/src/stores/heatmapStore.js` | New |
| `frontend/src/stores/forumStore.js` | New |
| `frontend/src/router/index.js` | Add heatmap, forum, cooperative routes |
| `frontend/src/services/api.js` | Add heatmap, forum, extension, cooperative API calls |
| `frontend/package.json` | Add `leaflet`, `@vue-leaflet/vue-leaflet` |

---

## Phase 5 — Accessibility, Governance & Trust ✅

**Scope implemented (2026-07-01):**
- Tier 6 partial: offline-first (Dexie queue), multi-language (EN/MS/ZH/TA), simplified UI mode
- Tier 8 full: confidence presentation, human-in-the-loop escalation, data privacy consent, drift monitoring
- Tier 7 (monetization) and remaining Tier 6 features deferred → `future_implementation.md`

### 5.1 Backend — Domain Layer
| File | Action |
|---|---|
| `domain/services/confidence_presentation.py` | New — wraps ConfidencePolicy; `requires_escalation()`, `get_escalation_reason()` |
| `domain/entities/farmer.py` | Update — add `preferred_language`, `data_sharing_consent` fields |

### 5.2 Backend — Infrastructure Layer
| File | Action |
|---|---|
| `infrastructure/ml/drift_monitor.py` | New — `DriftMonitor` class; `flag_for_review()` inserts to `drift_review_queue` |
| `infrastructure/persistence/models.py` | Add `preferred_language`, `data_sharing_consent` to `FarmerRecord`; add `DriftReviewRecord` |
| `infrastructure/persistence/postgres_farmer_repo.py` | Update — add `update_consent()`, update `_to_entity()` |

### 5.3 Backend — Application Layer
| File | Action |
|---|---|
| `application/use_cases/predict_disease.py` | Update — add `extension_repo`, `drift_monitor`, `heatmap_consent` params; auto-escalate for low-confidence/severe; flag for drift review |
| `application/dtos/farmer_dto.py` | Update — add `preferred_language`, `data_sharing_consent` to `FarmerProfileDTO`; new `UpdateConsentDTO` |

### 5.4 Backend — Interface Layer
| File | Action |
|---|---|
| `interface/routers/prediction_router.py` | Update — add `DriftMonitor`, `extension_repo`, `_get_heatmap_consent` dep |
| `interface/routers/auth_router.py` | Update — add `PATCH /api/v1/auth/consent` endpoint |
| `interface/routers/admin_router.py` | New — `GET /api/v1/admin/drift-review`, `PATCH .../mark-reviewed` |
| `interface/routers/__init__.py` | Update — add `admin_router` |
| `main.py` | Update — register `admin_router` |

### 5.5 Frontend — Multi-Language Support (Tier 6)
| File | Action |
|---|---|
| `frontend/package.json` | Add `vue-i18n@9`, `dexie`, `vite-plugin-pwa` |
| `frontend/src/i18n/index.js` | New — i18n instance with locale detection |
| `frontend/src/i18n/en.json` | New — English translations |
| `frontend/src/i18n/ms.json` | New — Bahasa Malaysia |
| `frontend/src/i18n/zh.json` | New — Simplified Chinese |
| `frontend/src/i18n/ta.json` | New — Tamil |
| `frontend/src/main.js` | Update — register i18n plugin |

### 5.6 Frontend — Offline-First Mode (Tier 6)
| File | Action |
|---|---|
| `frontend/vite.config.js` | Add `VitePWA` plugin — service worker + API cache |
| `frontend/src/services/offlineQueue.js` | New — Dexie IndexedDB queue; `queueScan()`, `syncPending()` |
| `frontend/src/App.vue` | Update — bind offline sync on `window.online` event |
| `frontend/src/services/api.js` | Update — add `updateConsent()`, `listDriftReviews()`, `markDriftReviewed()` |

### 5.7 Frontend — Simplified UI Mode (Tier 6)
| File | Action |
|---|---|
| `frontend/src/stores/settingsStore.js` | New — persists `uiMode` + `locale` to localStorage |
| `frontend/src/components/settings/UIModeToggle.vue` | New — Standard / Simple toggle |
| `frontend/src/App.vue` | Apply `:class="settings.uiMode"` on root div |
| `frontend/src/assets/simple-mode.css` | New — `.simple` overrides: 48px tap targets, high-contrast, large font |
| `frontend/src/views/SettingsView.vue` | New — combined settings page (UI mode + language + privacy link) |
| `frontend/src/components/layout/AppHeader.vue` | Update — nav translated via `$t()`; Settings nav link |

### 5.8 Frontend — Data Privacy UI (Tier 8)
| File | Action |
|---|---|
| `frontend/src/views/PrivacySettingsView.vue` | New — toggle heatmap + research consent; calls `PATCH /api/v1/auth/consent` |
| `frontend/src/router/index.js` | Add `/settings`, `/settings/privacy` routes |

---

## Mobile Responsiveness ✅ (2026-07-01)

| File | Change |
|---|---|
| `frontend/src/assets/main.css` | Added `--card-bg` CSS variable; mobile container padding; `font-size: 16px` on inputs to prevent iOS zoom |
| `frontend/src/components/layout/AppHeader.vue` | Hamburger menu for ≤768px; nav collapses to full-width dropdown; X animation; auth section moves inside dropdown on mobile |
| `frontend/src/views/HomeView.vue` | Hero h1 scales to `1.6rem` on mobile; reduced top padding |
| `frontend/src/components/ImageDropzone.vue` | Mobile text variant ("Tap to upload"); reduced padding; preview-actions stack vertically on mobile |
| `frontend/src/components/ResultCard.vue` | `disease-row` wraps on narrow screens; body padding reduced; h2 scales |
| `frontend/src/views/WeatherRiskView.vue` | Input row stacks vertically; buttons go full-width on mobile |
| `frontend/src/views/HeatmapView.vue` | Input row stacks; map height reduced to 300px on mobile |
| `frontend/src/views/forum/ForumListView.vue` | Disease-select stacks; submit button full-width on mobile |
| `frontend/src/views/cooperative/CooperativeDashboardView.vue` | Form-row stacks; coop-header stacks on mobile |
| `frontend/src/views/SettingsView.vue` | Language grid goes single-column on narrow phones |

---

## Build Order

1. Phase 2 backend (DB migration → domain → infra → use cases → routers)
2. Phase 2 seed script (`seed_prevention_tips.py`)
3. Phase 2 frontend (chat + weather + prevention components/views)
4. Phase 3 backend (DB migration → auth → farm/plot domain → routers)
5. Phase 3 docker-compose (Redis)
6. Phase 3 frontend (auth + farm views + stores)
7. Phase 4 backend (DB migration → outbreak/forum/extension/cooperative)
8. Phase 4 frontend (Leaflet heatmap + forum + cooperative views)
9. Phase 5 backend (DB migration → confidence presentation → drift monitor)
10. Phase 5 frontend (offline queue + simplified UI + privacy settings)

---

## API Keys / Services Required

| Phase | Service | Status |
|---|---|---|
| Phase 2 | Open-Meteo | ✅ No key needed (free, open) |
| Phase 2 | Anthropic LLM | ⏳ Stub in place — activate when key available |
| Phase 3 | Redis | ✅ Self-hosted in Docker |
| Phase 3 | JWT | ✅ Self-contained (python-jose) |
| Phase 3 | WhatsApp OTP | ⏳ Future — see `future_implementation.md` |
| Phase 4 | Leaflet/OpenStreetMap | ✅ No key needed |
| Phase 5 | Twilio SMS | ⏳ Scaffold only — see `future_implementation.md` |
| Phase 5 | Billplz payments | ⏳ Scaffold only — see `future_implementation.md` |

---

## Baseline Model Comparison — Per-Model Folder Restructure ✅ IMPLEMENTED

**Status:** All 8 folders (`AlexNet/`, `VGG16/`, `MobileNetV2/`,
`EfficientNetB0/`, `ResNet50/`, `KNN/`, `SVM/`, `RandomForest/`) plus
`compare_models.py` have been created per the structure below. Each DL
baseline uses two-stage fine-tuning (frozen head, then per-model
architecture-native unfreeze targets — Conv4/5 for AlexNet, block4/5 for
VGG16, last 5 blocks for MobileNetV2, last 2 MBConv groups for
EfficientNetB0, layer3/4 for ResNet50), basic augmentation only, a
standard `Dropout(0.2)->Linear` head, no attention, no weighted sampler,
plain cross-entropy loss, and its own independent `config.py` (own
epochs/LR/patience — not imported from `resnet34_model/src/config.py`).
Classical-ML baselines use a frozen pretrained-ResNet34 feature extractor
with scikit-learn's stock defaults (KNN n_neighbors=5, SVM RBF C=1.0,
RandomForest n_estimators=100). Training/evaluation itself has not been
run yet (requires HPC/GPU + the processed dataset) — only the code
exists so far.

**Goal:** Compare the proposed ResNet34 (`resnet34_model/`) against 8 baselines
(AlexNet, VGG16, MobileNetV2, EfficientNet-B0, ResNet50, k-NN, SVM, Random Forest)
to support the FYP Results chapter, per `baseline_models_plan.md`.

**Change from `baseline_models_plan.md`:** that document nests every baseline
under `resnet34_model/outputs/<model>/outputs/` and assumes a shared,
already-refactored `src/train.py::run_experiment()` / `src/evaluate.py`
API. Neither matches what's actually in `resnet34_model/src/` today (those
are standalone scripts: `config.py`, `dataset.py`, `model.py`, `train.py`,
`evaluate.py`, each with a `main()`, no `output_dir` params, no `src.`
package prefix — see e.g. `resnet34_model/src/train.py`).

Instead, each baseline gets its **own top-level folder**, sibling to
`resnet34_model/`, `backend/`, `frontend/`, mirroring the coding style
`resnet34_model/src/` already uses (self-contained scripts, not a shared
`baselines.py`/`run_baselines.py` abstraction). No dataset duplication —
every folder's `config.py` points at `resnet34_model/data/processed` via a
relative path.

### New top-level folders

```
AlexNet/
VGG16/
MobileNetV2/
EfficientNetB0/
ResNet50/
KNN/
SVM/
RandomForest/
```

Each DL folder (`AlexNet/`, `VGG16/`, `MobileNetV2/`, `EfficientNetB0/`, `ResNet50/`):
```
<Model>/
├── src/
│   ├── config.py       # DATA_DIR -> ../resnet34_model/data/processed (relative), OUTPUT_DIR -> ./outputs
│   ├── dataset.py       # get_dataloaders() — basic transform only (RandomResizedCrop, HFlip, Rotate±15, Normalize), no weighted sampler
│   ├── model.py         # build_<model>() — pretrained ImageNet weights + standard head: Dropout(0.2) → Linear(N, 10)
│   ├── train.py         # main() — Stage A (frozen backbone) then Stage B (unfreeze last block(s)), same 2-stage shape as resnet34_model/src/train.py but per-model unfreeze targets from baseline_models_plan.md
│   └── evaluate.py       # main() — eval on processed/test only, writes eval_results.json, cm_processed_test.png, classification_report.txt to ./outputs
├── outputs/              # created at runtime
├── requirements.txt      # same as resnet34_model/requirements.txt + timm for EfficientNetB0
└── README.md             # how to run: cd <Model> && python src/train.py && python src/evaluate.py
```

Each classical-ML folder (`KNN/`, `SVM/`, `RandomForest/`):
```
<Model>/
├── src/
│   ├── config.py        # same DATA_DIR/OUTPUT_DIR convention
│   ├── extract_features.py  # pretrained ResNet34 (fc → Identity) feature extractor, shared logic duplicated per folder (per your "separate folder each" choice)
│   └── train_evaluate.py    # fit classifier on train features, evaluate on test features, write eval_results.json etc.
├── outputs/
├── requirements.txt      # scikit-learn, torch, torchvision
└── README.md
```

### Shared comparison table script

A single top-level script reads `eval_results.json` from every model
folder's `outputs/` and prints/saves the comparison table (mirrors
`generate_comparison_table()` from `baseline_models_plan.md`):
```
compare_models.py   # reads AlexNet/outputs/eval_results.json, VGG16/outputs/..., etc.
                     # plus resnet34_model/outputs/evaluation_report/eval_results.json for the proposed model
```

### Steps

1. Create the 8 top-level folders with the structure above.
2. For each DL baseline: `config.py`, `dataset.py` (basic-augmentation only,
   per `baseline_models_plan.md`), `model.py` (standard head per
   architecture), `train.py` (2-stage fine-tune, unfreeze targets from
   `baseline_models_plan.md`'s `get_stage_b_params_*` functions), `evaluate.py`.
3. For each classical-ML baseline: `config.py`, `extract_features.py`,
   `train_evaluate.py`.
4. Add `compare_models.py` at repo root.
5. Update root `requirements.txt` / each folder's own `requirements.txt`
   (`timm` for EfficientNetB0, `scikit-learn` for classical ML).
6. Do **not** modify `resnet34_model/` — its existing scripts are left as-is;
   `resnet34_model/outputs/evaluation_report/eval_results.json` (or
   equivalent) is only read by `compare_models.py`, not written to.

**Not doing:** the `baselines.py`/`run_baselines.py`/`BASELINE=<name>` env-var
launcher pattern, or modifying `resnet34_model/src/train.py` /
`evaluate.py` to accept `output_dir` — those don't match the existing
codebase and aren't needed once each model owns its own folder.

---

## Plan 1 — Background-Randomization Run for EfficientNetB0 (NEW, pending approval)

**Task:** Add ONE new experiment row, `efficientnetb0_on_bgrand` = the existing
`efficientnetb0_on` Stack-ON solution **plus** background-randomization
augmentation (per `plan1_background_randomization.md`). Measure whether it
shrinks the real-world generalization gap.

### Hard constraints (user + the plan)
1. **Do NOT modify any existing file.** No edits to `experiments/common/*`,
   the existing `efficientnetb0_*.yaml`, `run.py`, `compare.py`, or the 12
   ablation runs. Everything new lives in a **new folder**.
2. **One variable.** New run differs from `efficientnetb0_on` by *exactly one
   thing*: a background-randomization transform prepended to the **train**
   pipeline. Same backbone, split, seed (42), budget, head, CBAM, label
   smoothing, basic + advanced augmentation.
3. **Train-set only.** Val/test/real-world get resize+normalize, nothing else.
4. **No real-world images in training.** Backgrounds are generic textures; the
   runner asserts `background_dir` is disjoint from the real-world test source.
5. Select on PlantVillage val macro-F1; read real-world once.

### Design: isolation by *importing* the shared code, never editing it
Reuse via import so the ablation code path stays byte-for-byte identical:
- `common.data` → `_basic_four`, `_advanced_block`, `IMAGENET_MEAN/STD`,
  `AlbumentationsImageFolder`, `seed_worker`, `build_eval_transform`.
- `common.engine` → `_run_epoch` unchanged.
- `common.backbones` → `build_backbone`. `common.seeding` → `seed_everything`.
- `common.evaluate` → `evaluate_run` **verbatim** (bgrand is train-only, so
  eval is identical; it rebuilds the model from the checkpoint's cfg).

### New files (all under `experiments/plan1_bgrand/` unless noted)
| File | Purpose |
|---|---|
| `__init__.py` | package marker |
| `bg_randomize.py` | `BackgroundRandomize` Albumentations `ImageOnlyTransform`: with prob `p`, segment the leaf, composite onto a random background, optional boundary blur. Loads backgrounds once. `segment_leaf(img)` helper. |
| `data_bgrand.py` | `build_train_loader_bgrand(...)`: mirrors `common.data.build_loaders` but **prepends** `BackgroundRandomize` to the train transform. Val/test loaders come from `common.data` unchanged. Same return-tuple shape. |
| `engine_bgrand.py` | `train_run_bgrand(cfg, results_dir, device)`: two-stage loop reusing `_run_epoch`/`build_backbone`/`seed_everything`; identical checkpointing + metrics.json. Only difference vs `engine.train_run`: train loader from `data_bgrand`. Saves same cfg/class_to_idx so `evaluate_run` works unchanged. |
| `run_bgrand.py` | Entry point. Loads config, resolves paths, **asserts** `background_dir` exists and is disjoint from `real_world_dir`, trains, then calls `common.evaluate.evaluate_run`. `--eval-only` supported. |
| `compare_bgrand.py` | Reads `efficientnetb0_on` vs `efficientnetb0_on_bgrand` eval JSONs; prints delta table (Acc, MacroF1, RW_Acc, RW_F1, Gap) + per-class real-world F1. Does **not** touch `compare.py`. |
| `sanity_check_masks.py` | Saves original/mask/composite grids for ~N images/class to eyeball segmentation **before** the full run (plan §5.2). |
| `make_synthetic_backgrounds.py` | Optional fallback: ~60 domain-neutral procedural textures into `data/backgrounds_generic/` so the pipeline runs before real CC0 textures are curated. |
| `configs/efficientnetb0_on_bgrand.yaml` | New run config (real runner schema). |
| `run_bgrand_slurm.sh` | HPC job: mask sanity-check → train+eval → `compare_bgrand`. Partition `gpu-24c-l4-4g`, gres `gpu:l4:1`. |
| `README.md` | Curate backgrounds, sanity-check, run, read result. |

**New data dir:** `data/backgrounds_generic/` (git-ignored images).

### Config (ADAPTED to the actual runner schema, not the plan's illustrative one)
```yaml
run_name: efficientnetb0_on_bgrand
seed: 42
backbone: efficientnetb0
data_dir: data/processed
real_world_dir: data/processed/real_environment_test
stack:                         # IDENTICAL to efficientnetb0_on.yaml
  advanced_augmentation: true
  label_smoothing: 0.1
  strong_head: true
  cbam: true
  stage_b: two_group
background_randomization:      # the ONE new block
  enabled: true
  prob: 0.5                    # p; tune on VAL only
  background_dir: data/backgrounds_generic
  segmentation: hsv_threshold  # hsv_threshold | pretrained (rembg, optional)
  boundary_blur: true
training:                      # IDENTICAL budget
  stage_a_epochs: 15
  stage_b_epochs: 25
  patience: 7
  stage_a_lr: 1.0e-3
  stage_b_lr: 1.0e-4
  batch_size: 32
```

### Segmentation (refinement of plan §3.3 Route A)
Segment **foreground-vs-uniform-background** (estimate background color from
image corners → mask pixels far from it → morphological close/open → largest
connected component) rather than a pure green threshold, so brown/yellow
**lesions** (the diseased pixels the label depends on) are preserved. `cv2`
(already pulled in by albumentations) provides HSV/morphology/connected
components/boundary blur. `pretrained` (rembg) is an optional fallback.

### Steps (after approval)
1. Create the `experiments/plan1_bgrand/` package + files above.
2. Create `data/backgrounds_generic/` + synthetic generator; gitignore bulk images.
3. You run (I ask yes/no each time): synthetic bgs (or your own) →
   `sanity_check_masks.py` → `run_bgrand_slurm.sh` on HPC → `compare_bgrand.py`.
4. Completion walkthrough.

### Will NOT do
- Edit `compare.py`/master table (bgrand compared via `compare_bgrand.py`).
- Tune `p` on real-world data; put real-world images in `background_dir`;
  retrain or alter any existing run.

### Open question before I build
**Backgrounds:** OK to generate ~60 synthetic domain-neutral textures so the
pipeline runs immediately (swap in real CC0 textures later)? Or point
`background_dir` at a texture folder you already have?
