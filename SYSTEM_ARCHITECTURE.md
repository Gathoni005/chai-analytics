# ChaiAnalytics — System Architecture & Implementation Guide

## 1. Development Planning & Phase Structure

The project follows a **6-step phased pipeline** designed for incremental delivery and containerized deployment.

### Phase 1 — Foundation & Data Ingestion
| Step | Script | Purpose |
|------|--------|---------|
| 1 | `verify_env.py` | Validate Python packages, MongoDB, Ollama, Open-Meteo |
| 2 | `seed_mongo.py` | Load `farms.json`, `ktda_pricing.json`, `synthetic_metadata.json` into MongoDB |
| 3 | `weather_fetch.py` | Fetch historical weather from Open-Meteo API → `weather_history` collection |

### Phase 2 — ML Model Training (run in `ml_engine` Docker container)
| Step | Script | Output |
|------|--------|--------|
| 4 | `preprocess.py` | Feature matrix `X_features.parquet`, target `y_target.parquet`, `preprocessor.pkl`, `row_ids.parquet` |
| 5a | `train_xgboost.py` | `xgb_yield.pkl`, `xgb_feature_importances.parquet`, `xgb_cv_results.parquet` |
| 5b | `train_salima.py` | Per-farm SARIMA models (`models/sarima/{factory}/{member_no}.pkl`) + pricing SARIMA |
| 6a | `build_graph.py` | Farm graph structure → `farm_graph_v1` document (nodes + edges) |
| 6b | `train_gnn.py` | GNN embeddings, clusters, anomaly scores → `gnn_outputs` collection |

### Phase 3 — Inference & Deployment
| Script | Purpose |
|--------|---------|
| `rule_engine.py` | Deterministic recommendation engine (XGB + SARIMA + GNN + pricing rules) |
| `upload_models_to_mongo.py` | Push trained artifacts to MongoDB GridFS for Render deployment |
| `benchmark_regression.py` | Model-native evaluation: SARIMA (expanding window CV), XGBoost (TimeSeriesSplit), GNN (cluster quality) |

### Phase 4 — API Server
- **Flask app** (`api/app.py`) with JWT auth, 4 blueprints: `auth`, `farms`, `insights`, `pricing`
- Deployed on **Render** (stateless, loads models from MongoDB GridFS at startup)

---

## 2. Database Structure Implementation

### MongoDB Collections

| Collection | Documents | Key Indexes | Purpose |
|------------|-----------|-------------|---------|
| **farms** | 240 | `ktda_member_no` (unique), `factory_code+collection_centre`, `factory_code` | Core farm registry with historical seasons, GPS, fertiliser applications |
| **ktda_pricing** | ~200 | `season_year+season_month_idx+factory_code`, `factory_code` | Monthly tea rates (monthly_rate, minibonus_rate, annual_bonus_rate) |
| **factory_metadata** | 1 | `_type` (unique) | Single config doc: factories, collection centres, agronomic params, rule thresholds |
| **weather_history** | ~30 | `collection_centre+year` (unique) | Open-Meteo monthly rainfall/temp per centre per year |
| **farm_graph_v1** | 1 | — | Graph document: nodes (27-dim features), edges (3 types), built by `build_graph.py` |
| **gnn_outputs** | 240 | `ktda_member_no` (implicit) | Per-farm: `cluster_id`, `anomaly_score`, `embedding` (32-dim) |
| **model_outputs** | 240 | `ktda_member_no` (unique), `last_computed` | Cached inference results from `/insights` endpoint |
| **admins** | 1+ | `username` (unique) | Admin credentials for dashboard access |
| **model_files** (GridFS) | ~6 files | — | Binary model artifacts: `xgb_yield.pkl`, `preprocessor.pkl`, parquet files |
| **sarima_meta** | ~200 | `ktda_member_no+factory_code` | Per-farm SARIMA metadata (order, AIC, forecast, CI) |
| **sarima_pricing_meta** | 6 | `factory_code+label` | Factory-level pricing SARIMA metadata |

### Data Flow
```
farms.json + ktda_pricing.json + synthetic_metadata.json
         ↓ seed_mongo.py
MongoDB collections (farms, ktda_pricing, factory_metadata)
         ↓ weather_fetch.py (external API)
weather_history
         ↓ preprocess.py (reads farms, weather, metadata)
X_features.parquet, y_target.parquet, preprocessor.pkl, row_ids.parquet
         ↓ train_xgboost.py
xgb_yield.pkl, xgb_feature_importances.parquet
         ↓ train_salima.py
models/sarima/{factory}/{member_no}.pkl + _meta.json
         ↓ build_graph.py (reads farms)
farm_graph_v1 (nodes + edges)
         ↓ train_gnn.py
gnn_outputs collection (cluster_id, anomaly_score, embedding)
         ↓ upload_models_to_mongo.py
MongoDB GridFS + sarima_meta + sarima_pricing_meta
         ↓ API (Flask) loads from GridFS at startup
Production inference
```

---

## 3. Security Considerations in the Code

### ✅ Current Protections

| Layer | Implementation |
|-------|----------------|
| **Authentication** | JWT tokens via `flask_jwt_extended`; `JWT_SECRET_KEY` from env (defaults to dev value) |
| **Authorization** | `@jwt_required()` on all blueprint routes; admin check via `get_jwt().get("role") == "admin"` |
| **Input Validation** | Explicit validation in `farms.py` POST `/daily` (date format, kg ≥ 0, type checks) |
| **Secrets Management** | All secrets via environment variables: `MONGODB_URI`, `JWT_SECRET_KEY`, `GROQ_API_KEY`, `MODELS_DIR` |
| **Password Handling** | Phase 1 demo: farmer password = member_no (explicitly documented as temporary); admin has separate password |
| **Error Handling** | Generic 500 handler doesn't leak stack traces to client (only `"internal server error"`) |
| **CORS** | Enabled globally (`flask_cors.CORS(app)`) — should be restricted to specific origins in prod |
| **GridFS** | Model files stored in MongoDB, not on container filesystem (stateless deployments) |

### ⚠️ Known Gaps (Marked for Production)

| Issue | Location | Mitigation Needed |
|-------|----------|-------------------|
| **Plaintext farmer passwords** | `auth.py` line 55: `if password != member_no:` | Replace with bcrypt/argon2 hash + salt; add `users` collection |
| **JWT never expires** | `app.py` line 21: `JWT_ACCESS_TOKEN_EXPIRES = False` | Set reasonable expiry (e.g., 24h) + refresh tokens |
| **Default JWT secret** | `app.py` line 20: `"dev-secret-change-in-prod"` | Generate strong random secret for production |
| **CORS allows all origins** | `app.py` line 17: `CORS(app)` | Restrict to `ALLOWED_ORIGINS` env var |
| **No rate limiting** | — | Add `flask-limiter` on `/auth/login` and `/farms/<id>/daily` |
| **Admin password in seed** | `seed_mongo.py` line 81: `"password": "farmer1"` | Use env var or secrets manager; force change on first login |
| **Groq API key in config** | `app.py` line 25: `GROQ_API_KEY` from env | OK (env-based), but ensure not committed to repo |
| **No HTTPS enforcement** | — | Behind Render TLS termination — OK for now |
| **MongoDB connection no TLS** | `utils.py` line 30: `MongoClient(uri)` | Use `mongodb+srv://` with TLS for Atlas; verify certs |

### Data Privacy Notes
- Farm data includes PII: `owner_name`, `ktda_member_no`, GPS coordinates
- GDPR/Kenya Data Protection Act compliance needed before real farmer data
- `farm_summary()` in `utils.py` explicitly selects safe fields for list endpoints

---

## 4. How Each Module Works

### 4.1 Data Ingestion Layer

#### `scripts/seed_mongo.py`
- **Purpose**: Idempotent bulk load of three JSON files into MongoDB
- **Key Logic**:
  - `load_json_array()` handles both `[...]` and `{"farms": [...]}` formats
  - `insert_many(ordered=False)` continues on duplicate key errors
  - `replace_one(upsert=True)` for metadata (single-document collection)
  - Creates 6 indexes: unique member_no, compound factory+centre, pricing lookup, weather unique, model_outputs unique
- **Admin seeding**: Creates `ktda_admin` / `farmer1` in `admins` collection

#### `scripts/weather_fetch.py`
- **Purpose**: Fetch 2010–2024 monthly weather from Open-Meteo archive API
- **Key Logic**:
  - Loads centres from `synthetic_metadata.json` (with fallback)
  - Seasons are **July–June** anchored: fetches `year` (Jul–Dec) + `year+1` (Jan–Jun)
  - `monthly_rainfall_mm[6:] + next_year[:6]` → season-aligned 12-month array
  - Upserts into `weather_history` with unique index on `(centre, year)`
  - Rate-limited: 0.4s between calls

---

### 4.2 Feature Engineering — `scripts/preprocess.py`

**Core Function**: `build_rows()` flattens every farm × season × month → 1 row per observation

| Feature Category | Features | Derivation |
|-----------------|----------|------------|
| **Static farm** | hectares, altitude_m, years_active, fairtrade | From farm doc |
| **Weather** | rainfall_mm, temp_c, rain_deficit_mm, rain_3mo_sum | Open-Meteo + synthetic fallback; deficit = rain - 115mm optimum |
| **Fertiliser** | fert_kg, manure_kg, fert_lag1_kg, fert_lag2_kg | Split by type (CAN/NPK vs Manure); lag = prior months |
| **Pruning** | is_pruning_month, months_since_pruning | Heuristic: yield ≤ 50% of expected monthly (base_yield × ha / 12) |
| **Seasonality** | is_minibonus_month, is_annual_bonus, season_month_sin/cos | Fixed calendar (Jul=0...Jun=11); cyclic encoding |
| **Rolling** | rolling_yield_3mo, prior_season_avg | 3-month window; expanding mean of same month in prior seasons |

**Preprocessor**: `ColumnTransformer` → `StandardScaler` (numeric) + `OrdinalEncoder` (factory_code, collection_centre)

**Outputs**: Parquet files + `preprocessor.pkl` (fitted transformer)

---

### 4.3 XGBoost Training — `scripts/train_xgboost.py`

| Component | Configuration |
|-----------|---------------|
| **Model** | `XGBRegressor`: 400 estimators, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, tree_method="hist" |
| **CV** | 5-fold `KFold(shuffle=True, random_state=42)` |
| **Metrics** | MAE, RMSE, MAPE (mask y > 10 to avoid division instability) |
| **Target** | MAE < 15% of mean yield |
| **Refit** | Full dataset after CV |
| **Outputs** | `xgb_yield.pkl`, `xgb_feature_importances.parquet`, `xgb_cv_results.parquet` |

---

### 4.4 SARIMA Training — `scripts/train_salima.py`

#### Per-Farm Yield Models
- **Input**: Monthly yield series (July-anchored, concatenated across seasons)
- **Algorithm**: `pmdarima.auto_arima` with seasonal m=12, stepwise AIC search
- **Constraints**: max_p=3, max_q=3, max_P=2, max_Q=2; needs ≥24 months
- **Output**: 6-month forecast + 80%/95% CI; saved as `.pkl` + `_meta.json`
- **Manifest**: `farm_manifest.parquet` tracks status (ok/skipped/failed) per farm

#### Factory Pricing Models (3 per factory)
- **Series**: monthly_rate (all months), minibonus_rate (Jul/Aug/Sep/Nov/Dec), annual_bonus (Jun only)
- **Seasonality**: m = min(12, n_obs // 2)
- **Output**: 12-period forecast + CI; saved to `models/sarima/pricing/`

---

### 4.5 Graph Construction — `scripts/build_graph.py`

**Nodes** (240 farms, 27 features each):
| Feature Group | Dims | Description |
|---------------|------|-------------|
| Season yield means | 15 | Mean monthly kg per season (padded to 15 seasons) |
| Agronomic scalars | 12 | overall_yield_mean, yield_trend_slope, yield_std, seasonal_amplitude, pruning_recovery_rate, rainfall_response_beta, fertiliser_efficiency, hectares, altitude_m, avg_rainfall_mm, years_active, fairtrade |

**Edges** (3 rules, max weight wins):
| Rule | Condition | Weight | Type |
|------|-----------|--------|------|
| Same collection centre | `centre_i == centre_j` | 1.0 | `same_centre` |
| Same factory | `factory_i == factory_j` | 0.6 | `same_factory` |
| GPS proximity | haversine < 5km | `min(0.8, 1.0 / (dist_km * 10))` | `gps_proximity` |

**Alpha tuning**: `--alpha` blends geographic vs feature similarity (default 0.6 = geographic bias)

**Output**: Single document `farm_graph_v1` with nodes, edges, edge_types, built_at, status="built"

---

### 4.6 GNN Training — `scripts/train_gnn.py`

**Architecture**: 2-layer GraphSAGE (PyTorch Geometric)
- **Input**: 27-dim normalized node features
- **Hidden**: 32-dim (configurable)
- **Output**: 32-dim embeddings
- **Loss**: Graph autoencoder (unsupervised)
  - Positive: cosine similarity of connected nodes (maximize)
  - Negative: random unconnected pairs (margin 0.5, minimize)
- **Training**: 200 epochs default, Adam lr=0.01, weight_decay=5e-4

**Post-processing**:
1. **K-means clustering** on embeddings → `cluster_id` (default 6 clusters = 3 per factory)
2. **Anomaly score**: Distance from cluster centroid, normalized 0–1 within cluster

**Outputs**:
- `gnn_outputs` collection: per-farm `cluster_id`, `anomaly_score`, `embedding`, `computed_at`
- `gnn_norm_params.pkl`: feature means/stds for inference normalization
- `farm_graph_v1` updated with `status="trained"`

---

### 4.7 Rule Engine — `scripts/rule_engine.py` & `api/blueprints/insights.py`

**Deterministic pipeline** (no ML inference here — uses pre-trained models):

| Stage | Logic |
|-------|-------|
| **XGBoost prediction** | Load feature row for farm+month → preprocess → predict |
| **SARIMA forecast** | Load per-farm `_meta.json` → 6-month forecast + CI |
| **Pricing forecast** | Load factory pricing `_meta.json` → 3 rate forecasts |
| **Current season** | Summarize latest historical season (total, avg, last month) |
| **Recommendations** | Rule-based (see below) |

**Recommendation Rules** (priority: high > medium > low):

| Category | Trigger | Threshold (configurable via `factory_metadata.rule_thresholds`) |
|----------|---------|---------------------------------------------------------------|
| **yield** | XGB predicted vs actual avg gap | `yield_gap_warn_pct=15%`, `yield_gap_alert_pct=30%` |
| **yield** | SARIMA 6mo forecast drop vs last month | `forecast_drop_pct=20%` |
| **pricing** | Minibonus month approaching (next 3 months) | Calendar-based (Jul/Aug/Sep/Nov/Dec) |
| **input** | Fertiliser in XGB top 5 features | Feature importance rank |
| **anomaly** | GNN anomaly_score | `gnn_anomaly_high=0.75`, `gnn_anomaly_medium=0.50` |

**Performance Score**: 80 − 20×high_priority − 8×medium_priority (clamped 0–100)

**Groq LLM Narrative** (optional, `?narrative=true`):
- Compresses pipeline result to ~150-word farmer summary
- Model: `qwen/qwen3-32b` (configurable)
- Strips `<think>` tags from reasoning models

---

### 4.8 Model Upload — `scripts/upload_models_to_mongo.py`

- **GridFS uploads** (large binary): `xgb_yield.pkl`, `preprocessor.pkl`, parquet files, `gnn_norm_params.pkl`
- **SARIMA metadata uploads** (JSON → collections):
  - `sarima_meta`: per-farm metadata (member_no, factory_code, order, AIC, forecast, CI)
  - `sarima_pricing_meta`: factory pricing metadata (factory_code, label, order, AIC, forecast)
- **Idempotent**: deletes existing GridFS file before re-upload; `replace_one(upsert=True)` for metadata

---

### 4.9 Benchmarking — `scripts/benchmark_regression.py`

**Model-native evaluation** (each model evaluated in its proper paradigm):

| Model | Method | Metrics |
|-------|--------|---------|
| **SARIMA** | Expanding-window time-series CV (train on seasons < S, test on S) | RMSE, MSE, R², MAE, MAPE per horizon (1, 3, 6 months) |
| **XGBoost** | TimeSeriesSplit CV (5 splits, no leakage) + holdout test (2023-2024) | Same metrics; per-farm breakdown |
| **GNN** | Cluster quality + anomaly analysis (unsupervised) | Silhouette, Davies-Bouldin, Calinski-Harabasz; anomaly score distribution; correlation with yield variability |

**CLI options**: `--max-farms N` (faster SARIMA), `--skip-*`, `--test-seasons`, `--horizons`

---

### 4.10 API Layer — `api/app.py` + Blueprints

**App Factory** (`create_app()`):
- Config from env: `JWT_SECRET_KEY`, `MONGODB_URI`, `MODELS_DIR`, `GROQ_API_KEY`, `GROQ_MODEL`
- JWT: access tokens non-expiring (dev), `JWTManager` init
- CORS enabled globally
- Blueprints registered: `/auth`, `/farms`, `/farms` (insights), `/pricing`
- Error handlers: 404, 500 (generic), `/health` endpoint

#### `api/blueprints/auth.py`
- **POST /auth/login**: body `{ktda_member_no, password}`
  - Checks `admins` collection first (username/password/role)
  - Falls back to `farms` collection (password = member_no for demo)
  - Returns JWT with `identity=member_no` and `role` claim for admins
- **GET /auth/me**: validates token, returns identity

#### `api/blueprints/farms.py`
- **GET /farms**: paginated, filterable (factory, centre, search), returns `farm_summary()`
- **GET /farms/all**: admin-only, minimal fields for dropdowns
- **GET /farms/<member_no>**: full farm doc + season summaries (totals, monthly arrays for charts)
- **POST /farms/<member_no>/daily**: append daily weighment; invalidates `model_outputs` cache

#### `api/blueprints/insights.py`
- **GET /farms/<member_no>/insights**: full pipeline
  - Query params: `refresh=true`, `narrative=true`, `month_idx=0-11`
  - Cache check: `model_outputs` collection (skip if fresh & no narrative)
  - Runs: XGB prediction → SARIMA forecast → pricing forecast → GNN → rule engine → score
  - Optional Groq narrative
  - Caches result in `model_outputs` with `computed_at`, `elapsed_seconds`

#### `api/blueprints/pricing.py`
- **GET /pricing/trends/<factory_code>**: historical rates + SARIMA forecasts (monthly, minibonus, annual bonus)
- **GET /pricing/centres**: factory + collection centre structure from metadata

#### `api/utils.py`
- **MongoDB connection**: singleton `_mongo_client`, `get_db()`
- **Model loading**: `get_models()` loads from GridFS once per process → caches in `_models` dict
  - Loads: xgb, preprocessor, importances, X, row_ids
- **SARIMA metadata**: `get_sarima_meta(member_no, factory_code)` → MongoDB first, disk fallback
- **Pricing metadata**: `get_pricing_meta(factory_code)` → 3 labels
- **GNN outputs**: `get_gnn_output(member_no)`, `get_cluster_peers(member_no, cluster_id)`
- **Response helper**: `farm_summary()` — safe fields for list endpoints

---

## 5. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **July–June season anchoring** | Matches KTDA tea season; all time-series aligned to this calendar |
| **Per-farm SARIMA (not global)** | Each farm has unique yield pattern; 240 small models > 1 large model |
| **GridFS for model artifacts** | Render has no persistent disk; models loaded from MongoDB at container startup |
| **Unsupervised GNN (GraphSAGE + K-means)** | No labeled anomaly data; structural similarity + clustering finds outliers naturally |
| **Deterministic rule engine + optional LLM** | Auditable recommendations; LLM only for narrative, not logic |
| **TimeSeriesSplit for XGBoost** | Prevents temporal leakage (standard KFold would leak future → past) |
| **Expanding window for SARIMA** | Mirrors production: train on history, forecast forward |
| **Feature store in MongoDB (not files)** | Single source of truth; versioned via upserts; works across containers |

---

## 6. Running the Full Pipeline

```bash
# 1. Verify environment
python scripts/verify_env.py --host-checks

# 2. Seed MongoDB (with --drop for clean slate)
python scripts/seed_mongo.py --drop

# 3. Fetch weather (run on WSL host, not Docker)
python scripts/weather_fetch.py

# 4. Preprocess (in ml_engine container)
docker compose exec ml_engine python scripts/preprocess.py --validate

# 5a. Train XGBoost
docker compose exec ml_engine python scripts/train_xgboost.py --validate

# 5b. Train SARIMA (farm + pricing)
docker compose exec ml_engine python scripts/train_salima.py --validate

# 6a. Build graph
python scripts/build_graph.py --validate

# 6b. Train GNN
python scripts/train_gnn.py --validate

# 7. Upload models to MongoDB (for Render deployment)
python scripts/upload_models_to_mongo.py

# 8. Run benchmark
python scripts/benchmark_regression.py --max-farms 10

# 9. Start API
cd api && python app.py
# or: flask --app app run
```

---

## 7. API Endpoint Summary

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/auth/login` | No | Get JWT (member_no as password for farmers) |
| GET | `/auth/me` | Yes | Validate token |
| GET | `/farms` | Yes | Paginated farm list with filters |
| GET | `/farms/all` | Admin | All farms for dropdowns |
| GET | `/farms/<member_no>` | Yes | Farm detail + season history |
| POST | `/farms/<member_no>/daily` | Yes | Record daily weighment |
| GET | `/farms/<member_no>/insights` | Yes | Full ML pipeline + recommendations |
| GET | `/pricing/trends/<factory_code>` | Yes | Historical + forecast rates |
| GET | `/pricing/centres` | Yes | Factory/centre hierarchy |

---

## 8. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MONGODB_URI` | Yes | `mongodb://localhost:27017/chaimterics` | MongoDB connection string |
| `JWT_SECRET_KEY` | Yes | `dev-secret-change-in-prod` | JWT signing secret |
| `GROQ_API_KEY` | For narrative | — | Groq API key for LLM narratives |
| `GROQ_MODEL` | No | `qwen/qwen3-32b` | Groq model ID |
| `MODELS_DIR` | No | `/app/models` | Local model artifacts path |
| `DATA_DIR` | No | `../data` | Data files path |
| `OLLAMA_HOST` | For Ollama | `http://localhost:11434` | Ollama API endpoint |

---

*Generated from codebase analysis — update when architecture changes.*