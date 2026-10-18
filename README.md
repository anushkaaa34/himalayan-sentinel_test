# Himalayan Sentinel — Hackathon Submission

A research prototype that keeps **historical disaster records separate from current environmental model scores**, uses the supplied XGBoost model, and optionally sends compact score summaries to Gemini.

## Run locally

```bash
cd himalayan_sentinel_submission
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Create .env and add GEMINI_API_KEY (optional), GOOGLE_MAPS_API_KEY, and GEE_PROJECT as needed.
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## Data flow
1. Historical register: event rows from `data/master_candidate_dataset.csv`; controls are excluded. These records are displayed as past catalogue candidates and are **never predicted**.
2. Live scoring: queries 28 completed days of ERA5-Land daily aggregates through Google Earth Engine at 12 fixed Himalayan sampling points, with CHIRPS precipitation as a separate cross-check.
3. Feature engineering: builds the model's 75 input columns in the same order as `model/feature_columns.json`; no event dates, labels, IDs, or coordinates are sent to the model.
4. Prediction: calls the supplied XGBoost model's `predict_proba`.
5. Gemini: optional Gemini 2.5 Flash synthesis. Gemini only summarizes the returned scores; it cannot change the model outputs.

## Configure Gemini
Add `GEMINI_API_KEY` to `.env` to enable analyst briefs. Never commit API keys. The UI works without Gemini, but generating the analyst brief requires a valid key and network access.

## Configure Google Maps
Set `GOOGLE_MAPS_API_KEY` in `.env` for local development and as a Cloud Run environment variable when deployed. The interactive map uses the Google Maps JavaScript API, which requires an API key with Maps JavaScript API enabled. Restrict the key to your app's website referrer and the Maps JavaScript API. The key is sent to the browser for map loading, so it is not a server-only secret.

## Important model/data caveats
- The provided model is an XGBoost classifier expecting 75 numeric features. The feature order is derived from the numeric environmental columns in the supplied master CSV after excluding labels, identifiers, coordinates, and event metadata. The model artifact itself does not contain human-readable feature names.
- The dashboard reports retrospective discrimination on the CSV's predefined test split (232 rows in 116 matched pairs). Pair, canonical-event, incident, episode and spatial groups do not cross the provided splits. However, the saved model does not identify which rows it was fitted on, so test-set exclusion from model training is not independently verifiable; treat these as provisional, not certified held-out performance. Because the test set is a balanced case-control sample, average precision and the observed event prevalence do not represent operational alert prevalence.
- Live feature parity is **not fully verified**. The code maps ERA5-Land daily temperature, precipitation, runoff, snowmelt and top-layer soil moisture into the dataset's four 7-day feature windows. Provider variable definitions and units can differ from the training source. Check `data_completeness`, compare feature ranges with training rows, and validate before treating scores as meaningful.
- Archive data may lag by several days. The app labels the actual last observation date instead of claiming real-time measurements.
- Scores are raw model outputs, **not calibrated probabilities** or official warnings. The score bands are UI triage bands only and are not validated alert thresholds.
- The historical dataset contains 629 event candidates and 629 matched controls. Event rows include unreviewed/provisional records; UI labels them as catalogue candidates and preserves source links where present.
- This is not a certified operational warning system. Do not use it for emergency decisions.
- Tiles and external API access require internet. For production/hackathon demo reliability, prefetch data before presenting and keep provider errors visible.
- If every live site query fails, `/api/live` returns a deterministic offline replay made from examples in the provided historical test split. The UI marks it as **OFFLINE DEMO REPLAY** and explains that its event/control examples are historical, not live sites or warnings.

## Useful endpoints
- `GET /api/health` — model, feature count and dataset row count
- `GET /api/history?years=10` — historical event candidates only
- `GET /api/live` — current scores and provider errors
- `GET /api/evaluation` — retrospective metrics on the dataset's predefined test split, with limitations
- `GET /api/demo-replay` — deterministic offline replay from historical test-set examples
- `POST /api/gemini` — Gemini summary of live results

## Suggested 90-second demo
1. Start with the evaluation panel: describe the 116 matched-pair test split, ROC-AUC and its pair-bootstrap interval, then immediately disclose that artifact training provenance is unverified and the sample is case-control balanced.
2. Refresh live data; open a site popup and point out observation date, completeness and that scores are experimental—not calibrated probabilities or official warnings.
3. Toggle the historical layer to distinguish catalogued past events from live environmental model outputs.
4. If live providers fail, use the clearly marked offline replay and explain that these are dated test-set examples, not current conditions.
5. Generate the Gemini brief as a narrative summary only; finish with feature-parity validation, calibration and operational evaluation as next steps.

## Earth Engine setup
The app initializes Earth Engine using `GEE_PROJECT` (default `ai-builder-cup-510010`). On this Mac, authenticate once in the same Python environment if needed with `earthengine authenticate`, then restart the app. The live endpoint reports initialization failures explicitly. ERA5-Land supplies model inputs; CHIRPS is a rainfall cross-check and does not replace ERA5 precipitation features.
