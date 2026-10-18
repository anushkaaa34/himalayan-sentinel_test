# Himalayan Sentinel

An environmental-intelligence prototype for the Sustainability & Social Impact theme. It brings together historical landslide catalogue candidates and separately computed environmental model scores across illustrative Himalayan sites. The aim is to support exploration and informed discussion—not to issue official warnings or direct emergency decisions.

## Features

- Interactive Google Map with separate live-score and historical-record layers.
- XGBoost scoring from a 28-day environmental window: ERA5-Land through Google Earth Engine, with CHIRPS precipitation as a cross-check.
- Retrospective metrics from the CSV's predefined test split.
- Optional Gemini analyst brief that summarizes results without changing model scores.
- Clearly marked offline replay for presentations when live site queries are unavailable.

## Run locally

Requirements: Python 3.10+ and internet access for external map/data/AI services.

```bash
cd himalayan_sentinel_submission
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root. Add the API keys and Earth Engine project you need:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
GEE_PROJECT=your_google_cloud_project_id
```

The Gemini and Google Maps keys are optional for starting the app, but the Gemini brief and interactive Google Map need their respective keys. Never commit `.env`; `.gcloudignore` excludes it from source deployments.

Initialize Earth Engine credentials for local use if needed, then start the server:

```bash
earthengine authenticate
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>.

## Deploy to Google Cloud Run (no Dockerfile)

The service has been deployed from source using Google Cloud Buildpacks. From the project root:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com earthengine.googleapis.com
gcloud run deploy himalayan-sentinel \
  --source . \
  --region asia-south1 \
  --command="" \
  --args="" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --set-env-vars=GEE_PROJECT=YOUR_PROJECT_ID \
  --allow-unauthenticated
```

The empty `--command` and `--args` clear any previously saved container entrypoint overrides so the Buildpacks-provided Uvicorn startup command is used. Keep the root `.gcloudignore` in place so `.env`, `.venv`, and local files aren't uploaded.

After deployment, configure runtime keys in **Cloud Run → service → Edit and deploy new revision → Variables & Secrets**:

- `GOOGLE_MAPS_API_KEY`: Maps JavaScript API key. This is delivered to browsers to load the map; restrict it to the Maps JavaScript API and your service's HTTPS referrer, for example `https://YOUR-SERVICE-URL/*`.
- `GEMINI_API_KEY`: Gemini API key. Prefer storing it in Secret Manager and mounting it as an environment variable on the service.
- `GEE_PROJECT`: the Cloud project registered for Earth Engine.

Deploy the new revision after changing variables or secrets. The Cloud Run runtime service account must be authorized to use Earth Engine in the configured project. If you choose `--allow-unauthenticated`, anyone with the service URL can access the app.

## Data flow

1. **Historical register:** reads event rows from `data/master_candidate_dataset.csv`; matched controls are excluded. These records are shown as past catalogue candidates and are never passed to the predictor.
2. **Environmental data:** queries 28 completed days of ERA5-Land daily aggregates at 12 fixed illustrative points. CHIRPS rainfall is a cross-check, not a replacement model input.
3. **Feature engineering:** maps provider observations into the 75-column order in `model/feature_columns.json`. Event labels, IDs, and coordinates are not passed into the model.
4. **Scoring:** calls the supplied XGBoost classifier's `predict_proba`. Scores are raw model outputs, not calibrated probabilities.
5. **Gemini:** optionally summarizes compact results. Gemini is a narrative layer only and cannot alter the model outputs.

## Model evaluation and limitations

- `/api/evaluation` reports retrospective metrics for the CSV's predefined test split: 232 rows comprising 116 events and 116 matched controls. Pair, canonical-event, incident, episode, and spatial groups do not cross the provided splits.
- The current model artifact does not contain a record of which rows were used to fit it. The model's exclusion of the test rows from training therefore cannot be independently verified from this repository. Treat the displayed results as provisional until training provenance is documented.
- The test split is a balanced case-control sample. Its event prevalence and average precision are not representative of real-world alert prevalence. The 0.50 classification cutoff shown for illustration is not an alert threshold.
- Live feature parity is not fully verified. Provider definitions and units may differ from those used to create the training features; validate feature distributions and model behavior before interpreting scores.
- The catalogue includes unreviewed/provisional event candidates. Individual records may require verification.
- Environmental archives may lag. The app reports the actual observation date and data completeness where available.
- This is not a certified operational warning system. Do not use it for emergency decisions.

## Offline demonstration

Select **Offline demo** in the dashboard to load a deterministic set of 12 examples from the dataset's designated test split. If all live site queries fail, `/api/live` automatically returns the same replay. The dashboard labels this mode **OFFLINE DEMO REPLAY** and gives the historical date range. These examples are historical records—not live conditions, current sites, forecasts, or warnings. Gemini briefs in this mode are also instructed to describe the data as historical.

## API endpoints

- `GET /api/health` — model, feature count, dataset rows, and Earth Engine initialization status.
- `GET /api/history?years=10` — historical event candidates only.
- `GET /api/live` — live site scores, provider errors, or labeled offline fallback.
- `GET /api/evaluation` — retrospective test-split metrics and evaluation caveats.
- `GET /api/demo-replay` — deterministic offline replay examples.
- `POST /api/gemini` — optional Gemini summary of supplied results.

## Suggested 90-second demo

1. Introduce the community resilience problem and state clearly that this is a research prototype, not an alert service.
2. Show the retrospective evaluation and disclose the model-provenance and matched-sample limitations.
3. Refresh live data and inspect an example site's observation date, completeness, and raw score.
4. Toggle historical catalogue events to show how past records remain separate from current environmental scoring.
5. If providers fail, use the clearly labeled Offline demo replay; explain that its examples are historical.
6. Generate the Gemini brief as a narrative summary, then close with the validation needed before real-world use.

## Earth Engine setup

The app initializes Earth Engine with `GEE_PROJECT`. The default is `ai-builder-cup-510010`; set the variable to the intended Google Cloud project when deploying elsewhere. Local development may require `earthengine authenticate`. On Cloud Run, use the runtime service account and ensure the project is registered for Earth Engine with the required permissions. ERA5-Land provides model inputs; CHIRPS is a separate rainfall cross-check.
