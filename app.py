import os, json, math, traceback
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import joblib
import ee
from html import escape
from urllib.parse import quote

ROOT=Path(__file__).resolve().parent
load_dotenv(ROOT/".env")
GEE_PROJECT=os.getenv("GEE_PROJECT", "ai-builder-cup-510010")
try:
    ee.Initialize(project=GEE_PROJECT)
    GEE_READY=True
    GEE_ERROR=None
except Exception as _gee_exc:
    GEE_READY=False
    GEE_ERROR=str(_gee_exc)
MODEL_PATH=ROOT/"model"/"himalayan_final_xgboost.pkl"
DATA_PATH=ROOT/"data"/"master_candidate_dataset.csv"
FEATURE_PATH=ROOT/"model"/"feature_columns.json"
with open(FEATURE_PATH) as f: FEATURES=json.load(f)
model=joblib.load(MODEL_PATH)
df=pd.read_csv(DATA_PATH, low_memory=False)
app=FastAPI(title="Himalayan Sentinel", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Illustrative monitoring grid across the Hindu Kush–Himalaya. Coordinates are fixed sampling points,
# not official warning stations. Forecast scores are research outputs, not public safety alerts.
SITES=[
 {"id":"west","name":"Kashmir Himalaya","lat":34.15,"lon":75.35,"region":"Western Himalaya"},
 {"id":"hp","name":"Himachal Himalaya","lat":31.55,"lon":77.20,"region":"Western Himalaya"},
 {"id":"utt","name":"Garhwal Himalaya","lat":30.45,"lon":79.30,"region":"Central Himalaya"},
 {"id":"nepal","name":"Central Nepal","lat":28.25,"lon":84.10,"region":"Central Himalaya"},
 {"id":"sikkim","name":"Sikkim Himalaya","lat":27.55,"lon":88.45,"region":"Eastern Himalaya"},
 {"id":"arun","name":"Eastern Himalaya","lat":27.10,"lon":92.20,"region":"Eastern Himalaya"},
 {"id":"bhutan","name":"Bhutan Himalaya","lat":27.45,"lon":90.45,"region":"Eastern Himalaya"},
 {"id":"ladakh","name":"Ladakh Range","lat":34.35,"lon":77.50,"region":"Trans-Himalaya"},
 {"id":"kumaon","name":"Kumaon Himalaya","lat":29.75,"lon":80.10,"region":"Central Himalaya"},
 {"id":"nepal_e","name":"Eastern Nepal","lat":27.55,"lon":87.20,"region":"Central Himalaya"},
 {"id":"meghalaya","name":"Meghalaya Hills","lat":25.55,"lon":91.80,"region":"Himalayan foothills"},
 {"id":"karakoram","name":"Karakoram","lat":35.40,"lon":76.50,"region":"Trans-Himalaya"},
]
EXCLUDE={"candidate_label","event_id","objectid","latitude","longitude","pair_group_id","matched_event_key","canonical_event_id","nearest_catalogue_event_days"}
DAILY_NAMES=["temperature_2m_mean","temperature_2m_min","temperature_2m_max","precipitation_sum","runoff_sum","snowmelt_sum","soil_moisture_0_to_7cm_mean"]

def risk_band(p):
    if p >= .70: return "High model score"
    if p >= .40: return "Raised model score"
    return "Lower model score"

def make_features(daily):
    # daily rows chronological oldest -> newest. Feature definitions match the master dataset's
    # four 7-day aggregates. Temperature is converted C -> K and precipitation/runoff/snowmelt
    # mm -> metres, consistent with the observed training feature ranges.
    d=daily.copy()
    for col in DAILY_NAMES:
        if col not in d: d[col]=np.nan
    d["temperature_2m_mean"] = d["temperature_2m_mean"] + 273.15
    for col in ["temperature_2m_min","temperature_2m_max"]:
        d[col] = d[col] + 273.15
    for col in ["precipitation_sum","runoff_sum","snowmelt_sum"]:
        d[col] = d[col] / 1000.0
    windows=[d.iloc[i*7:(i+1)*7] for i in range(4)]
    f={}
    f["temperature_valid_days_28d"]=int(d["temperature_2m_mean"].notna().sum())
    f["precip_valid_days_28d"]=int(d["precipitation_sum"].notna().sum())
    f["runoff_valid_days_28d"]=int(d["runoff_sum"].notna().sum())
    f["snowmelt_valid_days_28d"]=int(d["snowmelt_sum"].notna().sum())
    f["soil_water_valid_days_28d"]=int(d["soil_moisture_0_to_7cm_mean"].notna().sum())
    for i,w in enumerate(windows,1):
        f[f"temperature_valid_days_w{i}"]=int(w["temperature_2m_mean"].notna().sum())
        f[f"temperature_mean_w{i}"]=w["temperature_2m_mean"].mean()
        f[f"temperature_min_w{i}"]=w["temperature_2m_min"].min()
        f[f"temperature_max_w{i}"]=w["temperature_2m_max"].max()
        f[f"precip_valid_days_w{i}"]=int(w["precipitation_sum"].notna().sum())
        f[f"precip_sum_w{i}"]=w["precipitation_sum"].sum(min_count=1)
        f[f"runoff_valid_days_w{i}"]=int(w["runoff_sum"].notna().sum())
        f[f"runoff_sum_w{i}"]=w["runoff_sum"].sum(min_count=1)
        f[f"snowmelt_valid_days_w{i}"]=int(w["snowmelt_sum"].notna().sum())
        f[f"snowmelt_sum_w{i}"]=w["snowmelt_sum"].sum(min_count=1)
        f[f"soil_water_valid_days_w{i}"]=int(w["soil_moisture_0_to_7cm_mean"].notna().sum())
        f[f"soil_water_mean_w{i}"]=w["soil_moisture_0_to_7cm_mean"].mean()
        f[f"soil_water_min_w{i}"]=w["soil_moisture_0_to_7cm_mean"].min()
        f[f"soil_water_max_w{i}"]=w["soil_moisture_0_to_7cm_mean"].max()
    for col, source in [("temperature_mean_28d","temperature_2m_mean"),("temperature_min_28d","temperature_2m_min"),("temperature_max_28d","temperature_2m_max"),
                        ("precip_sum_28d","precipitation_sum"),("runoff_sum_28d","runoff_sum"),("snowmelt_sum_28d","snowmelt_sum"),
                        ("soil_water_mean_28d","soil_moisture_0_to_7cm_mean"),("soil_water_min_28d","soil_moisture_0_to_7cm_mean"),("soil_water_max_28d","soil_moisture_0_to_7cm_mean")]:
        if col.startswith("temperature_min"): f[col]=d[source].min()
        elif col.startswith("temperature_max"): f[col]=d[source].max()
        elif col.startswith("soil_water_min"): f[col]=d[source].min()
        elif col.startswith("soil_water_max"): f[col]=d[source].max()
        else: f[col]=d[source].sum(min_count=1) if source in ["precipitation_sum","runoff_sum","snowmelt_sum"] else d[source].mean()
    f["temperature_mean_trend_w4_minus_w1"]=f["temperature_mean_w4"]-f["temperature_mean_w1"]
    f["precip_sum_change_w4_minus_w1"]=f["precip_sum_w4"]-f["precip_sum_w1"]
    f["runoff_sum_change_w4_minus_w1"]=f["runoff_sum_w4"]-f["runoff_sum_w1"]
    f["snowmelt_sum_change_w4_minus_w1"]=f["snowmelt_sum_w4"]-f["snowmelt_sum_w1"]
    f["soil_water_mean_trend_w4_minus_w1"]=f["soil_water_mean_w4"]-f["soil_water_mean_w1"]
    return pd.DataFrame([[f.get(c,np.nan) for c in FEATURES]],columns=FEATURES)

def binary_metrics(y, scores):
    y=np.asarray(y,dtype=np.int8)
    scores=np.asarray(scores,dtype=float)
    positives=int(y.sum())
    negatives=len(y)-positives
    if positives == 0 or negatives == 0:
        raise ValueError("Evaluation split must contain both event and control examples")

    order=np.argsort(scores,kind="mergesort")
    sorted_scores=scores[order]
    sorted_ranks=np.arange(1,len(y)+1,dtype=float)
    unique_scores,starts,counts=np.unique(sorted_scores,return_index=True,return_counts=True)
    del unique_scores
    for start,count in zip(starts,counts):
        if count > 1:
            sorted_ranks[start:start+count]=np.mean(sorted_ranks[start:start+count])
    ranks=np.empty(len(y),dtype=float)
    ranks[order]=sorted_ranks
    roc_auc=(ranks[y == 1].sum()-positives*(positives+1)/2)/(positives*negatives)

    thresholds=np.unique(scores)[::-1]
    true_positives=0
    seen=0
    previous_recall=0.0
    average_precision=0.0
    for threshold in thresholds:
        at_threshold=scores == threshold
        true_positives+=int(np.count_nonzero(y[at_threshold] == 1))
        seen+=int(np.count_nonzero(at_threshold))
        recall=true_positives/positives
        average_precision+=(recall-previous_recall)*(true_positives/seen)
        previous_recall=recall
    predicted=scores >= 0.5
    tp=int(np.count_nonzero(predicted & (y == 1)))
    fn=positives-tp
    tn=int(np.count_nonzero(~predicted & (y == 0)))
    fp=negatives-tn
    return {
        "roc_auc":float(roc_auc),
        "average_precision":average_precision,
        "recall_at_0_50":tp/positives,
        "specificity_at_0_50":tn/negatives,
        "confusion_at_0_50":{"tp":tp,"fn":fn,"tn":tn,"fp":fp},
    }

def build_evaluation_and_demo():
    test=df[df["split"].astype(str).str.lower().eq("test")].copy()
    if test.empty:
        raise ValueError("Dataset has no predefined test split")
    y=pd.to_numeric(test["candidate_label"],errors="raise").to_numpy(dtype=np.int8)
    scores=model.predict_proba(test[FEATURES])[:,1]
    metrics=binary_metrics(y,scores)

    group_fields=("pair_group_id","canonical_event_id","incident_group_id","episode_group_id","spatial_group")
    group_disjoint={}
    for field in group_fields:
        if field not in df:
            continue
        group_splits=df.dropna(subset=[field]).groupby(field)["split"].nunique()
        group_disjoint[field]=bool((group_splits <= 1).all())

    pair_ids=test["pair_group_id"].dropna().unique()
    rng=np.random.default_rng(42)
    bootstrap_auc=[]
    pair_indices={pair_id:np.flatnonzero(test["pair_group_id"].to_numpy() == pair_id) for pair_id in pair_ids}
    for _ in range(500):
        sampled_pairs=rng.choice(pair_ids,size=len(pair_ids),replace=True)
        sampled_indices=np.concatenate([pair_indices[pair_id] for pair_id in sampled_pairs])
        bootstrap_auc.append(binary_metrics(y[sampled_indices],scores[sampled_indices])["roc_auc"])

    evaluation={
        **metrics,
        "roc_auc_95_ci":[float(x) for x in np.quantile(bootstrap_auc,[0.025,0.975])],
        "test_rows":int(len(test)),
        "event_rows":int(np.count_nonzero(y == 1)),
        "control_rows":int(np.count_nonzero(y == 0)),
        "matched_test_pairs":int(test["pair_group_id"].nunique()),
        "group_disjoint":group_disjoint,
        "split_name":"Provided dataset test split",
        "threshold_note":"0.50 is an illustrative classification cutoff, not an alert threshold.",
        "limitations":[
            "The saved model does not record its training rows; test-set exclusion from model fitting cannot be independently verified.",
            "The test split is a 1:1 matched case-control sample, so its event prevalence and average precision do not represent real-world alert prevalence.",
            "Historical catalogue labels and live environmental feature parity remain uncertain.",
        ],
    }

    demo=[]
    score_by_index=pd.Series(scores,index=test.index)
    for sample_type in ("event","control"):
        class_rows=test[test["sample_type"].astype(str).str.lower().eq(sample_type)].copy()
        class_rows["_score"]=score_by_index.loc[class_rows.index]
        class_rows=class_rows.sort_values("_score",kind="mergesort")
        positions=np.linspace(0,len(class_rows)-1,min(6,len(class_rows)),dtype=int)
        for _,row in class_rows.iloc[positions].iterrows():
            score=float(row["_score"])
            observed_date=row.get("event_day") or row.get("target_start") or ""
            temperature=row.get("temperature_mean_28d")
            demo.append({
                "id":str(row.get("sample_id")),
                "name":f"DEMO · {'Historical test event' if sample_type == 'event' else 'Matched test control'} · {row.get('country_name') or 'Himalayan sample'}",
                "lat":float(row["latitude"]),
                "lon":float(row["longitude"]),
                "region":str(row.get("spatial_group") or row.get("country_name") or "Historical test sample"),
                "score":score,
                "band":risk_band(score),
                "as_of":str(observed_date)[:10],
                "window_start":str(row.get("forecast_origin") or "")[:10],
                "valid_days":int(row.get("temperature_valid_days_28d") or 0),
                "data_completeness":round(float(np.mean([
                    min(1.0,float(row.get(f"{name}_valid_days_28d",0) or 0)/28)
                    for name in ("temperature","precip","runoff","snowmelt","soil_water")
                ])),3),
                "latest_temp_c":None if pd.isna(temperature) else round(float(temperature)-273.15,1),
                "rain_28d_mm":None if pd.isna(row.get("precip_sum_28d")) else round(float(row["precip_sum_28d"])*1000,1),
                "sample_type":sample_type,
                "demo_only":True,
            })
    return evaluation,demo

EVALUATION,DEMO_RESULTS=build_evaluation_and_demo()

def fetch_site(site):
    if not GEE_READY:
        raise RuntimeError("Google Earth Engine is not initialized: "+str(GEE_ERROR))
    # ERA5-Land daily aggregates are the model's primary weather/hydrology source.
    # Leave the latest 7 days out to reduce archive-latency failures.
    end=date.today()-timedelta(days=7)
    start=end-timedelta(days=27)
    start_s=start.isoformat()
    end_exclusive=(end+timedelta(days=1)).isoformat()
    point=ee.Geometry.Point([site["lon"],site["lat"]])
    era_bands=[
        "temperature_2m","temperature_2m_min","temperature_2m_max",
        "total_precipitation_sum","runoff_sum","snowmelt_sum",
        "volumetric_soil_water_layer_1"
    ]
    collection=(ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start_s,end_exclusive).select(era_bands).sort("system:time_start"))
    # One server-side reduction per date, returned as a single compact FeatureCollection.
    def extract(image):
        image=ee.Image(image)
        values=image.reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=11132,
            bestEffort=True, maxPixels=1e6
        )
        return ee.Feature(None, values).set(
            "date", ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        )
    rows=ee.FeatureCollection(collection.map(extract)).getInfo().get("features",[])
    if len(rows)<20:
        raise RuntimeError(f"ERA5-Land returned only {len(rows)} daily rows for {start_s} to {end.isoformat()}")
    records=[]
    for row in rows:
        props=row.get("properties",{})
        # Model engineering below expects Celsius/mm and converts to Kelvin/metres.
        def val(k):
            x=props.get(k)
            return np.nan if x is None else float(x)
        records.append({
            "time":props.get("date"),
            "temperature_2m_mean":val("temperature_2m")-273.15 if props.get("temperature_2m") is not None else np.nan,
            "temperature_2m_min":val("temperature_2m_min")-273.15 if props.get("temperature_2m_min") is not None else np.nan,
            "temperature_2m_max":val("temperature_2m_max")-273.15 if props.get("temperature_2m_max") is not None else np.nan,
            "precipitation_sum":val("total_precipitation_sum")*1000 if props.get("total_precipitation_sum") is not None else np.nan,
            "runoff_sum":val("runoff_sum")*1000 if props.get("runoff_sum") is not None else np.nan,
            "snowmelt_sum":val("snowmelt_sum")*1000 if props.get("snowmelt_sum") is not None else np.nan,
            "soil_moisture_0_to_7cm_mean":val("volumetric_soil_water_layer_1")
        })
    obs=pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    for c in DAILY_NAMES:
        if c not in obs: obs[c]=np.nan
    X=make_features(obs)
    if X.isna().all(axis=1).iloc[0]:
        raise RuntimeError("All engineered model features are missing")
    p=float(model.predict_proba(X)[0,1])
    latest=obs.iloc[-1]
    # CHIRPS is a separate precipitation cross-check, not a substitute for ERA5-Land features.
    chirps_mm=None
    chirps_error=None
    try:
        chirps=(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(start_s,end_exclusive).select("precipitation"))
        ch_rows=ee.FeatureCollection(chirps.map(lambda im: ee.Feature(None,
            ee.Image(im).reduceRegion(ee.Reducer.first(),point,scale=5566,bestEffort=True,maxPixels=1e6)
        ).set("date",ee.Date(ee.Image(im).get("system:time_start")).format("YYYY-MM-dd")))).getInfo().get("features",[])
        ch_vals=[x.get("properties",{}).get("precipitation") for x in ch_rows]
        ch_vals=[float(x) for x in ch_vals if x is not None]
        chirps_mm=round(sum(ch_vals),1) if ch_vals else None
    except Exception as ex:
        chirps_error=str(ex)[:180]
    return {**site,"score":p,"band":risk_band(p),"as_of":str(obs["time"].iloc[-1]),
        "window_start":str(obs["time"].iloc[0]),
        "valid_days":int(obs["temperature_2m_mean"].notna().sum()),
        "data_completeness":round(float(obs[DAILY_NAMES].notna().mean().mean()),3),
        "latest_temp_c":None if pd.isna(latest["temperature_2m_mean"]) else round(float(latest["temperature_2m_mean"]),1),
        "rain_28d_mm":None if pd.isna(X.iloc[0].get("precip_sum_28d")) else round(float(X.iloc[0]["precip_sum_28d"]*1000),1),
        "chirps_rain_28d_mm":chirps_mm,"chirps_error":chirps_error,
        "provider":"Google Earth Engine: ERA5-Land Daily Aggregates; CHIRPS cross-check",
        "model_type":type(model).__name__}

# @app.get("/",response_class=HTMLResponse)
# def home():
#     return (ROOT/"static"/"index.html").read_text()

@app.get("/", response_class=HTMLResponse)
def home():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    key_literal = json.dumps(maps_key).replace("<", "\\u003c")
    html = html.replace("'YOUR_GOOGLE_MAPS_API_KEY'", key_literal)
    maps_script = ""
    if maps_key:
        encoded_key = escape(quote(maps_key, safe=""), quote=True)
        maps_script = (
            '<script async src="https://maps.googleapis.com/maps/api/js?'
            f'key={encoded_key}&amp;loading=async&amp;callback=initGoogleMap" '
            'onerror="showMapError(\'Could not load Google Maps. Check network access and API key settings.\')">'
            '</script>'
        )
    html = html.replace("<!-- GOOGLE_MAPS_SCRIPT -->", maps_script)

    return HTMLResponse(content=html)


@app.get("/api/health")
def health():
    return {"ok":True,"model":type(model).__name__,"expected_features":len(FEATURES),"dataset_rows":len(df),"events":int((df.sample_type=="event").sum()),"controls":int((df.sample_type=="control").sum()),"gee_ready":GEE_READY,"gee_error":GEE_ERROR,"gee_project":GEE_PROJECT}

@app.get("/api/history")
def history(years:int=Query(default=10,ge=1,le=20),limit:int=Query(default=1500,ge=1,le=5000)):
    # Historical catalog records are displayed as records, never passed through the predictor.
    events=df[df["sample_type"].astype(str).str.lower().eq("event")].copy()
    date_col="event_day" if "event_day" in events else "target_start"
    events["_date"]=pd.to_datetime(events[date_col],errors="coerce")
    cutoff=pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(days=365.25*years)
    events=events[events["_date"].notna() & (events["_date"]>=cutoff)].sort_values("_date",ascending=False).head(limit)
    out=[]
    for _,r in events.iterrows():
        if pd.isna(r.get("latitude")) or pd.isna(r.get("longitude")): continue
        out.append({"date":r["_date"].date().isoformat(),"lat":float(r["latitude"]),"lon":float(r["longitude"]),
          "title":str(r.get("event_title") or "Catalogued landslide"),"country":str(r.get("country_name") or ""),
          "source":str(r.get("source_name") or ""),"url":str(r.get("source_link") or ""),
          "status":str(r.get("provisional_status") or "Unreviewed catalogue candidate"),
          "quality":str(r.get("label_quality_tier") or "Unverified candidate")})
    return {"count":len(out),"items":out,"note":"Catalogue candidates are historical records, not predictions. Individual records may require verification."}

@app.get("/api/evaluation")
def evaluation():
    return EVALUATION

def demo_replay_response(errors=None):
    return {"generated_at":pd.Timestamp.now(tz="UTC").isoformat(),"mode":"demo_replay","demo_only":True,
      "provider":"Offline replay from the provided historical test split","model":type(model).__name__,
      "results":DEMO_RESULTS,"errors":errors or [],
      "note":"Offline demonstration only: these are historical event/control examples from the dataset's designated test split, not live conditions, current sites, forecasts, or warnings."}

@app.get("/api/demo-replay")
def demo_replay():
    return demo_replay_response()

@app.get("/api/live")
def live():
    results=[]; errors=[]
    for s in SITES:
        try: results.append(fetch_site(s))
        except Exception as e: errors.append({"site":s["name"],"error":str(e)[:220]})
    if results:
        return {"generated_at":pd.Timestamp.now(tz="UTC").isoformat(),"mode":"live","demo_only":False,
          "provider":"Google Earth Engine ERA5-Land + CHIRPS","model":type(model).__name__,
          "results":results,"errors":errors,"note":"Research scores only. ERA5-Land and CHIRPS may lag; scores are uncalibrated and not official warnings."}
    return demo_replay_response(errors)

@app.post("/api/gemini")
def gemini(payload:dict):
    key=os.getenv("GEMINI_API_KEY")
    if not key: return {"ok":False,"error":"Set GEMINI_API_KEY in .env to enable Gemini summaries."}
    # Send only compact prediction results; Gemini explains outputs and must not alter model scores.
    results=payload.get("results",[])[:20]
    source_context=(
        "The input is an offline replay of historical examples from the dataset's designated test split, not current sites or conditions. "
        "Do not frame these as current risks or recommend current-site actions. "
        if payload.get("demo_only") else
        "The input contains current provider-derived site scores, but they are experimental research outputs, not warnings. "
    )
    prompt=("Write a complete, concise analyst brief for a Himalayan hazard research dashboard in at most 140 words. "
      "Use three short sections: Situation; Sites to review (up to three, ranked only by model score); "
      "Limitations and verification. Explain that scores are uncalibrated research outputs, not official warnings. "
      "Historical events are past records, not predictions. "+source_context+
      "Mention data completeness/lag only when provided. "
      "Do not invent causes, observations, or certainty. Finish every sentence and include the human verification recommendation. "
      "Prediction data: "+json.dumps(results,ensure_ascii=False))
    try:
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
        resp = requests.post(
            url,
            headers={
                "x-goog-api-key": key,
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2048,
                },
            },
            timeout=45,
        )
        resp.raise_for_status()
        body = resp.json()
        candidates = body.get("candidates", [])
        if not candidates:
            block_reason = body.get("promptFeedback", {}).get("blockReason")
            detail = f"Gemini did not return a report. Block reason: {block_reason}." if block_reason else "Gemini did not return a report."
            return {"ok": False, "error": detail}
        candidate = candidates[0]
        text = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", []))
        if not text:
            return {"ok": False, "error": "Gemini returned no report text."}
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        return {
            "ok": True,
            "summary": text,
            "model": "gemini-flash-latest",
            "complete": finish_reason == "STOP",
            "finish_reason": finish_reason,
        }

    except Exception as e:
        return {"ok":False,"error":"Gemini request failed: "+str(e)[:240]}

app.mount("/static",StaticFiles(directory=ROOT/"static"),name="static")
