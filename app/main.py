# main.py
import json
import os
import logging
from urllib.parse import urlparse
import boto3
import botocore
import xgboost as xgb
import types
import collections
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

#Logging ==========
class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Include extra fields
        for k, v in record.__dict__.items():
            if k not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__:
                data[k] = v
        return json.dumps(data)

#logging.basicConfig(level=logging.INFO) 
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter()) 

LOG = logging.getLogger("model-api")
LOG.setLevel(logging.INFO)
LOG.addHandler(handler) 
LOG.info(
    "Test Msg",
    extra={"user_id": 123, "ip": "1.2.3.4"}
)
#==================


# Required env vars
MODEL_URI = os.environ.get("MODEL_S3_URI")
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/tmp/model")
AWS_REGION = os.environ.get("AWS_REGION", None)

app = FastAPI(title="MLflow Model Serve")

class PredictRequest(BaseModel):
    instances: Optional[List[Dict[str, Any]]] = None
    instance: Optional[Dict[str, Any]] = None

class PredictResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[float]] = None

def ensure_local_dir(path: str):
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o777)
    except Exception as e:
        LOG.error("chmod failed: %s", e)

def find_mlmodel_dir(start_path: str):
    for root, dirs, files in os.walk(start_path):
        if "MLmodel" in files:
            return root
    LOG.error(f"Could Not Find Model in {start_path=}")
    return None

def download_s3_prefix(s3_uri: str, dst: str):
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError("download_s3_prefix expects an s3:// URI")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    LOG.info("Downloading S3 bucket=%s prefix=%s to %s", bucket, prefix, dst)
    client = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION else boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = os.path.relpath(key, prefix)
            target = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            client.download_file(Bucket=bucket, Key=key, Filename=target)
            found = True
    if not found:
        raise RuntimeError(f"No objects found at {s3_uri}")
    LOG.info("S3 download complete.")

def try_mlflow_download(uri: str, dst: str):
    try:
        local = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=dst)
        return local
    except Exception as e:
        LOG.warning("mlflow.artifacts.download_artifacts failed: %s", e)
        return None

def download_model_artifacts(model_uri: str, dst: str):
    ensure_local_dir(dst)
    if not model_uri:
        LOG.error("MODEL_S3_URI not set")
        raise RuntimeError("MODEL_S3_URI not set")

    # MLflow artifact download
    if model_uri.startswith(("runs:", "models:", "artifact:")):
        res = try_mlflow_download(model_uri, dst)
        mlmodel_dir = find_mlmodel_dir(dst) or (find_mlmodel_dir(res) if res else None)
        if mlmodel_dir:
            LOG.info("Found MLmodel at %s", mlmodel_dir)
            return mlmodel_dir

    # S3 fallback
    parsed = urlparse(model_uri)
    if parsed.scheme == "s3":
        download_s3_prefix(model_uri, dst)
        mlmodel_dir = find_mlmodel_dir(dst)
        if mlmodel_dir:
            LOG.info("Found MLmodel at %s", mlmodel_dir)
            return mlmodel_dir
        else:
            raise RuntimeError(f"No MLmodel found under {dst} after s3 download.")

    raise ValueError(f"Unsupported model URI: {model_uri}")

def _safe_setattr(obj, name, val):
    try:
        setattr(obj, name, val)
        return True
    except Exception:
        try:
            # fallback: use __dict__ if possible
            obj.__dict__[name] = val
            return True
        except Exception:
            return False

def patch_xgb_instances(obj, max_depth=6, seen=None, patched_info=None):
    """
    Recursively walk `obj` to find XGBModel/XGBClassifier instances and add missing attributes.
    Returns number of patched instances and a dict of details in patched_info.
    """
    if seen is None:
        seen = set()
    if patched_info is None:
        patched_info = {"count": 0, "patched_fields": {}}

    try:
        obj_id = id(obj)
    except Exception:
        return patched_info

    if obj_id in seen:
        return patched_info
    seen.add(obj_id)

    # Direct instance check
    try:
        if isinstance(obj, (xgb.sklearn.XGBModel, xgb.sklearn.XGBClassifier)):
            fields_added = []
            # common missing attributes and safe defaults
            defaults = {
                "use_label_encoder": False,
                "gpu_id": -1,                    # -1 meaning CPU/no GPU
                "validate_parameters": True,
                # add other safe defaults as needed
            }
            for k, v in defaults.items():
                if not hasattr(obj, k):
                    ok = _safe_setattr(obj, k, v)
                    if ok:
                        fields_added.append(k)
            # some objects require set_params to update internal state
            try:
                obj.set_params(**{k: getattr(obj, k) for k in defaults if hasattr(obj, k)})
            except Exception:
                # ignore if set_params not applicable
                pass

            patched_info["count"] += 1
            if fields_added:
                patched_info["patched_fields"].setdefault(type(obj).__name__, []).extend(fields_added)
            return patched_info
    except Exception:
        # ignore instance checking errors and continue
        pass

    # Avoid descending into primitives / modules / functions
    if max_depth <= 0:
        return patched_info
    if isinstance(obj, (str, bytes, bytearray, int, float, complex, bool)):
        return patched_info
    if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.ModuleType, type)):
        return patched_info

    # If mapping
    if isinstance(obj, dict):
        for v in obj.values():
            patch_xgb_instances(v, max_depth - 1, seen, patched_info)
        return patched_info

    # If iterable
    if isinstance(obj, (list, tuple, set, frozenset, collections.deque)):
        for v in obj:
            patch_xgb_instances(v, max_depth - 1, seen, patched_info)
        return patched_info

    # Otherwise iterate attributes
    for attr in dir(obj):
        if attr.startswith("__"):
            continue
        # skip likely-large attributes
        if attr in ("_getstate_", "__getstate__", "_repr_html_", "feature_names"):
            continue
        try:
            v = getattr(obj, attr)
        except Exception:
            continue
        patch_xgb_instances(v, max_depth - 1, seen, patched_info)

    return patched_info

def load_local_model(local_path: str):
    """Load model from local path using pyfunc and patch XGBoost instances for missing attributes."""
    LOG.info("Loading model from %s using pyfunc", local_path)
    mlmodel_dir = find_mlmodel_dir(local_path)
    if not mlmodel_dir:
        raise RuntimeError(f"No MLmodel found under {local_path}")

    model = mlflow.pyfunc.load_model(mlmodel_dir)
    LOG.info("Model loaded as pyfunc. Patching nested XGBoost objects (if any).")

    try:
        info = patch_xgb_instances(model, max_depth=6)
        LOG.info("Patched XGBoost objects: count=%d patched_fields=%s", info.get("count", 0), info.get("patched_fields", {}))
    except Exception as e:
        LOG.warning("patch_xgb_instances error: %s", e)

    return model, "pyfunc"
# Global model
model = None
model_type = None

@app.on_event("startup")
def startup_event():
    global model, model_type
    LOG.info("Startup: MODEL_URI=%s LOCAL_MODEL_DIR=%s", MODEL_URI, LOCAL_MODEL_DIR)
    ensure_local_dir(LOCAL_MODEL_DIR)

    # Download and load model
    model_dir = download_model_artifacts(MODEL_URI, LOCAL_MODEL_DIR)
    model, model_type = load_local_model(model_dir)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    instances = req.instances or ([req.instance] if req.instance else None)
    if not instances:
        raise HTTPException(status_code=400, detail="Provide 'instance' or 'instances'")
    df = pd.DataFrame(instances)

    try:
        preds = model.predict(df).tolist()
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            # try:
            #     proba = model.predict_proba(df)
            #     LOG.info(f"predict_proba returns{proba=}", extra={'result': proba})
            #     if proba.ndim == 2 and proba.shape[1] >= 2:
            #         probs = proba[:, 1].tolist()
            # except Exception:
            #     probs = None
        LOG.info("Returning Response", extra=dict(predictions=preds, probabilities=probs))        
        return PredictResponse(predictions=preds, probabilities=probs)
    except Exception as e:
        LOG.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
