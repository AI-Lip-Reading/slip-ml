import json
import os
import boto3
import torch
from inference import load_model_from_s3, end_to_end_inference, VALLRModel, PHONEME_VOCAB, IDX_TO_PHONEME

s3_client = boto3.client("s3")

def model_fn(model_dir):
    s3_bucket = os.environ.get("SM_DEFAULT_BUCKET", "slip-ml")
    vallr_s3_key = os.environ.get("VALLR_S3_KEY", "models/vallr_model.tar.gz")
    llama_s3_key = os.environ.get("LLAMA_S3_KEY", "models/llama_model.tar.gz")
    local_vallr_path = os.path.join(model_dir, "vallr_model.tar.gz")
    local_llama_path = os.path.join(model_dir, "llama_model.tar.gz")

    vallr_model, _ = load_model_from_s3(s3_bucket, vallr_s3_key, local_vallr_path, model_type="vallr")
    llama_model, tokenizer = load_model_from_s3(s3_bucket, llama_s3_key, local_llama_path, model_type="llama")
    return {"vallr_model": vallr_model, "llama_model": llama_model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        input_s3_path = data.get("input_s3_path")
        if not input_s3_path:
            raise ValueError("input_s3_path required in JSON payload")
        s3_bucket = input_s3_path.split("/")[2]
        s3_key = "/".join(input_s3_path.split("/")[3:])
        local_path = f"/tmp/{os.path.basename(s3_key)}"
        s3_client.download_file(s3_bucket, s3_key, local_path)
        return local_path, s3_key.endswith(".npz")
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    local_path, is_npz = input_data
    vallr_model = model["vallr_model"]
    llama_model = model["llama_model"]
    tokenizer = model["tokenizer"]
    phonemes, text = end_to_end_inference(vallr_model, llama_model, tokenizer, local_path, is_npz=is_npz)
    return {"phonemes": phonemes, "text": text}

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")