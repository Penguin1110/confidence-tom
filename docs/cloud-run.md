# Cloud Run Jobs

This project should run on Cloud Run as a Job, not a Service. The mainline
experiments are batch workloads and do not expose an HTTP server.

## Build

```bash
export PROJECT_ID="your-gcp-project"
export REGION="us-central1"
export REPO="confidence-tom"
export BUCKET="$PROJECT_ID-confidence-tom-outputs"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mainline:latest"

gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="confidence-tom containers"

gcloud storage buckets create "gs://$BUCKET" \
  --location="$REGION"

gcloud builds submit \
  --config cloudbuild.cloudrun.yaml \
  --substitutions _REGION="$REGION",_REPO="$REPO",_IMAGE=mainline,_TAG=latest .
```

## Deploy A GPU Job

Cloud Run GPU jobs use one L4 GPU. For local Hugging Face inference, start with
small models that fit in 24 GB VRAM, or use quantized/Ollama-style serving
instead of loading a 14B bf16 model directly.

```bash
gcloud run jobs deploy confidence-tom-mainline \
  --image "$IMAGE" \
  --region "$REGION" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu 8 \
  --memory 32Gi \
  --task-timeout 3600 \
  --max-retries 0 \
  --add-volume name=outputs,type=cloud-storage,bucket="$BUCKET" \
  --add-volume-mount volume=outputs,mount-path=/workspace/outputs \
  --set-env-vars CONFIDENCE_TOM_OUTPUT_ROOT=/workspace/outputs,HF_HOME=/workspace/.cache/huggingface \
  --set-secrets OPENROUTER_API_KEY=openrouter-api-key:latest
```

The job service account needs permission to write to the bucket, for example
`roles/storage.objectUser` on `gs://$BUCKET`.

## Run A Minimal Mainline Smoke

This executes one OlympiadBench task. The small worker below uses in-process
local transformers inference; the large worker still uses OpenRouter.

```bash
gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/run/core/run_prefix_oracle_gain_mapping.py,output_dir=/workspace/outputs/results/cloudrun_smoke_qwen_openai_1,dataset.benchmark=olympiadbench,dataset.limit=1,dataset.olympiadbench=1,execution.task_concurrency=1,execution.retry_attempts=1,small_worker.backend=local,small_worker.local_model_name=Qwen/Qwen3-4B,small_worker.model=qwen/qwen3-14b:nitro,small_worker.label=Qwen-Local,small_worker.max_tokens=1024,large_worker.model=openai/gpt-5.4,large_worker.label=GPT-5.4,large_worker.max_tokens=4096,extractor.enabled=true
```

## Follow-Up Analysis

Run analysis as separate jobs against the same output volume or persisted
results directory.

```bash
gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/prefix/analyze_prefix_oracle_gain.py,output_dir=/workspace/outputs/results/cloudrun_smoke_qwen_openai_1,dataset.benchmark=olympiadbench,dataset.olympiadbench=1,analysis.summary_json=/workspace/outputs/results/cloudrun_smoke_qwen_openai_1/summary.json,analysis.per_prefix_rows_csv=/workspace/outputs/results/cloudrun_smoke_qwen_openai_1/per_prefix_rows.csv
```

## Output Persistence

Cloud Run container filesystem is ephemeral. Before running expensive jobs,
persist outputs using one of these:

- Mount a Cloud Storage bucket volume at `/workspace/outputs` as shown above.
- Write outputs to a durable remote path from the job.
- Copy `/workspace/outputs` to Cloud Storage before the job exits.

For long sweeps, prefer one Cloud Run Job execution per run directory, then run
analysis jobs after the result JSONs are durable.
