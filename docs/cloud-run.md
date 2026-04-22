# Cloud Run Jobs

This project should run on Cloud Run as a Job, not a Service. The mainline
experiments are batch workloads and do not expose an HTTP server.

Important: rotate any OpenRouter key that was pasted into chat or shell history.
Store the new key only in Secret Manager.

## Build And Deploy

```bash
export PROJECT_ID="your-gcp-project"
export REGION="asia-southeast1"
export REPO="confidence-tom"
export BUCKET="$PROJECT_ID-confidence-tom-outputs"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mainline:latest"

gcloud config set project "$PROJECT_ID"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com

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

Create or update the OpenRouter secret:

```bash
read -s OPENROUTER_API_KEY
printf "%s" "$OPENROUTER_API_KEY" | \
  gcloud secrets create openrouter-api-key --data-file=- --project="$PROJECT_ID"

# If the secret already exists:
printf "%s" "$OPENROUTER_API_KEY" | \
  gcloud secrets versions add openrouter-api-key --data-file=- --project="$PROJECT_ID"
```

Deploy the job:

```bash
gcloud run jobs deploy confidence-tom-mainline \
  --image "$IMAGE" \
  --region "$REGION" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu 4 \
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

## Smoke Test First

Run one task before spending money on the full matrix.

```bash
gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/run/core/run_prefix_oracle_gain_mapping.py,output_dir=/workspace/outputs/results/openrouter_test_v1,dataset.benchmark=olympiadbench,dataset.limit=1,dataset.olympiadbench=1,execution.task_concurrency=1,execution.retry_attempts=1,small_worker.backend=local,small_worker.local_model_name=Qwen/Qwen2.5-0.5B-Instruct,small_worker.model=qwen/qwen3-14b:nitro,small_worker.label=Qwen-Local,small_worker.max_tokens=1024,large_worker.backend=openrouter,large_worker.model=google/gemini-2.0-flash-001,large_worker.label=Gemini-2.0-Flash,large_worker.max_tokens=4096,extractor.enabled=true
```

Check output:

```bash
gcloud storage ls "gs://$BUCKET/results/openrouter_test_v1/"
```

## Complete Canonical Experiment

The downstream predictor dataset currently expects these 12 run directories:

```text
qwen_to_openai_50
qwen_to_anthropic_50
llama_to_openai_50
llama_to_anthropic_50
mistral_to_openai_50
mistral_to_anthropic_50
livebench_qwen_to_openai_30
livebench_qwen_to_anthropic_30
livebench_llama_to_openai_30
livebench_llama_to_anthropic_30
livebench_mistral_to_openai_30
livebench_mistral_to_anthropic_30
```

Use small local models if you want the small worker to run on Cloud Run GPU.
Replace local model names with models that fit on one L4 GPU. Public models do
not need `HF_TOKEN`; gated/private models do.

```bash
export QWEN_LOCAL_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export LLAMA_LOCAL_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export MISTRAL_LOCAL_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

run_prefix() {
  local run_name="$1"
  local benchmark="$2"
  local limit="$3"
  local small_label="$4"
  local small_model="$5"
  local large_family="$6"
  local large_model="$7"
  local large_label="$8"

  local dataset_count_arg="dataset.olympiadbench=$limit"
  if [ "$benchmark" = "livebench_reasoning" ]; then
    dataset_count_arg="dataset.livebench_reasoning=$limit"
  fi

  gcloud run jobs execute confidence-tom-mainline \
    --region "$REGION" \
    --wait \
    --args=experiments/mainline/run/core/run_prefix_oracle_gain_mapping.py,output_dir=/workspace/outputs/results/$run_name,dataset.benchmark=$benchmark,dataset.limit=$limit,$dataset_count_arg,execution.task_concurrency=1,execution.retry_attempts=3,small_worker.backend=local,small_worker.local_model_name=$small_model,small_worker.model=$small_model,small_worker.label=$small_label,small_worker.max_tokens=1024,large_worker.backend=openrouter,large_worker.model=$large_model,large_worker.label=$large_label,large_worker.max_tokens=4096,extractor.enabled=true
}

analyze_prefix() {
  local run_name="$1"
  local benchmark="$2"
  local limit="$3"
  local dataset_count_arg="dataset.olympiadbench=$limit"
  if [ "$benchmark" = "livebench_reasoning" ]; then
    dataset_count_arg="dataset.livebench_reasoning=$limit"
  fi

  gcloud run jobs execute confidence-tom-mainline \
    --region "$REGION" \
    --wait \
    --args=experiments/mainline/analysis/prefix/analyze_prefix_oracle_gain.py,output_dir=/workspace/outputs/results/$run_name,dataset.benchmark=$benchmark,$dataset_count_arg,analysis.summary_json=/workspace/outputs/results/$run_name/summary.json,analysis.per_prefix_rows_csv=/workspace/outputs/results/$run_name/per_prefix_rows.csv
}
```

Run OlympiadBench 50:

```bash
run_prefix qwen_to_openai_50 olympiadbench 50 Qwen-Local "$QWEN_LOCAL_MODEL" openai openai/gpt-5.4 GPT-5.4
analyze_prefix qwen_to_openai_50 olympiadbench 50

run_prefix qwen_to_anthropic_50 olympiadbench 50 Qwen-Local "$QWEN_LOCAL_MODEL" anthropic anthropic/claude-opus-4.6 Claude-Opus-4.6
analyze_prefix qwen_to_anthropic_50 olympiadbench 50

run_prefix llama_to_openai_50 olympiadbench 50 Llama-Local "$LLAMA_LOCAL_MODEL" openai openai/gpt-5.4 GPT-5.4
analyze_prefix llama_to_openai_50 olympiadbench 50

run_prefix llama_to_anthropic_50 olympiadbench 50 Llama-Local "$LLAMA_LOCAL_MODEL" anthropic anthropic/claude-opus-4.6 Claude-Opus-4.6
analyze_prefix llama_to_anthropic_50 olympiadbench 50

run_prefix mistral_to_openai_50 olympiadbench 50 Mistral-Local "$MISTRAL_LOCAL_MODEL" openai openai/gpt-5.4 GPT-5.4
analyze_prefix mistral_to_openai_50 olympiadbench 50

run_prefix mistral_to_anthropic_50 olympiadbench 50 Mistral-Local "$MISTRAL_LOCAL_MODEL" anthropic anthropic/claude-opus-4.6 Claude-Opus-4.6
analyze_prefix mistral_to_anthropic_50 olympiadbench 50
```

Run LiveBench Reasoning 30:

```bash
run_prefix livebench_qwen_to_openai_30 livebench_reasoning 30 Qwen-Local "$QWEN_LOCAL_MODEL" openai openai/gpt-5.4 GPT-5.4
analyze_prefix livebench_qwen_to_openai_30 livebench_reasoning 30

run_prefix livebench_qwen_to_anthropic_30 livebench_reasoning 30 Qwen-Local "$QWEN_LOCAL_MODEL" anthropic anthropic/claude-opus-4.6 Claude-Opus-4.6
analyze_prefix livebench_qwen_to_anthropic_30 livebench_reasoning 30

run_prefix livebench_llama_to_openai_30 livebench_reasoning 30 Llama-Local "$LLAMA_LOCAL_MODEL" openai openai/gpt-5.4 GPT-5.4
analyze_prefix livebench_llama_to_openai_30 livebench_reasoning 30

run_prefix livebench_llama_to_anthropic_30 livebench_reasoning 30 Llama-Local "$LLAMA_LOCAL_MODEL" anthropic anthropic/claude-opus-4.6 Claude-Opus-4.6
analyze_prefix livebench_llama_to_anthropic_30 livebench_reasoning 30

run_prefix livebench_mistral_to_openai_30 livebench_reasoning 30 Mistral-Local "$MISTRAL_LOCAL_MODEL" openai openai/gpt-5.4 GPT-5.4
analyze_prefix livebench_mistral_to_openai_30 livebench_reasoning 30

run_prefix livebench_mistral_to_anthropic_30 livebench_reasoning 30 Mistral-Local "$MISTRAL_LOCAL_MODEL" anthropic anthropic/claude-opus-4.6 Claude-Opus-4.6
analyze_prefix livebench_mistral_to_anthropic_30 livebench_reasoning 30
```

## Complete Post-Processing

After all 12 run directories contain a result JSON, `summary.json`, and
`per_prefix_rows.csv`, run these jobs.

```bash
gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/trace/analyze_trace_taxonomy.py

gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/data/build_prefix_predictor_dataset.py

gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/data/train_prefix_gain_baseline.py

gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/prefix/analyze_prefix_diagnostics.py

gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/prefix/analyze_prefix_task_structure.py

gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/embedding/analyze_cross_benchmark_task_clusters.py
```

Optional re-entry controls:

```bash
gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/run/core/run_prefix_reentry_controls.py,--category,stable-success,--max-rows,100,--concurrency,1,--small-backend,local,--small-local-model-name,$QWEN_LOCAL_MODEL

gcloud run jobs execute confidence-tom-mainline \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/prefix/analyze_prefix_reentry_controls.py
```

## Check Outputs

```bash
gcloud storage ls "gs://$BUCKET/results/"
gcloud storage ls "gs://$BUCKET/results/qwen_to_openai_50/"
gcloud storage ls "gs://$BUCKET/results/_prefix_predictor_v1/"
```

Cloud Run container filesystem is ephemeral. The Cloud Storage mount at
`/workspace/outputs` is the durable output location.
