# Cloud Run Jobs

This project should run on Cloud Run as a Job, not a Service. The mainline
experiments are batch workloads and do not expose an HTTP server.

This Cloud Run path assumes a pure local-model setup inside the container.
The mainline batch job does not require OpenRouter.

## Build And Deploy

```bash
export PROJECT_ID="your-gcp-project"
export REGION="asia-southeast1"
export REPO="confidence-tom"
export BUCKET="$PROJECT_ID-confidence-tom-outputs"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/reentry-mainline:latest"

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
  --substitutions _REGION="$REGION",_REPO="$REPO",_IMAGE=reentry-mainline,_TAG=latest .
```

Deploy the job:

```bash
gcloud run jobs deploy confidence-tom-reentry \
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
  --set-env-vars CONFIDENCE_TOM_OUTPUT_ROOT=/workspace/outputs,HF_HOME=/workspace/.cache/huggingface
```

The job service account needs permission to write to the bucket, for example
`roles/storage.objectUser` on `gs://$BUCKET`.

## How Small Models Load

There are two small-model paths in the mainline repo:

- `small_worker.backend=local`
  Loads Hugging Face weights inside the container with `transformers`.
  The code path is `src/confidence_tom/infra/client.py` plus
  `src/confidence_tom/infra/client_local.py`.
  `Dockerfile.cloudrun` already installs the `local-inference` dependency group and
  caches models under `/workspace/.cache/huggingface`.
- `small_worker.backend=ollama`
  Sends OpenAI-compatible requests to an Ollama server. This is convenient on a VM
  you control, but it is not the default Cloud Run path in this repo.

For Cloud Run, `local` is the in-container way to run public Hugging Face small models.
Public checkpoints do not need `HF_TOKEN`; gated checkpoints do.

Practical note: a single `nvidia-l4` Cloud Run job is fine for smoke tests and smaller
local checkpoints, but `24B` to `32B` models may need quantization, lower concurrency,
or a different serving stack before they are production-stable.

## Smoke Test First

Run one task before spending money on the full matrix.

```bash
gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/run/core/run_prefix_oracle_gain_mapping.py,output_dir=/workspace/outputs/results/local_test_v1,dataset.benchmark=olympiadbench,dataset.limit=1,dataset.olympiadbench=1,execution.task_concurrency=1,execution.retry_attempts=1,small_worker.backend=local,small_worker.local_model_name=Qwen/Qwen2.5-7B-Instruct,small_worker.model=Qwen/Qwen2.5-7B-Instruct,small_worker.label=Qwen-Local,small_worker.max_tokens=1024,large_worker.backend=local,large_worker.local_model_name=Qwen/Qwen2.5-14B-Instruct,large_worker.model=Qwen/Qwen2.5-14B-Instruct,large_worker.label=Qwen-Local-Large,large_worker.max_tokens=2048,extractor.enabled=false
```

Check output:

```bash
gcloud storage ls "gs://$BUCKET/results/local_test_v1/"
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

Use local models for both small and large workers.
Replace model names with checkpoints that fit on one L4 GPU. Public models do
not need `HF_TOKEN`; gated or private models do.

```bash
export QWEN_LOCAL_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export QWEN_LARGE_LOCAL_MODEL="Qwen/Qwen2.5-14B-Instruct"
export MISTRAL_LOCAL_MODEL="mistralai/Ministral-8B-Instruct-2410"

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

  gcloud run jobs execute confidence-tom-reentry \
    --region "$REGION" \
    --wait \
    --args=experiments/mainline/run/core/run_prefix_oracle_gain_mapping.py,output_dir=/workspace/outputs/results/$run_name,dataset.benchmark=$benchmark,dataset.limit=$limit,$dataset_count_arg,execution.task_concurrency=1,execution.retry_attempts=3,small_worker.backend=local,small_worker.local_model_name=$small_model,small_worker.model=$small_model,small_worker.label=$small_label,small_worker.max_tokens=1024,large_worker.backend=local,large_worker.local_model_name=$large_model,large_worker.model=$large_model,large_worker.label=$large_label,large_worker.max_tokens=4096,extractor.enabled=false
}

analyze_prefix() {
  local run_name="$1"
  local benchmark="$2"
  local limit="$3"
  local dataset_count_arg="dataset.olympiadbench=$limit"
  if [ "$benchmark" = "livebench_reasoning" ]; then
    dataset_count_arg="dataset.livebench_reasoning=$limit"
  fi

  gcloud run jobs execute confidence-tom-reentry \
    --region "$REGION" \
    --wait \
    --args=experiments/mainline/analysis/prefix/analyze_prefix_oracle_gain.py,output_dir=/workspace/outputs/results/$run_name,dataset.benchmark=$benchmark,$dataset_count_arg,analysis.summary_json=/workspace/outputs/results/$run_name/summary.json,analysis.per_prefix_rows_csv=/workspace/outputs/results/$run_name/per_prefix_rows.csv
}
```

Run OlympiadBench 50:

```bash
run_prefix qwen_to_localqwen_50 olympiadbench 50 Qwen-Local "$QWEN_LOCAL_MODEL" local_qwen "$QWEN_LARGE_LOCAL_MODEL" Qwen-Local-Large
analyze_prefix qwen_to_localqwen_50 olympiadbench 50

run_prefix mistral_to_localqwen_50 olympiadbench 50 Mistral-Local "$MISTRAL_LOCAL_MODEL" local_qwen "$QWEN_LARGE_LOCAL_MODEL" Qwen-Local-Large
analyze_prefix mistral_to_localqwen_50 olympiadbench 50
```

Run LiveBench Reasoning 30:

```bash
run_prefix livebench_qwen_to_localqwen_30 livebench_reasoning 30 Qwen-Local "$QWEN_LOCAL_MODEL" local_qwen "$QWEN_LARGE_LOCAL_MODEL" Qwen-Local-Large
analyze_prefix livebench_qwen_to_localqwen_30 livebench_reasoning 30

run_prefix livebench_mistral_to_localqwen_30 livebench_reasoning 30 Mistral-Local "$MISTRAL_LOCAL_MODEL" local_qwen "$QWEN_LARGE_LOCAL_MODEL" Qwen-Local-Large
analyze_prefix livebench_mistral_to_localqwen_30 livebench_reasoning 30
```

## Complete Post-Processing

After all 12 run directories contain a result JSON, `summary.json`, and
`per_prefix_rows.csv`, run these jobs.

```bash
gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/trace/analyze_trace_taxonomy.py

gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/data/build_prefix_predictor_dataset.py

gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/data/train_prefix_gain_baseline.py

gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/prefix/analyze_prefix_diagnostics.py

gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/prefix/analyze_prefix_task_structure.py

gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/embedding/analyze_cross_benchmark_task_clusters.py
```

Optional re-entry controls:

```bash
gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/run/core/run_prefix_reentry_controls.py,--category,stable-success,--max-rows,100,--concurrency,1,--small-backend,local,--small-local-model-name,$QWEN_LOCAL_MODEL

gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/analysis/prefix/analyze_prefix_reentry_controls.py
```

Re-entry mainline dry-run on the orchestrator:

```bash
gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/run/batch/run_reentry_mainline.py,--preset,reentry_livebench_local,--phase,all,--dry-run
```

Transformer probe after re-entry:

```bash
gcloud run jobs execute confidence-tom-reentry \
  --region "$REGION" \
  --wait \
  --args=experiments/mainline/run/core/run_prefix_reentry_probe.py,--rows,/workspace/outputs/results/_reentry_livebench_local_v1/reentry_rows.jsonl,--output-dir,/workspace/outputs/results/_reentry_livebench_local_v1/probe,--backend,transformers,--local-model-map,qwen=Qwen/Qwen3-14B,--local-model-map,gemma=google/gemma-3-4b-it,--local-model-map,mistral=mistralai/Ministral-8B-Instruct-2410,--local-model-map,olmo=allenai/olmo-2-13b-instruct
```

Recommended production split:

- Keep `confidence-tom-reentry` as the generation/re-entry job.
- Run a second `transformers` probe pass for hidden states and attention summaries after the re-entry rows exist.

## Check Outputs

```bash
gcloud storage ls "gs://$BUCKET/results/"
gcloud storage ls "gs://$BUCKET/results/qwen_to_openai_50/"
gcloud storage ls "gs://$BUCKET/results/_prefix_predictor_v1/"
```

Cloud Run container filesystem is ephemeral. The Cloud Storage mount at
`/workspace/outputs` is the durable output location.
