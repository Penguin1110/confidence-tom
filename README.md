# Confidence-ToM: 大模型心智理論實驗

測試大模型能否讀懂小模型的內在信心狀態 (Theory of Mind for Confidence Prediction)

## 🎯 研究問題

**大模型能否透過閱讀小模型的思考過程，準確預測小模型的自評信心？**

## 🔬 實驗架構

### 第一階段：數據生產 (The Generator)
- **角色**：小模型 (Target Subject, e.g., Gemma-2-9B-It)
- **任務**：回答問題並輸出思考過程與自評信心
- **產出物**：`[Question, CoT, Answer, True_Confidence]`

### 第二階段：盲測預測 (The Observer)
- **角色**：大模型 (Predictor, e.g., GPT-4o, Claude 3.5, Gemini 1.5 Pro)
- **任務**：閱讀小模型的思考過程，預測其信心狀態
- **產出物**：`[Question, Predicted_Confidence, Prediction_Reasoning]`

### 第三階段：對齊分析 (The Evaluator)
- **角色**：分析腳本
- **任務**：比較 True_Confidence 與 Predicted_Confidence

## 🏗️ 專案結構

```
confidence-tom/
├── src/confidence_tom/
│   ├── __init__.py
│   ├── config.py
│   ├── client.py
│   ├── generator/          # 第一階段：小模型產生數據
│   ├── observer/           # 第二階段：大模型預測信心
│   └── evaluator/          # 第三階段：對齊分析
├── experiments/
├── results/
├── configs/
└── tests/
```

## 🛠️ 安裝

```bash
# 安裝所有依賴
uv sync --all-groups

# 安裝 pre-commit hook
uv run pre-commit install
```

## 🚀 執行實驗

```bash
# 第一階段：生成小模型回答
uv run python experiments/run_generator.py

# 第二階段：大模型預測信心
uv run python experiments/run_observer.py

# 第三階段：分析結果
uv run python experiments/run_evaluator.py
```

## 📄 License

MIT
