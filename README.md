# Confidence-ToM: 大模型心智理論實驗

測試大模型能否讀懂小模型的內在信心狀態 (Theory of Mind for Confidence Prediction)

## 🎯 研究問題

**大模型能否透過閱讀小模型的思考過程，準確預測小模型的「經驗正確率」($C_{beh}$)？**

### 核心概念

- **Subject (小模型)**: Gemma-3-27B，負責回答問題並產生思考過程 (CoT)
- **Observer (大模型)**: GPT-4o / Claude 3.5 / Gemini 1.5 Pro，負責預測 Subject 的信心狀態
- **Ground Truth ($C_{beh}$)**: 對同一題目進行 K=10 次獨立採樣後的**答對比例**

## 🔬 三組對照實驗

### 組別 A：Blind Observer (純 ToM 組)
- **資訊量**：題目 + 小模型的 CoT + 小模型答案（**看不到正確答案**）
- **目的**：測試 Observer 是否能單純從推理邏輯的「嚴密程度」和「內在一致性」預判小模型會不會出錯
- **特點**：完全排除「後見之明偏誤」

### 組別 B：Informed Observer (後見之明組)
- **資訊量**：題目 + **正確答案** + 小模型的 CoT + 小模型答案
- **目的**：測試當 AI 知道正確答案時，是否能更精準地定位錯誤，還是會變得過於嚴苛
- **特點**：作為後見之明偏誤的基準組

### 組別 C：Frame-Aware Observer (P2+ 最強介入組)
- **資訊量**：與組別 B 相同，但在閱讀小模型答案前，必須先執行「**邏輯陷阱宣告**」
- **目的**：測試「強制前瞻性思考」能否修正組別 B 的偏見
- **特點**：包含診斷分析 (Trap Detection) 和「歪打正著」(Luck Factor) 偵測

## 🏗️ 專案結構

```
confidence-tom/
├── src/confidence_tom/
│   ├── __init__.py
│   ├── client.py              # LLM API 客戶端
│   ├── dataset.py             # 資料集載入 (BBH, GSM8K, TruthfulQA)
│   ├── generator/             # 第一階段：小模型產生數據
│   │   ├── generator.py       # SubjectGenerator (K次採樣)
│   │   └── models.py          # 資料結構 (含結構化信心解釋)
│   ├── observer/              # 第二階段：大模型預測信心
│   │   ├── observer.py        # 三組 Observer 實作
│   │   ├── models.py          # 判斷輸出結構 (含陷阱宣告)
│   │   └── protocols.py       # 協議上下文建構器
│   └── evaluator/             # 第三階段：對齊分析
│       └── evaluator.py       # 量化指標計算
├── experiments/
│   ├── run_generator.py       # 執行第一階段
│   ├── run_observer.py        # 執行第二階段 (三組對照)
│   └── analyze_results.py     # 第三階段分析與視覺化
├── configs/
│   └── config.yaml            # Hydra 配置檔
├── results/                   # 實驗輸出
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
# 第一階段：生成小模型回答 (K=10 次採樣計算 C_beh)
uv run python experiments/run_generator.py

# 第二階段：三組大模型觀察實驗
uv run python experiments/run_observer.py

# 第三階段：分析結果並生成報告
uv run python experiments/analyze_results.py
```

## 📊 評估指標

| 指標                      | 說明                                           |
| ------------------------- | ---------------------------------------------- |
| **Prediction MAE**        | $\|C_{pred} - C_{beh}\|$ - Observer 預測精準度 |
| **Self-Assessment MAE**   | $\|C_{rep} - C_{beh}\|$ - Subject 自評精準度   |
| **Hindsight Bias**        | Group B vs Group A 的誤差差異                  |
| **P2+ Improvement**       | Group C vs Group B 的誤差改善                  |
| **Luck Factor Detection** | 偵測「答對但理由錯」的樣本                     |

## 📈 輸出報告

- `results/analysis_report.json` - 完整分析報告
- `results/plots/` - 視覺化圖表
  - `prediction_accuracy_by_group.png` - 各組預測精準度
  - `hindsight_bias_analysis.png` - 後見之明偏誤分析
  - `calibration_curves.png` - 校準曲線
  - `p2_plus_effectiveness.png` - P2+ 協議效果
  - `luck_factor_analysis.png` - 歪打正著分析

## 📄 License

MIT
