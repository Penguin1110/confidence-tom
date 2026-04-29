# Mainline Experiments

主線實驗分成三塊：

- `run/`: 真正啟動實驗的入口
- `analysis/`: 對既有結果做分析、畫圖、整理 trace taxonomy
- `data/`: 建 dataset、訓練 baseline、產生中間資料

如果只是要找「從哪裡開始跑」，先看 `run/`。

目前 `re-entry` 已經是 mainline 的一條獨立執行線：

- `run/batch/run_reentry_mainline.py`: 大規模 `prepare -> re-entry -> analyze`
- `run/core/run_prefix_reentry_controls.py`: 對既有 prefix 結果做 re-entry stability controls
