# 基於介入價值（VoI）的多步推理協作框架提案

## 一、核心想法

在多步推理任務中，小模型便宜但容易在中途發生推理漂移；大模型較穩，但推理成本高。
我們希望建立一個協作框架：

- 由 **small worker** 先執行多步推理
- 由一個輕量級 **router / controller** 監控中間狀態
- 當預估「現在切換給大模型接手」是划算的，就讓 **large worker** 接管後續推理

這個問題的核心，不是單純比較大小模型誰比較強，而是：

> 在什麼時候讓大模型介入，才能用最小額外成本換到最大的成功率提升？

---

## 二、研究動機

目前 LLM 在多步推理場景中常有兩個明顯問題：

1. **小模型容易中途漂移**
- 前幾步看起來合理
- 但一旦某個中間步錯了，後面整條鏈都會崩

2. **大模型雖然穩，但成本過高**
- 若每題都直接用大模型，系統成本難以接受
- 對實際部署來說不一定划算

因此，更合理的問題不是：

- 要不要用大模型？

而是：

- **什麼時候該讓大模型接手？**

---

## 三、研究問題（RQs）

### RQ1：如何定義介入價值？

對於推理中任一步驟 `s_t`，我們希望衡量：

- 如果現在切給大模型，最後成功率會提升多少？
- 這個提升值是否值得額外成本？

### RQ2：有哪些外部可觀測特徵，可以預測「現在不救，後面會崩」？

在閉源模型場景下，我們通常看不到 hidden states，因此只能依靠外部訊號，例如：

- self-confidence
- 語義軌跡變化
- step-to-step 一致性
- 環境回饋訊號

### RQ3：如何在有限預算下蒐集反事實資料？

若要訓練 router，我們需要知道：

- 若在某步切換成大模型，最後會不會比較好？

這本質上是反事實問題，因此資料收集策略本身就是研究的一部分。

---

## 四、系統角色定義

為了避免和現有 `observer` 實驗混淆，這個提案統一使用以下角色：

### 1. Small Worker

- 負責先行推理
- 成本低
- 容易在複雜任務中發生 drift

### 2. Large Worker

- 在需要時接手剩餘推理
- 成本較高
- 預期更穩、更可恢復錯誤狀態

### 3. Router / Controller

- 不直接解題
- 只根據當前狀態 `s_t` 決定：
  - `continue with small worker`
  - `handoff to large worker`

---

## 五、理論框架：VoI（Value of Intervention）

我們將介入問題寫成一個效用最大化問題：

\[
\text{VoI}(s_t) = [P(y_t \mid s_t) - \lambda C_t] - [P(y_c \mid s_t) - \lambda C_c]
\]

其中：

- `P(y_t | s_t)`：在狀態 `s_t` 下，若切換給 large worker，最終成功的機率
- `P(y_c | s_t)`：在狀態 `s_t` 下，若讓 small worker 繼續，最終成功的機率
- `C_t`：切換後的成本
- `C_c`：繼續小模型的成本
- `λ`：成本敏感係數

當：

\[
\text{VoI}(s_t) > 0
\]

表示在這個步驟介入是划算的。

在實務上，router 不會直接知道真實的 `VoI`，而是學習用外部特徵去**近似預測**它。

---

## 六、介入單位

第一版建議使用：

- **step-level intervention**

也就是：

- small worker 每完成一步
- router 讀取當前狀態
- 判斷是否切給 large worker

這樣的好處是：

- 符合多步推理的自然結構
- 容易定義反事實
- 也能比較細緻地分析哪種步驟最值得救

---

## 七、可觀測特徵設計

第一版只使用外部可觀測特徵，不依賴閉源模型的內部狀態。

### 1. 信心訊號

- 當步自報信心
- 最近數步信心下降幅度
- 信心波動程度

### 2. 語義軌跡（semantic trajectory）

- 相鄰步驟 embedding distance
- 與前一個穩定狀態的偏移
- 語義跳躍是否異常變大

假設：

- 語義空間中的異常跳躍，常對應到邏輯斷裂或推理漂移

### 3. 文字行為特徵

- 是否反覆修正自己
- 是否出現矛盾或回退
- 是否開始輸出無法落地的泛化句子
- step 長度是否異常膨脹

### 4. 環境回饋（若有）

若任務是 tool-use / code / SQL，則可加入：

- execution error
- invalid tool arguments
- sandbox failure
- environment mismatch

---

## 八、資料收集：反事實標註

這個計畫的關鍵難點在於，我們需要知道：

- 如果當時切成大模型，最後結果會不會更好？

因此資料收集分成兩層：

### 1. Silver Counterfactual

流程：

1. 用 small worker 跑完整條推理軌跡
2. 對於最終失敗的軌跡，回溯每個候選步驟 `t`
3. 從 `s_t` 起補跑一次 large worker 接手版本
4. 比較最終成敗與成本

這樣可得到：

- 哪些失敗軌跡是可救的
- 哪些步驟一旦切換就值得

### 2. Gold Validation

只看失敗軌跡會有偏差，因為有些看似成功的狀態其實高風險。

因此需要額外：

- 從成功軌跡中抽樣一部分
- 一樣做介入測試

目的：

- 修正「成功就一定不用救」的偏差
- 捕捉那些 recoverable-but-risky 的狀態

---

## 九、Router 模型

第一版建議不用太重的模型，直接採用：

- `XGBoost`
或
- `LightGBM`

理由：

- 易於訓練
- 易於 debug
- feature importance 容易解釋
- 方便回答「到底哪個訊號最有用」

之後若要再做更強模型，可再考慮：

- small neural classifier
- sequence model
- policy learning

---

## 十、Baseline 設計

為了讓這個系統不是單純工程 demo，至少需要以下 baseline：

### Baseline 1：Always Small

- 全程只用 small worker

### Baseline 2：Always Large

- 全程只用 large worker

### Baseline 3：Fixed-Step Handoff

- 在固定步數切換
- 不考慮當前狀態

### Baseline 4：Confidence Threshold

- 當 small worker 自報信心低於某門檻時切換

你的方法應該要證明：

- 在相近成本下成功率更高
或
- 在相近成功率下成本更低

---

## 十一、評估指標

### 1. Final Success Rate

- 最終任務是否成功

### 2. Total Token Cost

- 全流程 token 消耗

### 3. Cost-Normalized Utility

- 單位成本所換到的成功率提升

### 4. Pareto Frontier

- 比較成功率與成本之間的 Pareto 效率

### 5. Intervention Precision / Recall

- Router 判斷介入的時機，是否真的值得救

---

## 十二、最小可行實驗（MVP）

第一版不要一開始就做太大，建議先縮成：

### 任務

- 數學 / 邏輯多步推理
- 優先考慮：
  - `OlympiadBench`
  - `LiveBench reasoning`

### 模型

- small worker：`Qwen-3-8B`
- large worker：`Qwen-3-32B`

### 資料

- 先收 `100 ~ 200` 條 small worker 軌跡
- 對失敗軌跡做反事實接管補跑
- 再抽部分成功軌跡做 gold validation

### 目標

- 訓練一個簡單 router
- 與 `confidence threshold` baseline 比較
- 驗證 selective intervention 是否更有效率

---

## 十三、與現有研究的連接

這個新方向和目前已完成的 `Qwen-3 scale + observer` 實驗並不是斷裂的。

它們之間的關係可以寫成：

1. 我們先證明：
   - 強 worker 的錯誤更具有欺騙性
2. 接著自然延伸到：
   - 如果錯誤真的有早期徵兆，那麼是否能在關鍵步驟及早介入？

也就是說，現有工作是在回答：

- **錯誤推理是否具有可欺騙性？**

下一步則是回答：

- **既然錯誤有代價，何時介入最划算？**

---

## 十四、最大風險

### 1. Step 定義可能太任意

- 若一步切得不自然，VoI 會失真

### 2. Counterfactual 標註成本高

- 若每步都補跑 large worker，預算可能撐不住

### 3. Router 可能學到 benchmark-specific 規則

- 泛化能力需特別驗證

### 4. 成本模型可能受 provider 影響

- token 成本與 latency 會隨 API provider 波動

---

## 十五、預期貢獻

若此方向成立，預期貢獻有三個：

1. 提出一個 cost-aware 的 LLM intervention formulation
2. 顯示 trace-based feature 可預測推理崩潰
3. 證明 selective intervention 比 naive routing 更有效率

---

## 十六、目前建議的下一步

1. 先固定第一版任務範圍
   - 只做數學 / 邏輯多步推理
2. 先完成 small worker step trace 的定義
3. 規劃 silver / gold 反事實資料收集流程
4. 實作最小版 router baseline

如果第一版能跑通，再擴到 tool-use 或 coding 任務。
