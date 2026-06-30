# Late-Success vs Late-Failure 中文整合版（Ollama-only）

資料來源是從 server 同步回來的 `results/`，並且只保留 `small_model.startswith("ollama/")` 的 30 題 `LiveBench reasoning` 結果。這份文件把前面的總體統計和後面的案例分析合成在一起，目的是把 `late-success` 與 `late-failure` 的差異講得既有數字、也有例子。

## 一句話結論

`late-success` 和 `late-failure` 的關鍵差別，不是 first-correct prefix 出現得多早，而是它出現之後，能不能被後續 trace 穩定保留下來。

## 定義

- `late-success`：要到比較後面才出現正確 prefix，或者局部正確訊號不夠穩，但 full trace 最終仍然答對。
- `late-failure`：中途曾出現局部正確 prefix，但 full trace 最終仍然答錯。

## 總體比較

| 類別 | 數量 | 佔比 | 平均步數 | 平均 first-correct frac | 中位數 first-correct frac | 平均 local correct rate | last-small-correct rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late-success | 32 | 0.133 | 3.875 | 0.628 | 0.500 | 0.562 | 0.781 |
| late-failure | 39 | 0.163 | 3.436 | 0.527 | 0.333 | 0.424 | 0.333 |

## 怎麼解讀這組數字

- `late-success` 平均來說比 `late-failure` 更晚才出現 first-correct prefix：`0.628` 對 `0.527`。
- 但 `late-success` 一旦出現局部正確，後面比較能維持住，所以 mean local correct rate 更高：`0.562` 對 `0.424`。
- `late-success` 在最後一步仍維持 local correctness 的比例也明顯更高：`25/32`，而 `late-failure` 只有 `13/39`。
- 這表示真正重要的不是「第幾步開始對」，而是「對了之後能不能繼續維持對」。

## 補充：taxonomy 的一個 edge case

目前定義下，`late-success` 裡有 `4` 題其實是：

- full trace 最後答對
- 但所有 local re-entry prefix 都沒有答對

也就是說，這些 case 在現行 taxonomy 裡被併進 `late-success`，但它們其實比較像「final answer correct without locally reusable prefix」。這點在解讀時要另外註明。

## 模型分布

| Small model | Late-success | Late-failure | Persistent-failure | Stable-success |
| --- | ---: | ---: | ---: | ---: |
| ollama/gemma3:4b | 6 | 11 | 40 | 3 |
| ollama/gemma4:e4b | 7 | 13 | 13 | 27 |
| ollama/ministral-3:3b | 9 | 9 | 37 | 5 |
| ollama/qwen3.5:4b | 10 | 6 | 6 | 38 |

這裡也能看出一些直覺上的差異：

- `ollama/qwen3.5:4b` 的 `stable-success` 很高，`late-failure` 比較少，代表它一旦形成正確 state，往往比較能保住。
- `ollama/gemma3:4b` 的 `late-failure` 和 `persistent-failure` 都偏多，代表它比較容易出現「曾經局部可用，但沒保住」或乾脆「一直沒形成可用 prefix」。
- `ollama/ministral-3:3b` 在 `late-success` 和 `late-failure` 上差不多，顯示它比較像處在一個不穩定的中間帶。

## 代表案例

下面挑四種最值得講的 pattern。小提醒：有些 task 在 `GPT-5.4` 和 `Claude-Opus-4.6` 版本都出現，所以你在原始案例集裡會看到同一題重複兩次。這裡我只抓現象本身，不把重點放在 large model 差異。

### 1. Strong Late-Success

- Small model：`ollama/gemma4:e4b`
- Task：`livebench_reasoning_448411..._0002`
- 指標：`first_correct_frac = 0.167`，`local_correct_rate = 0.833`，`last_small_correct = True`

這題的特徵是：

- 前面不是每一步都穩，但一旦 trace 累積到夠多資訊後，後續大多數 prefix 都能重新走到正解。
- full trace 最終也答對。

它代表的是：

> 可用的局部正確 state 一旦形成，就被後續 reasoning 保留了下來。

所以這類 case 雖然不一定是最早就對，但它的關鍵是「後面穩」。

### 2. Edge Late-Success

- Small model：`ollama/gemma3:4b`
- Task：`livebench_reasoning_a0f4b0..._0009`
- 指標：`first_correct_frac = 1.000`，`local_correct_rate = 0.000`，`last_small_correct = False`

這題很特別：

- full trace 最後答對，答案是 `4`
- 但所有 prefix re-entry 的 small continuation 都答成 `7`

它代表的是：

> final answer 雖然對，但 trace 中沒有被 prefix probe 捕捉到可重用的局部正確狀態。

所以它提醒我們，現在的 `late-success` 類別其實混了兩種東西：

- 真正「晚但可保留」的 success
- final answer 對、但 prefix probe 看不出可重用 correctness 的 edge case

### 3. Strong Late-Failure

- Small model：`ollama/gemma3:4b`
- Task：`livebench_reasoning_b183db..._0012`
- 指標：`first_correct_frac = 0.333`，`local_correct_rate = 1.000`，`last_small_correct = True`

這題是最有研究味的 case：

- 幾乎每個 prefix re-entry 都能得到正確 local continuation
- 但 full trace 最終還是答錯

這正是 `late-failure` 最強的版本，也就是：

> 模型中途其實已經處在一個可以走到正解的 state，但原本那條完整 trace 沒有把這個 state 保留下來。

如果要跟老師講「不是單純能力不足，而是 state preservation 出問題」，這題最好用。

### 4. Fragile Late-Failure

- Small model：`ollama/gemma3:4b`
- Task：`livebench_reasoning_84df21..._0018`
- 指標：`first_correct_frac = 0.833`，`local_correct_rate = 0.167`，`last_small_correct = False`

這題是另一種比較脆弱的 `late-failure`：

- 一直到很後面才短暫出現一次 correct prefix
- 之後又掉回去
- full trace 最後也答錯

它代表的是：

> 正確訊號不是完全沒出現，而是出現得太晚、太弱、太不穩，還沒形成可被保留的 state。

這和上面的 `strong_late_failure` 不一樣。前者是「其實已經有穩定正確 state，但丟掉了」；這題更像是「正確 state 只閃一下，根本沒站穩」。

## 這四個案例合起來說明了什麼

這幾個案例放在一起看，會得到一個很清楚的 picture：

- 有些題目不是早早就可重用，而是到中後段才累積出可用 prefix；只要形成後能保住，就會變成 `late-success`。
- 有些題目中途其實已經有正確狀態，但完整 trace 最後沒有延續它，這就是 `late-failure`。
- `late-failure` 內部又有兩種不同味道：
  - 強型：局部正確其實一直存在，但 full trace 沒保住
  - 脆弱型：局部正確只短暫出現，還沒來得及穩定就消失

所以如果要把這條研究線往前推，一個很自然的 framing 會是：

> re-entry 不是在找某個 universal 最佳步數，而是在 probe 一條 trace 的 correctness state 是否已經形成、以及能不能被保留。

## 可以直接拿去講的中文版本

如果你要用比較口語的方式講，我會建議這樣說：

> 我們現在看到，`late-success` 和 `late-failure` 的差別，不是誰比較早出現 correct prefix，而是 correct prefix 出現之後能不能被後面的 reasoning 保住。`late-success` 甚至平均比 `late-failure` 更晚才出現 first-correct prefix，但它後續的 local correctness 和最後一步 correctness 都比較高。這表示真正重要的不是步數本身，而是 trace 的 state persistence。
> 更關鍵的是，有一類 `late-failure` 題目，中途幾乎每個 prefix re-entry 都能答對，但 full trace 最後還是答錯。這說明問題不只是模型會不會，而是模型能不能保住已經出現的正確狀態。這也支持我們把這個工作定位成 trace-level diagnosis / taxonomy，而不是單純在找某個固定最佳 re-entry step。

## 對應檔案

- 英文總結版：[late_success_vs_late_failure_ollama_only.md](/Users/powerarena/Documents/GitHub/confidence-tom/docs/mainline/generated/analysis/trace/late_success_vs_late_failure_ollama_only.md)
- 英文案例集：[late_success_vs_late_failure_case_studies_ollama_only.md](/Users/powerarena/Documents/GitHub/confidence-tom/docs/mainline/generated/analysis/trace/late_success_vs_late_failure_case_studies_ollama_only.md)
