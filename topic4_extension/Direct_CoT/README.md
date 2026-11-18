# SciBench Direct Answer 与 CoT 对照

本实验回答同一款本地模型在同一批科学数值题上，先写推导是否会影响最终答案的正确率。队友已有的 ARC 选择题和 SciBench 复核实验提供项目背景。这里新增 Topic 4 要求的推理策略对照。

正式实验已完成，见 [实验结果](实验结果.md) 和 [逐题复核](results/formal/错题复核.md)。

## 固定设计

| 项目 | 设置 |
|---|---|
| 正式题库 | Local1 固定版本中的 100 道 SciBench 原题，每题两组各作答一次。6 道参数变式不计入正式结果。 |
| 预跑题库 | `atkins_002`、`atkins_007`、`atkins_012`、`atkins_013`，与正式题库没有已标记的家族重叠。 |
| 权重和后端 | 与队友相同的 Qwen3-1.7B Q4_K_M GGUF 与 llama.cpp b12404。两组均使用同一个 Windows 后端与设备。 |
| Direct Answer | 提示模型只给包含 `answer`、`unit` 的最终 JSON。 |
| CoT | 提示模型以 `Step 1:` 起头，写 2–4 个简短步骤，展示公式和计算，再给同格式的最终 JSON。 |
| 控制项 | 相同题干、目标单位、常量说明、JSON 答案格式和评分代码。两组均关闭内建 thinking，使用温度 0 的贪心解码、相同的 768 输出 token 上限，不提供示例、工具结果或参考答案。 |
| 顺序 | 在每道题内用固定种子打乱两组顺序。关掉提示词缓存，以减少上一条请求对当前请求的影响。 |
| 判分 | 沿用归档中的 `scoring.py`。要求可解析的数值答案、匹配指定单位，且误差不超过题目的 1% 相对容差或该题原有的绝对容差（取较大者）。同时报告 5% 容差结果。 |

保留每一次正式输出。数字无法解析、单位不符合、答案错误或输出达到上限的情况均进入检查。提示遵从情况另行报告，正确率采用上表同一套判分规则。

输入来自 [Local1 `33139f4`](https://github.com/leasyqaq/Local1/tree/33139f48c4eab8cd94b0531d7895d50bada4da59)，其中 SciBench 来源版本和许可见 `data/source_manifest.json`、`data/SciBench_LICENSE.txt`。题库已有部分答案审计和题意澄清，采用这份固定子集的题目与参考答案。该子集经过有目的的筛选，结果适用于这批题和当前设置。

## 验收步骤

1. 校验 GGUF 的 SHA-256，确认 Windows 推理后端版本和实际设备。当前框架会核验数据、评分代码及模型文件。
2. 用 4 道预跑题验证 8 次完整调用。检查 Direct 是否能输出纯答案 JSON，CoT 是否有可见推导且最终 JSON 可解析。预跑检查格式和工程稳定性，正确率高低不作为通过条件。若需修改提示或输出上限，只在预跑阶段做，并记录最终方案。
3. 固定方案后生成 200 条正式记录。每个题号必须各有一条 Direct 和 CoT 结果。运行记录保留完整提示词、输出、答题用时、token 数、停止原因和评分。两组结果共用同一个运行配置。
4. 从原始输出重新判分并检查配置，生成正确率、5% 容差结果、同题正误变化、按来源分层和资源开销。报告 CoT−Direct 的差值及按题目家族重采样的 95% 区间。逐题 McNemar 检验作为补充，题目家族依赖作为限制。
5. 人工复核两组正误不同的题目，以及答案解析、单位或长度上限导致的失败。从其余两组都答错的题目中抽查 10 道，按题目 ID 排序后用固定种子 `20261118` 抽样；不足 10 道时全查。记下题目 ID、可见证据和参考计算。Direct 只留下答案时，记录具体数值或单位差异，不推断内部思路。报告结果方向、限制和资源成本。

贪心解码下，完全相同的设置复跑主要用于核验可复现性。独立复跑使用新输出目录，并记录环境与逐题差异。若另外研究采样随机性，应单独固定双组采样参数和种子列表。

## Windows 入口

需要 Python 3.11 或以上。运行与分析仅使用标准库。Windows Vulkan 版取自 [llama.cpp b12404](https://github.com/ggml-org/llama.cpp/releases/tag/b12404)。基础权重使用 Local1 下载脚本指定的 [Ollama 模型文件](https://registry.ollama.ai/v2/library/qwen3/blobs/sha256:3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6)，文件放在 `_runtime/qwen3-1.7b-Q4_K_M.gguf`，运行器会校验哈希。

准备好环境并继续实验时，在本目录执行以下命令。先验收预跑，再执行正式运行和汇总。`Vulkan0` 应与本机 `llama-server.exe --list-devices` 的 NVIDIA 设备编号一致。

```powershell
python .\fetch_model.py
$server = Resolve-Path .\_runtime\llama-b12404-vulkan\llama-server.exe
python .\run_direct_cot.py pilot --server-exe $server.Path --out .\results\pilot_v4

python .\run_direct_cot.py formal --server-exe $server.Path
python .\analyze_results.py
```

初版预跑发现 CoT 会把答案放进 LaTeX。后续版本明确要求在纯文本 JSON 前展示步骤，最终通过 4 题 8 次输出检查。各版预跑分别保存在 `results/pilot/`、`results/pilot_v2/`、`results/pilot_v3/` 和 `results/pilot_v4/`。调试依据是提示遵从和答案能否解析。

正式结果将写入 `results/formal/`，包括 `config.json`、`records.jsonl`、`paired_scores.csv`、`by_source.csv`、`summary.json` 以及 `error_review_queue.csv`（有待复核题目时）。再次运行时，配置一致才会继续未完成的题目。独立复跑可用 `--out` 指定新目录。
