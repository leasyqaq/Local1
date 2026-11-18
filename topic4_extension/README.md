# SDSC6020 Topic 4 实验与汇报交接

主实验在同一批 100 道 SciBench 数值原题上用同一个 Qwen3-1.7B Q4_K_M 比较直接回答和逐步推导。正确率为 **5% 和 25%**。三项扩展分别研究多次推导投票、选择何时复核、用程序修复漏写的指数。

制作汇报优先阅读 [PPT 汇报指引](PPT_HANDOFF.md)，图表数字见 [ppt_metrics.csv](ppt_metrics.csv)。如果由 agent 接手，从 [AGENTS.md](AGENTS.md) 开始。项目原始记录与新增实验各自保留。

| 目录 | 结果 | 原始证据 |
|---|---|---|
| [Direct_CoT](Direct_CoT/实验结果.md) | Direct Answer 5%，CoT 25%。CoT 改对 23 题、改错 3 题。 | [200 份原始作答](Direct_CoT/results/formal/records.jsonl)、[逐题成绩](Direct_CoT/results/formal/paired_scores.csv)、[错误复核](Direct_CoT/results/formal/错题复核.md) |
| [Self_Consistency](Self_Consistency/实验结果.md) | 单次采样 25%，三次采样投票 23%。输出成本约三倍。 | [300 份采样](Self_Consistency/results/formal/records.jsonl)、[100 次投票](Self_Consistency/results/formal/vote_records.jsonl) |
| [Adaptive_Solver_Verifier](Adaptive_Solver_Verifier/实验结果.md) | 每题复核 26%，自适应复核 25%。自适应在全量记录中重放，使用其中 7 次复核。 | [100 份复核](Adaptive_Solver_Verifier/results/formal/reviews.jsonl)、[触发依据](Adaptive_Solver_Verifier/results/formal/routing.jsonl) |
| [Programmatic_Verifier](Programmatic_Verifier/实验结果.md) | 补回漏写的指数达到 28%，改对 3 题、改错 0 题。规则受这批题中的案例启发，结果为探索性。 | [5 处实际修改](Programmatic_Verifier/results/formal/repairs.jsonl)、[逐题对照](Programmatic_Verifier/results/formal/paired_scores.csv) |

正式评分沿用队友的单位匹配及数值容差规则。主结果使用 1% 相对容差或该题原绝对容差的较大者。每份报告也记录了 5% 容差的敏感性结果。正式推理共新增 600 次模型生成（200 主对照、300 采样、100 全量复核），另外保存预跑。程序校验复用主实验输出。

在本目录执行 `python verify_handoff.py`，使用 Python 3.11+ 的标准库，无需 GPU 或下载模型。脚本验证文件摘要、主实验分数和正式输出，并运行三个 bonus 的独立复算。各实验的参数、完整运行方式及验收标准在相应目录的 `README.md`。重新生成模型输出会用到已固定摘要的 GGUF 和 llama.cpp b12404。归档自带固定输入、原始作答、配置和完整评分，因此做报告及离线复算可以直接使用存档。

数据源为 [SciBench](https://github.com/mandyyyyii/scibench/tree/e14e0ca3d3db0f02928a18d435b9c3d15656a7c5)，选择与修订来源见 [来源记录](Direct_CoT/data/source_manifest.json)、[子集清单](Direct_CoT/data/expanded_manifest.json) 和 [参考答案审计](Direct_CoT/data/expanded_answer_audit.csv)。[SciBench 许可](Direct_CoT/data/SciBench_LICENSE.txt)随附。原项目基准为 [Local1 `33139f4`](https://github.com/leasyqaq/Local1/tree/33139f48c4eab8cd94b0531d7895d50bada4da59)。

实验约定和过程见 [agent.md](agent.md) 与 [progress.md](progress.md)。所有报告和数字均可定位至正式 `summary.json`、`paired_scores.csv` 或原始输出。
