# SDSC6020 Topic 4 实验说明

队友的 Local1 在 ARC 上比较输出头、校准与示例检索，也在 SciBench 上比较四种复核策略。这里的 `Direct_CoT/` 补做 Topic 4 要求的 Direct Answer 与 CoT 对照。两个项目的原始记录各自保留。

- 队友的固定版本为 [Local1 `33139f4`](https://github.com/leasyqaq/Local1/tree/33139f48c4eab8cd94b0531d7895d50bada4da59)。正式实验使用 `Direct_CoT/data/expanded_100.jsonl` 中的 100 道原题。6 道参数变式不进入分母。
- 沿用同一份 Qwen3-1.7B Q4_K_M 权重与 `scoring.py`。模型、题库、评分代码和推理配置在运行时校验或记录。正式组使用同一推理后端、温度、输出上限、答案格式和判分标准。
- Direct Answer 只要求最终答案。CoT 要求展示推导，再给同格式的最终答案。两组关闭模型内建 thinking，且不提供外部工具反馈、示例或参考答案。
- 用正式题集之外的题做运行预检。保留格式错误及达到 token 上限的正式输出。调整提示词或运行环境时，用新目录保存结果，并记录差异。
- 汇报两组总体正确率与同题得分变化，错误诊断以原始输出和题库参考记录为依据。实验协议和验收标准见 [Direct_CoT/README.md](Direct_CoT/README.md)，进度见 [progress.md](progress.md)。

## 文件管理

- 本项目的文件统一保存在 `6020project/` 内。现有课程与队友文档留在根目录，新增对照实验的代码与说明放在 `Direct_CoT/`。
- 固定题库和来源放在 `Direct_CoT/data/`。预跑、正式实验和独立复跑的记录各用 `Direct_CoT/results/` 下的独立目录。
- 模型、运行时、原始分包归档及项目临时文件放在 `Direct_CoT/_runtime/`，该目录已在 Git 忽略规则中。正式结果以归档的原始输出和逐题评分为准。
- 三次采样投票的 bonus 实验放在 `Self_Consistency/`，复用原题、CoT 提示、权重和判分代码。见 [Self-Consistency 协议与验收](Self_Consistency/README.md)。
- 自适应 Solver–Verifier 放在 `Adaptive_Solver_Verifier/`。路由与复核只使用题目及生成文本。见 [自适应复核协议与验收](Adaptive_Solver_Verifier/README.md)。
- 程序化校验放在 `Programmatic_Verifier/`。仅重放已保存的 CoT 输出，使用原评分规则。见 [程序化校验协议](Programmatic_Verifier/README.md)。
