# Topic 4 汇报制作指引

**建议主线：逐步推导帮助答对更多科学数值题，但多花推理预算并不一定改善结果。** 用原 CoT 的一次答案转写错误，把主实验、采样投票、模型复核和程序校验串起来。推荐控制在约 10 分钟，预留 2 分钟讨论。图表源数据在 [ppt_metrics.csv](ppt_metrics.csv)。

| 页 | 内容与呈现 | 证据与可讲结论 |
|---|---|---|
| 1 | **Scientific Reasoning with Local LLMs**。用一句研究问题说明为何比较推理方式和成本。 | “When does an explicit reasoning strategy help a small local model solve scientific numerical problems?” |
| 2 | **公平对照**。用两条流程展示同一题、同一权重和相同评分。Direct 只输出答案，CoT 写推导后输出同格式 JSON。 | 100 道 SciBench 数值原题，97 个题目家族，Qwen3-1.7B Q4_K_M，主评分为单位匹配和数值容差。两组都关闭内建 thinking，CoT 由提示词要求可见推导。[方法报告](Direct_CoT/实验结果.md) |
| 3 | **主结果**。两根柱子画 5% 与 25%。旁边用四个数字展示配对变化和成本。 | 两者都对 2、只有 Direct 对 3、只有 CoT 对 23、两者都错 72。差值 +20 个百分点，按题目家族重采样区间 +11.0 至 +29.6。平均输出 23.91 vs 357.79 token。可以补充一例 CoT 改对（`fund_047`）及一例改错（`fund_030`）。[主实验汇总](Direct_CoT/results/formal/summary.json) |
| 4 | **Self-Consistency**。配对柱子比较单次采样 25% 与三次采样投票 23%，另标约 3 倍输出 token。 | 300 次采样。单次采样取第一份，以 1% 数值接近和单位匹配聚类投票。相对第一次采样，投票改对 1 题、改错 3 题，差值 95% 区间 −6.0 至 +2.0 个百分点。`chemmc_013` 的两个错误波长约为正确答案的十倍，投票选了错误多数。[投票报告](Self_Consistency/实验结果.md) |
| 5 | **什么时候值得复核？** 用三行列出原 CoT、全量复核、自适应复核，左右并排展示正确题数和额外调用。 | 正确数 25、26、25，额外调用 0、100、7。全量复核改对 2、改错 1。自适应在同一批复核记录上重放，7 道均未改对。节省 93% 指的是相对全量的**额外复核调用**。全量比基线多对 1 道，配对区间跨 0。[复核报告](Adaptive_Solver_Verifier/实验结果.md) |
| 6 | **发现错误后用程序做精确修复**。用同一题展示“正文 → JSON → 程序补指数”。给出 25% → 28% 及“额外模型调用 0”。 | `chemmc_009` 正文写 (4.747\times10^{14}\) Hz，JSON 填 `4.747`。模型复核再次漏指数。程序从已写出的正文提取指数，只在尾数和单位吻合时改写 JSON。5 处修改中改对 3、改错 0，2 道正文计算本来就错。规则从已见错误构思，幻灯片标注“探索性”，独立题集尚未验证。[程序校验报告](Programmatic_Verifier/实验结果.md) |
| 7 | **队友工作如何接上**。把已有工具反馈及 ARC 工作放入单独的结果区或附录。 | 历史 SciBench 实验比较共享初答后的复核及计算反馈，实验条件与这里的 Direct/CoT 不同。ARC 实验是 600 道选择题测试，准确率 76% 等来自另一个题型和流程。沿用原仓库表格和解释，分别标注题型、分母和实验协议。 |
| 8 | **结论与局限**。保留准确率、成本、两个负结果和探索性发现。 | 只在这批题、当前小模型和提示条件下得出主实验结论。最终数值评分不能替代对推导步骤的检查。投票可能重复系统性错误，模型复核也可能沿用错答。程序补指数是输出校验，未对科学公式进行普遍验证。 |

可直接用于英文汇报的三句话：

- “On the same 100 SciBench problems, explicit CoT increased final-answer accuracy from 5% to 25% under a fixed model and scorer.”
- “Three-sample voting did not improve the sampled CoT baseline. Selective verification used 7 review calls and retained the baseline accuracy.”
- “A deterministic exponent repair fixed three transcription errors without further generation. This is an exploratory result on the same set used to motivate the rule.”

PPT 中的数字应取自正式结果。“百分点”表示正确率之差。可以分别显示正确率与 token/调用次数，避免用一个轴混用不同量纲。团队原有选择题图表与本次 SciBench 图表分别呈现。模型复核 100 次是已运行的离线对照，7 次是实际触发后使用的子集成本。程序校验没有新增模型生成，耗时未单独测速。
