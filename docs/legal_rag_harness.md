# 法律 RAG Harness

`run_legal_rag_harness.py` 用于对当前法律 RAG 做可回归、可比较的批量评测。

## 能评什么

- `scope_accuracy`：是否正确识别为法律问题
- `retrieval_hit_rate`：目标法条是否被召回
- `citation_hit_rate`：最终引用是否命中目标法条
- `answer_pass_rate`：答案是否满足 benchmark 里定义的必须包含/必须排除约束
- `overall_pass_rate`：单题综合通过率
- `avg_latency_ms`：平均响应时延

## 运行方式

```bash
python run_legal_rag_harness.py --details
```

指定模式：

```bash
python run_legal_rag_harness.py --modes hybrid llm_retrieval
```

设置回归门槛：

```bash
python run_legal_rag_harness.py ^
  --fail-below-overall-pass-rate 0.8 ^
  --fail-below-retrieval-hit-rate 0.9
```

图形界面：

```bash
python legal_rag_harness_gui.py
```

## Benchmark 格式

示例见 [legal_qa_benchmark.json](/D:/PythonFile/JCAI/RAG/eval/legal_qa_benchmark.json)。

每条样本支持以下字段：

- `id`：唯一标识
- `question`：待评测问题
- `expected_scope`：通常为 `legal`
- `expected_references`：期望命中的来源与条次
- `require_retrieval_hit`：是否把召回命中纳入通过条件，默认当 `expected_references` 非空时为 `true`
- `require_citation_hit`：是否把最终引用命中纳入通过条件，默认 `false`
- `answer_checks.must_include_all`：答案必须全部包含的短语
- `answer_checks.must_include_any`：答案必须至少命中每组中的一个短语
- `answer_checks.must_exclude_all`：答案不应包含的短语

## 输出

默认输出到：

`eval/reports/legal_rag_harness_latest.json`

该 JSON 包含：

- 知识库统计
- 模式级指标
- 每题明细
- 失败原因
- 候选召回与引用预览
