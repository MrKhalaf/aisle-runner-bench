# PrepAI Eval

Benchmark suite for evaluating browser automation approaches for e-commerce cart building tasks.

## Purpose

Systematically compare:
- **Models**: Claude Opus 4.5, Sonnet 4, Haiku 3.5, GPT-4o, Gemini 2.0
- **Libraries**: browser-use, Skyvern, custom Playwright, Stagehand
- **Prompting strategies**: Zero-shot, few-shot, DSPy-optimized

## Quick Start

```bash
pip install -e .
prepai-eval run --task sysco-login --model claude-opus-4-5
prepai-eval compare --results results/
```

## Metrics

- **Success rate**: % of tasks completed correctly
- **Steps**: Number of LLM calls to complete task
- **Latency**: Total time and time per step
- **Cost**: Token usage and $ cost per task
- **Robustness**: Success rate across multiple runs

## Project Status

See [PLAN.md](./PLAN.md) for implementation roadmap.
