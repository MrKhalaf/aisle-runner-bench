# Aisle Runner Bench - Implementation Plan

## Overview

Build a benchmark framework to evaluate browser automation approaches for e-commerce tasks (specifically Sysco cart building). This enables data-driven decisions on model/library selection and provides infrastructure for DSPy prompt optimization.

---

## Phase 1: Core Infrastructure

### 1.1 Task Definitions
Define atomic, measurable tasks:

```python
@dataclass
class Task:
    id: str                    # "sysco-login", "sysco-add-item"
    description: str           # Human-readable description
    start_url: str            # Starting point
    success_criteria: list    # How to verify completion
    max_steps: int            # Step budget
    timeout_seconds: int      # Time budget
```

**Initial Tasks:**
- `sysco-login` - Log into Sysco with credentials
- `sysco-search` - Search for a specific product
- `sysco-add-to-cart` - Add a product with quantity
- `sysco-checkout-flow` - Navigate to checkout (don't submit)
- `sysco-full-order` - Complete flow: login → add items → cart

### 1.2 Evaluation Harness

```
prepai-eval/
├── src/
│   ├── tasks/           # Task definitions
│   ├── runners/         # Library adapters (browser-use, skyvern, etc.)
│   ├── metrics/         # Metric collection
│   ├── reporters/       # Output formats (JSON, HTML, CSV)
│   └── eval.py          # Main orchestrator
├── tasks/               # Task YAML definitions
├── results/             # Run outputs
└── analysis/            # Jupyter notebooks for analysis
```

### 1.3 Metrics Collection

Per-run metrics:
```python
@dataclass
class RunResult:
    task_id: str
    model: str
    library: str
    success: bool
    steps: int
    total_time_seconds: float
    time_per_step: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str]
    trace: list[StepTrace]  # Full action history
```

---

## Phase 2: Library Adapters

### 2.1 Adapter Interface

```python
class BrowserRunner(Protocol):
    async def run_task(self, task: Task, config: RunConfig) -> RunResult:
        """Execute a task and return results."""
        ...
```

### 2.2 Implementations

| Library | Status | Notes |
|---------|--------|-------|
| browser-use | Priority 1 | Current baseline |
| Skyvern | Priority 2 | Cloud-based, may have better selectors |
| Stagehand | Priority 3 | Browserbase's AI browser SDK |
| Custom Playwright | Priority 4 | Direct comparison without framework |

### 2.3 Model Configurations

```yaml
models:
  claude-opus-4-5:
    provider: anthropic
    model_id: claude-opus-4-5-20251101
    cost_per_1k_input: 0.015
    cost_per_1k_output: 0.075

  claude-sonnet-4:
    provider: anthropic
    model_id: claude-sonnet-4-20250514
    cost_per_1k_input: 0.003
    cost_per_1k_output: 0.015

  claude-haiku-35:
    provider: anthropic
    model_id: claude-3-5-haiku-20241022
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.005

  gpt-4o:
    provider: openai
    model_id: gpt-4o
    cost_per_1k_input: 0.005
    cost_per_1k_output: 0.015
```

---

## Phase 3: Test Suite

### 3.1 Mock Environment

For fast iteration without hitting real Sysco:
- Local mock server mimicking Sysco pages
- Deterministic responses
- Enables CI/CD integration

### 3.2 Live Environment

For real-world accuracy:
- Actual Sysco site (with test account)
- Rate limiting to avoid abuse
- Screenshot capture for debugging

### 3.3 Ground Truth Dataset

```yaml
# tasks/sysco-add-spinach.yaml
task: sysco-add-to-cart
product: "spinach case"
expected_match: "Imperial Fresh Spinach"
quantity: 1
start_state: logged_in
success_criteria:
  - cart_contains: "spinach"
  - cart_quantity: 1
```

---

## Phase 4: DSPy Integration

### 4.1 Signature Definition

```python
class BrowserAction(dspy.Signature):
    """Given a screenshot and task, output the next browser action."""

    screenshot: dspy.Image = dspy.InputField()
    task: str = dspy.InputField()
    page_elements: str = dspy.InputField()
    history: str = dspy.InputField()

    reasoning: str = dspy.OutputField()
    action: str = dspy.OutputField()
```

### 4.2 Optimization Targets

- **Minimize steps** - Fewer LLM calls = faster + cheaper
- **Maximize success rate** - Task completion
- **Optimize action selection** - Better first-attempt accuracy

### 4.3 Training Data

Use eval runs to build training set:
```python
# Successful traces become positive examples
# Failed traces with corrections become negative examples
```

---

## Phase 5: Reporting & Analysis

### 5.1 CLI Reports

```bash
$ prepai-eval compare --results results/2024-01-*

┌─────────────────┬─────────┬───────┬──────────┬─────────┐
│ Config          │ Success │ Steps │ Time (s) │ Cost    │
├─────────────────┼─────────┼───────┼──────────┼─────────┤
│ opus/browser-use│ 95%     │ 24    │ 142      │ $0.89   │
│ sonnet/browser  │ 88%     │ 31    │ 98       │ $0.23   │
│ haiku/browser   │ 72%     │ 45    │ 67       │ $0.08   │
│ gpt4o/browser   │ 85%     │ 28    │ 112      │ $0.34   │
└─────────────────┴─────────┴───────┴──────────┴─────────┘
```

### 5.2 Detailed Analysis

Jupyter notebooks for:
- Step-by-step trace visualization
- Failure mode analysis
- Cost/performance Pareto frontier
- Model behavior comparison

---

## Implementation Order

1. **Week 1**: Core infrastructure
   - Task definition schema
   - Run harness skeleton
   - browser-use adapter
   - Basic metrics

2. **Week 2**: Multi-model support
   - Model configuration system
   - Sonnet, Haiku, Opus comparison
   - Cost tracking

3. **Week 3**: Additional libraries
   - Skyvern adapter
   - Stagehand adapter
   - Comparison baselines

4. **Week 4**: DSPy integration
   - Signature definitions
   - Training data pipeline
   - Initial optimization experiments

5. **Week 5**: Polish
   - Mock environment for CI
   - HTML reports
   - Documentation

---

## Open Questions

1. **Mock vs Live**: How much can we trust mock environment results?
2. **Variance**: How many runs needed for statistical significance?
3. **Task granularity**: Atomic tasks vs end-to-end flows?
4. **Multi-site**: Extend to other e-commerce sites?

---

## Success Criteria

- [ ] Can run same task across 3+ models with one command
- [ ] Automated metrics collection (success, steps, time, cost)
- [ ] Reproducible results with seed control
- [ ] Clear winner identification for production use
- [ ] DSPy optimization shows measurable improvement
