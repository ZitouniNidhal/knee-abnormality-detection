# Silent Model Drift

**Detect unannounced behavior changes in LLM API endpoints without model access.**

Managed LLM providers can change weights, routing, safety policies, decoding defaults, tool behavior, or infrastructure while preserving the public model name. A RAG pipeline or agent can then regress without a clear deployment event to explain it.

Silent Model Drift treats an endpoint as a black box. It repeatedly sends a compact, versioned suite of sentinel prompts, converts outputs into privacy-conscious behavioral fingerprints, and compares a fresh run against a locked baseline using permutation tests.

> A positive result means that observed endpoint behavior changed under the measured configuration. It does **not** prove that the provider changed model weights; routing, policies, retrieval, tool availability, or API settings may also cause a shift.

## What it provides

- A versioned JSON sentinel-suite format with immutable suite hashes.
- Repeated sampling to model normal endpoint nondeterminism.
- Content fingerprints that retain output structure and lexical similarity without storing raw output by default.
- Per-sentinel kernel two-sample permutation tests with Benjamini-Hochberg false-discovery-rate correction.
- A separate dispersion test to distinguish central behavioral shifts from increased nondeterminism.
- Analytical sample-size planning for a target standardized effect and power.
- A dependency-free Python API and CLI, including OpenAI Chat Completions, Anthropic Messages, and Google Gemini adapters.
- JSON reports suitable for CI alerts, dashboards, or long-running monitors.

## Quick start

```bash
cd silent-model-drift
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Validate the bundled prompt suite.
silent-drift validate-suite examples/sentinels.json

# Estimate the number of independent observations required in each window.
silent-drift plan --effect-size 0.8 --power 0.9 --repeats 5
```

Capture a baseline with a pinned request configuration. API credentials stay in environment variables and are never written to an artifact.

```bash
$env:OPENAI_API_KEY = "..."
silent-drift capture --suite examples/sentinels.json --provider openai --model gpt-4.1-mini --temperature 0 --repeats 8 --output artifacts/baseline.json
```

Later, collect a comparison run with the same suite and request settings, then compare it to the baseline:

```bash
silent-drift capture --suite examples/sentinels.json --provider openai --model gpt-4.1-mini --temperature 0 --repeats 8 --output artifacts/current.json

silent-drift compare --baseline artifacts/baseline.json --candidate artifacts/current.json --permutations 1999 --output artifacts/report.json
```

The comparison command returns exit code `2` when it detects a behavior or nondeterminism change, which makes it convenient for scheduled CI checks. It returns `0` when no statistically supported change is found and `1` for invalid input or operational failures.

Run a complete local rehearsal without an API key:

```bash
silent-drift capture --suite examples/sentinels.json --provider replay --model offline --replay-responses examples/replay-baseline.json --repeats 8 --output artifacts/baseline.json
silent-drift capture --suite examples/sentinels.json --provider replay --model offline --replay-responses examples/replay-candidate.json --repeats 8 --output artifacts/candidate.json
silent-drift compare --baseline artifacts/baseline.json --candidate artifacts/candidate.json --permutations 999
```

## How it works

Each sentinel is sampled several times in the baseline window and again in the candidate window. For each prompt, the project computes a hybrid response distance from:

- normalized-output equality;
- token-set overlap;
- character n-gram overlap;
- relative output length; and
- coarse JSON-output structure.

It applies a kernel maximum mean discrepancy (MMD) test to the two response distributions and obtains a p-value by permuting window labels. That makes no normality assumption about LLM output. P-values across prompts are adjusted with Benjamini-Hochberg; Fisher's method supplies a suite-level summary.

The report also compares within-window response dispersion, cross-window central separation, and medoid displacement:

| Result | Interpretation |
| --- | --- |
| `behavioral_shift` | A sentinel's response distribution and central response changed. |
| `nondeterminism_increase` | Output variability increased without a material medoid displacement. |
| `distribution_shift` | The output distribution changed, but the source of the change is not separable with current evidence. |
| `stable` | No statistically supported change at the selected false-discovery rate. |

Read the detailed methodology in [docs/methodology.md](docs/methodology.md).

## Designing a useful sentinel suite

Use short prompts that exercise the behavior your application relies on rather than generic trivia. A balanced suite commonly includes:

- schema-following and structured-output prompts;
- retrieval-grounding prompts with controlled context;
- refusal/safety boundary prompts that are safe to run;
- multilingual, formatting, and tool-selection cases if your application uses them;
- domain-specific decision prompts with non-sensitive synthetic inputs.

Keep system prompt, tools, retrieval corpus snapshot, temperature, token budget, API region, and model name fixed where possible. Add tags to prompts so a report can identify the affected behavior family.

Provider-specific non-secret settings, such as a tool definition or response format, may be stored in a JSON file and supplied with `capture --extras settings.json`. They become part of the request-config digest. Record region, SDK version, service tier, and any retrieval snapshot identifier there too.

The bundled `examples/sentinels.json` intentionally uses harmless, provider-agnostic probes. Replace it with a suite representing your own production contract before treating alerts as operational signals.

## Privacy and artifact handling

By default, captures store only hashes and coarse features of model output. This is enough for the included detector, but token and n-gram hashes should still be treated as potentially sensitive metadata. Raw output is stored only when `--store-raw` is passed. Do not place production secrets, personal data, or proprietary context in sentinel prompts or captured artifacts.

## Research questions supported

### How many prompts are enough?

There is no universal number. Power depends on effect size, response variability, multiple-testing burden, and correlation among repeated calls to the same prompt. The `plan` command uses the equal-size two-sample normal approximation:

```text
n per window = 2 × ((z(1 - α / 2) + z(power)) / standardized_effect)^2
```

It then converts observations to prompts using the number of repeats and an optional intra-prompt correlation. Treat the result as a planning lower bound; validate it with a historical replay or a controlled provider/model comparison.

### Can it distinguish a model change from nondeterministic noise?

It can distinguish a persistent distribution or medoid shift from increased within-window dispersion under a fixed measurement setup. It cannot identify the provider-side cause. For high-confidence incidents, repeat the candidate run, inspect an independent sentinel family, and compare request metadata before escalation.

## Python API

```python
from silent_model_drift.fingerprint import make_fingerprint
from silent_model_drift.statistics import DriftConfig, analyze_windows

baseline = [make_fingerprint("The answer is blue.") for _ in range(8)]
candidate = [make_fingerprint("The answer is green.") for _ in range(8)]

report = analyze_windows(
    {"color": baseline},
    {"color": candidate},
    config=DriftConfig(permutations=999, seed=7),
)
print(report.conclusion)
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

## Publishing to GitHub

The project is self-contained in this directory. From inside it, create your repository and push it:

```bash
git init
git add .
git commit -m "Initial release of Silent Model Drift"
gh repo create silent-model-drift --public --source=. --remote=origin --push
```

If you do not use the GitHub CLI, create an empty GitHub repository, add it as `origin`, and push the `main` branch.

## License

MIT. See [LICENSE](LICENSE).

