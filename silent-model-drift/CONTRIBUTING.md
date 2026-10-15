# Contributing

Thanks for improving Silent Model Drift.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Use .venv\Scripts\Activate.ps1 in PowerShell.
pip install -e ".[dev]"
pytest
ruff check .
```

## Contribution principles

- Keep the core package dependency-free unless a dependency provides a clear, measured benefit.
- Preserve reproducibility: statistical functions must accept a seed when randomness is used.
- Do not log credentials, raw prompts, or raw outputs by default.
- Add tests for every behavior change and keep all user-facing text and code comments in English.
- Describe statistical assumptions and limitations in documentation, not only in code.

## Reporting issues

When filing a detector issue, include the Python version, package version, sanitized run manifests, sample counts, and the command used. Never include API keys or sensitive prompt/response data.

