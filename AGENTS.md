# AGENTS.md

## Project Positioning

This repository is a self-hosted customization fork of PDFMathTranslate.

- Primary target environment: Ubuntu/Debian Linux Docker containers.
- Treat Docker usability, image reproducibility, and container runtime stability as first-class requirements.
- Windows EXE, macOS, and cloud one-click deploy flows are secondary unless they share the same code path or block Linux Docker usage.

## Project Map

- CLI entrypoint: `pdf2zh_next/main.py`
- Main translation orchestration: `pdf2zh_next/high_level.py`
- Config and argument parsing: `pdf2zh_next/config/`
- Translation backends and cache/rate limiting: `pdf2zh_next/translator/`
- Gradio WebUI: `pdf2zh_next/gui.py`
- Main Docker build: `Dockerfile`
- Other Docker variants: `script/Dockerfile.China`, `script/Dockerfile.Demo`
- Package metadata and dependencies: `pyproject.toml`
- Docs entrypoints: `README.md`, `docs/en/getting-started/`, `docs/zh/getting-started/`, `mkdocs.yml`

## Working Rules

- Read the relevant code before editing. For most tasks, start with `README.md`, `Dockerfile`, `pyproject.toml`, and the affected files under `pdf2zh_next/`.
- Prefer changes that keep both CLI mode and `pdf2zh --gui` usable inside headless Linux containers.
- When changing Docker or build logic, prefer deterministic and repeatable builds. Avoid adding brittle remote `ADD` steps or cache-busting network calls unless explicitly required.
- When adding system packages, use Debian/Ubuntu package names and keep images minimal with `--no-install-recommends` where practical.
- Assume BabelDOC warmup and asset/model downloads can affect image build time and container startup time. Account for that in Docker-related decisions.
- Do not spend time on Windows-only or macOS-only behavior unless the issue also affects shared Python code paths.
- Do not rely on manual browser interaction as the only verification path. Prefer CLI or container smoke checks.

## Documentation Rules

- `README.md` is the landing page for this fork and should keep the Linux Docker focus visible.
- If install, runtime, or configuration behavior changes, update the relevant Docker/build docs before finishing.
- At minimum, sync user-facing behavior changes into `docs/en` and `docs/zh` when those pages are affected. Other language docs may lag unless explicitly requested.
- If MkDocs navigation or page locations change, verify `mkdocs.yml` still points to the right files.

## Verification

- For config and model changes, start with targeted tests:
  - `uv run pytest tests/config test/test_cache.py`
- For package sanity:
  - `uv run pdf2zh --version`
- When Docker-related files change:
  - `docker build -t pdfmathtranslate-self .`
- For a container smoke check, prefer a non-interactive command such as:
  - `docker run --rm pdfmathtranslate-self pdf2zh --version`
- If docs structure or MkDocs config changes and the environment has docs dependencies installed:
  - `uv run mkdocs build`

## Practical Notes

- External translation providers often require API keys. Do not depend on live provider calls for routine verification.
- The repository may contain local, uncommitted user edits. Preserve unrelated changes.
- The current main image path is based on Debian Bookworm slim plus `uv`; keep Linux container assumptions aligned with that unless intentionally changing the base strategy.
