# Data Labs - Apache Beam Labs

Hands-on data engineering labs focused on Apache Beam pipelines, reproducible execution, and structured output artifacts.

## Changes Made for Data Labs (Apache Beam)

- Added production-style Beam scripts under `src/` instead of notebook-only logic
- Added `beam_wordcount.py` with configurable normalization, filtering, and top-N controls
- Added `beam_text_stats.py` to compute dataset-level quality and size metrics
- Standardized outputs into machine-readable artifacts (`csv`, `json`) under `outputs/`
- Switched setup and execution flow to `uv` (`uv sync`, `uv run`)
- Added local project config via `pyproject.toml` for dependency management

## Project Overview

This lab demonstrates how to build repeatable Apache Beam pipelines for text processing and profiling.

### Lab: King Lear Text Processing with Apache Beam

**Key Features:**
- Beam-based batch processing on local text input
- Configurable word counting (`lowercase`, `min-word-len`, `top-n`)
- Global text profiling (line count, token count, averages)
- Deterministic sorted outputs for easier testing and diffing
- `uv` workflow for consistent environment setup

## Current Structure

```text
Data_Labs/Apache_Beam_Labs/
  data/
    kinglear.txt
  outputs/
  src/
    beam_wordcount.py
    beam_text_stats.py
  Try_Apache_Beam_Python.ipynb
  pyproject.toml
  requirements.txt
  README.md
```

## Prerequisites

### For Local Development
- Python 3.11 or higher
- [uv package manager](https://docs.astral.sh/uv/)
- Git

## Getting Started

### Local Setup

1. **Navigate to the lab**
   ```bash
   cd Data_Labs/Apache_Beam_Labs
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

## Running Locally

### Pipeline 1: Word Count

```bash
uv run python src/beam_wordcount.py \
  --input data/kinglear.txt \
  --output outputs/wordcount \
  --lowercase \
  --min-word-len 3 \
  --top-n 50
```

Output:
- `outputs/wordcount.csv` (`word,count`)

### Pipeline 2: Text Stats

```bash
uv run python src/beam_text_stats.py \
  --input data/kinglear.txt \
  --output outputs/text_stats
```

Output:
- `outputs/text_stats.json` (single JSON line with aggregate metrics)

## Dependencies

- **apache-beam** - Distributed data processing SDK