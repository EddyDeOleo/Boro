# Boro – Robotic Execution Intelligence

**Tagline:** *Define what should happen. See what actually happened. Verify the evidence.*

---

## Overview

Boro is a verification platform for robotic task execution. It analyses video recordings of a robot performing a task, extracts visual evidence using Meta's **SAM 3.1** segmentation model, interprets natural‑language task descriptions via **Muse Spark**, and produces a structured verification report that details observed events, criteria satisfaction, and an overall verdict.

The core of Boro is a **FastAPI** service (`src/api/main.py`) that wires together the following components:

- **Perception** – `MetaSAM31Client` uploads videos to the Meta Files API, streams SAM 3.1 segmentation results, and provides parsed observations.
- **Interpretation** – `MuseSparkClient` converts a textual task description into a formal task definition (entities, regions, criteria).
- **Verification Pipeline** – combines perception observations with the task definition to evaluate each verification criterion and produce a verdict (`VERIFIED`, `FAILED`, or `INSUFFICIENT_EVIDENCE`).


---

## Repository Structure

```text
Boro/
├─ .git/                     # Git metadata (repository root)
├─ .env.example              # Example environment file
├─ requirements.txt          # Python dependencies
├─ src/                      # Python source code
│   ├─ api/                  # FastAPI entry point and routes
│   ├─ application/          # Service layer (verification, execution, etc.)
│   ├─ perception/           # Meta SAM 3.1 client and providers
│   ├─ interpretation/       # Muse Spark client and task interpreter
│   └─ models/               # Domain models (events, criteria, reports)
├─ tests/                    # Unit and integration tests (currently import‑path errors)
├─ evaluation/               # JSON verification reports
│   └─ boro_episode_000025_report.json
├─ demo-video/               # Remotion front‑end (Node.js project)
├─ frontend/                 # Optional front‑end (not used by core API)
├─ README_READINESS_AUDIT.md # Internal audit documentation
├─ README.md                 # **This file**
└─ ... (other files)
```

The top‑level entry point for the API is `src/api/main.py`, which creates a FastAPI app (`app`). The repository is a standalone Git repository (`git rev-parse --show-toplevel` → `C:/Users/User/source/repos/Boro`).

---

## Environment Variables

Only the Meta API key is required for the backend services. Create a `.env` file (or set the variables in your environment) based on the example:

```text
META_API_KEY=your_meta_api_key_here
```

Both `MetaSAM31Client` and `MuseSparkClient` read the key from `META_API_KEY` (or `MODEL_API_KEY` if present).

---

## Building & Running the Backend

```bash
# Python 3.12 – install dependencies
python -m venv .venv
source .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the API (development mode)
uvicorn src.api.main:app --reload
```

The API exposes the following endpoints (see `src/api/routes/` for details):

- `POST /verification` – submit video and task description, receive a verification report.
- `POST /task_interpretation` – obtain a structured task definition from natural language.
- `POST /robotic_execution` – orchestrates execution verification.
- `GET /health` – health check.

**Note:** The test suite currently fails to import the `src` package because the repository root is not on `PYTHONPATH`. To run tests locally, either install the package (`pip install -e .`) or prepend the repository root to `PYTHONPATH`.

---

## Evaluation Report Example

The only verification report present in the repository is `evaluation/boro_episode_000025_report.json`. Its **verdict** is `INSUFFICIENT_EVIDENCE` because the mandatory criterion `stable_target_placement` was not satisfied.

```json
{
  "schema_version": "1.0",
  "report": {
    "overview": {
      "verdict": "INSUFFICIENT_EVIDENCE",
      "summary": "The available evidence or task coverage is insufficient to verify the complete task."
    },
    "criteria": [
      {"criterion_id": "coordinated_transport", "status": "SATISFIED"},
      {"criterion_id": "relative_separation", "status": "SATISFIED"},
      {"criterion_id": "stable_target_placement", "status": "INSUFFICIENT_EVIDENCE", "is_mandatory": true}
    ]
  }
}
```

The report demonstrates that Boro can detect events such as `OBJECT_MOVING_WITH_EFFECTOR` and `RELATIVE_SEPARATION`, but it correctly refrains from declaring a `VERIFIED` verdict when required evidence (stable placement) is missing.

---

## Test Videos

The repository includes a small set of demonstration videos used for evaluation. They are hosted on Hugging Face:

- https://huggingface.co/datasets/ExylosAi/pick_and_place_sample/blob/main/videos/chunk-000/observation.images.front_cam/episode_000011.mp4

You can download this external MP4 and upload it to Boro for testing.

---

## Limitations & Current State

- **Test Suite:** Import errors prevent the existing tests from running. Adjust the import path or install the package to enable testing.
- **Front‑End Build:** The Remotion demo has not been built; its success is unverified.
- **License:** No `LICENSE` file is present in the repository.
- **External Calls:** Both Meta SAM 3.1 and Muse Spark clients require valid API credentials; without them the services will raise configuration errors.

---

## Contributing

Contributions are welcome. When adding new functionality, please:

1. Update the `README.md` with verified information.
2. Add or adjust tests and ensure they pass with a proper `PYTHONPATH` or package installation.
3. Document any new environment variables in `.env.example`.

---

## Contact

For questions or assistance, open an issue in the repository or contact the maintainers.

---
