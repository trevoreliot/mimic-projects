# Project Context: Clinical Entity Linking Pipeline (MIMIC-III Proof of Concept)

## Overview
This project is a modular, hardware-agnostic Clinical Entity Linking (EL) and Named Entity Recognition (NER) pipeline. The initial goal is to build a Proof of Concept (PoC) using de-identified clinical notes from the **MIMIC-III dataset**, leveraging a dual-engine pipeline 1. **spaCy** and **medspaCy** and 2. SapBERT (along with other candidate libraries like Hugging Face `transformers` or `scispacy` if needed). 

The architecture must be strictly decoupled to support two critical requirements:
1.  **DataSource Agnostic:** The core NLP and extraction logic must not depend on MIMIC-III specifics. Once the PoC is validated, switching to an internal or alternative clinical data source should only require updating the ingestion layer.  The agent and llm will worry more about the pipeline, training, and analysis; the user will spend more time on the sql.
2.  **Hardware Agnostic (CUDA & Apple Silicon):** Model training and inference will initially run on **Nvidia GPUs (CUDA)**. However, downstream enterprise execution will occur on **Apple Silicon (MPS)**. Core libraries, device selection, and tensor operations must dynamically adapt to the available hardware accelerator without rewriting execution code.

---

## Technical Stack & Tooling

* **Language:** Python 3.11+
* **Package Management:** `uv` (utilizing `pyproject.toml` and workspace/environment isolation)
* **Environment Control:** `direnv` for automatic local path and environment variable loading.
* **Dual-Engine NLP Frameworks:** `spacy`, `medspacy`, `SapBERT`
* **Potential Model Libraries:** `transformers` (Hugging Face), `torch`
* **Database Backend:** PostgreSQL (for local data mirroring/caching)

---

## Architectural Principles & Layout

To ensure modularity for both data sources and compute backends, the codebase adheres to a strict layer separation:

```text
├── config/              # Hardware and pipeline configuration profiles (YAML/TOML)
├── data/                # Local data storage (Git ignored, MIMIC-III CSVs/Parquet)
├── src/
│   ├── ingestion/       # Data-source specific loaders (implements a standard interface)
│   ├── pipeline/        # Core NLP, NER, and Entity Linking logic (spaCy/medspaCy and SapBERT components)
│   ├── training/        # Model fine-tuning and training orchestrators
│   └── utils/           # Hardware detection, device assignment, and logging utilities
│   └── analysis/        # Additional analysis utilizing entity features along with other variables
│   └── viz/             # Visualization of analytics and entity linking results
├── pyproject.toml       # Managed via uv
└── README.md

