## Technology Stack and Rationale

### Languages and Runtime

- Python 3.10+: Mature data ecosystem, ML libraries, and strong tooling for testing and packaging.

### Core Libraries

- Streamlit: Rapid UI for data apps; simple deployment and interactive dashboards.
- pandas / numpy: Data manipulation and numerical computing.
- scikit-learn: Solid baselines and production-ready classical ML.
- SHAP: Model-agnostic explainability; TreeExplainer for tree models.
- requests: Reliable HTTP client for ingestion.
- NLTK (VADER): Lightweight sentiment baseline for preprocessing robustness.

### Persistence

- SQLite: Embedded database for demos and small deployments; easy to vendor and backup.

### Testing and Quality

- pytest, pytest-cov, coverage: Unit tests and coverage reporting.

### CI/CD

- GitHub Actions: Portable CI with Python matrix, coverage upload, and room for Docker builds.

### Containerization

- Docker: Reproducible runtime; pins dependencies and simplifies deployment.

### Observability

- Logging via `logging` module; file + console handlers in ingestion pipeline.

### Security Considerations

- Secrets via environment variables (`.env` not committed). Example template provided in `.env.example`.
- Network calls constrained with timeouts and retry/backoff.

