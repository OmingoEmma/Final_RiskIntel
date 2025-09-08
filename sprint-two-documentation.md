## Sprint Two: Development Methodology, Results, Evaluation, and Critical Analysis Pack

### How to integrate into your dissertation
- Place the Methodology content into Chapter 3 (or your equivalent Methods chapter), subsection “Sprint Two Methodology”.
- Place the Results content into Chapter 4/5 “Results” with subsections for model performance and explainability.
- Place the Evaluation Framework either at the end of Methods or as the opening subsection of Results.
- Place the Critical Analysis into “Discussion” (or “Analysis and Discussion”).
- Keep the Academic Writing Templates in Appendix (or reuse inline as needed).

---

## 1. Methodology (Sprint Two)

### 1.1 Sprint overview
- Sprint ID: [Sprint Two]
- Dates: [Start Date] – [End Date]
- Objectives: [e.g., improve model generalisation; integrate explainability; harden data pipeline]
- Scope statement: [concise definition of Sprint Two boundaries]
- Team/roles: [roles and responsibilities]

### 1.2 Agile process and governance
- Ceremonies: sprint planning, daily stand‑ups, backlog refinement, sprint review/demo, retrospective.
- Artefacts: product backlog, sprint backlog, increment, burndown chart, definition of done.
- Estimation: [e.g., story points via Planning Poker] and velocity tracking: [value].
- Tooling: [Jira/GitHub Projects], [Git], CI/CD: [GitHub Actions/GitLab CI], MLOps: [MLflow/DVC], experimentation: [Weights & Biases/Neptune].

### 1.3 Epic → Activities mapping
Use or adapt the table below to document your actual Epics and the concrete development activities completed in Sprint Two.

| Epic ID | Epic name | User stories (IDs) | Key development activities | Deliverables | Evidence (links/artifacts) | Acceptance criteria |
|---|---|---|---|---|---|---|
| EPIC-2.1 | [Data Pipeline Hardening] | [US-2.1.1, US-2.1.2] | [e.g., schema enforcement, drift checks, data validation tests] | [validated dataset v2, data validation report] | [PR-123, MLflow run-456, DVC tag v2] | [all checks pass ≥ 95%; no critical data issues] |
| EPIC-2.2 | [Model Iteration and Explainability] | [US-2.2.1, US-2.2.2] | [hyperparameter search; SHAP integration; calibration] | [model v1.2, SHAP dashboards] | [PR-124, run-457, notebooks/exp_12.ipynb] | [AUC ≥ 0.85; ECE ≤ 0.05] |
| EPIC-2.3 | [Interpretability UI/UX] | [US-2.3.1] | [UI for per‑prediction explanation; counterfactuals] | [UI v0.4] | [PR-125, demo video link] | [usability score ≥ 70 SUS] |
| EPIC-2.4 | [MLOps & Evaluation Automation] | [US-2.4.1, US-2.4.2] | [CI tests; model evaluation workflow; reporting] | [CI pipeline; nightly eval report] | [CI job #789, artifact report.pdf] | [pipeline pass rate ≥ 95%] |

### 1.4 Per‑Epic method write‑ups (template)
For each Epic, include a concise methods narrative using the template below.

#### [EPIC-2.x Title]
- Goal: [problem the Epic addresses]
- Rationale: [why it matters to research aims]
- User stories: [US-… with brief descriptions]
- Methods and tools: [algorithms, libraries, configurations]
- Data and preprocessing: [sources; filters; balancing; leakage mitigations]
- Experiment design: [CV strategy; temporal holdout; seeds; ablations]
- Implementation details: [key modules; interfaces; infra]
- Quality assurance: [unit/integration tests; data validation]
- Acceptance criteria & Definition of Done: [objective thresholds and DoD]
- Risks and mitigations: [top risks and how addressed]
- Evidence: [commits/PRs; MLflow runs; dataset versions]

### 1.5 Agile reflection for Sprint Two
- What went well: [e.g., reduced cycle time; clearer acceptance criteria]
- What was challenging: [e.g., data drift; scope creep]
- Changes since Sprint One: [process/architecture improvements]
- Retrospective actions: [action items, owners, expected impact]
- Metrics: velocity [x], carry‑over [y], defect escape rate [z], PR lead time [t].

> Citation template for Scrum: [Scrum Guide, 2020] or similar; include full reference in References.

---

## 2. Results (Templates with placeholders)

### 2.1 Data and experimental setup
- Datasets: [name, version, provenance, license]
- Splits: [train/val/test percentages or temporal windows]
- Preprocessing: [imputation, scaling, encoding, deduplication]
- Environment: [hardware; OS; Python version; library versions]
- Reproducibility: [random seeds, deterministic ops, run registry IDs]

### 2.2 Model versions compared
| Model ID | Model family | Training data | Key hyperparameters | Features | Notes |
|---|---|---|---|---|---|
| M‑1.0 | [e.g., XGBoost] | [split/version] | [n_estimators, depth, lr] | [feature set v2] | [baseline] |
| M‑1.1 | [XGBoost calibrated] | [split/version] | [params] | [feature set v2] | [post‑hoc calibration] |
| M‑1.2 | [TabNet/NN] | [split/version] | [params] | [feature set v2] | [regularisation changes] |

### 2.3 Primary performance metrics
Select the relevant block(s) and remove others.

#### Classification metrics
- AUC‑ROC: [value ± CI]
- Average Precision (PR‑AUC): [value ± CI]
- Accuracy / Balanced Accuracy: [value]
- F1 (macro/micro/weighted): [values]
- Calibration: ECE [value]; Brier score [value]
- Confusion matrix @ threshold τ = [value]: [insert figure/table]

#### Regression metrics
- MAE: [value ± CI]
- RMSE: [value ± CI]
- R²: [value]
- MAPE / sMAPE: [value]

#### Time‑series / forecasting (if applicable)
- MASE: [value]
- sMAPE: [value]
- Coverage of prediction intervals: [value]

### 2.4 Secondary analyses
- Ablations: [which components removed/added and effect]
- Sensitivity to data shift: [train‑test Δ distribution; performance drift]
- Fairness slices: [metric deltas across groups]
- Error analysis: [frequent error modes; top‑k misclassifications]

### 2.5 Explainability results (SHAP and related)
- Global importance (mean |SHAP|): [insert bar plot]
- SHAP beeswarm (all features): [insert plot]
- Dependence plots: [top‑n features]
- Waterfall plot (reference instance): [insert figure]
- Interaction values (if computed): [notes/plot]
- Faithfulness checks: [insertion/deletion AUC]

Placeholders for figures:
- Figure X: Global SHAP bar plot for model [ID].
- Figure Y: SHAP beeswarm showing distribution of per‑feature impacts.
- Figure Z: Dependence plot for feature [f], colored by [g].

### 2.6 Efficiency and operations
- Training time (per model): [values]
- Inference latency (p50/p95): [values]
- Memory footprint: [values]
- CI/CD metrics: pipeline pass rate [value], test coverage [value]
- Data pipeline health: validation pass rate [value]

### 2.7 Result interpretation
- Practical significance: [effect sizes; business/clinical relevance]
- Comparison to baseline/literature: [citations]
- Limitations of results: [caveats]

---

## 3. Evaluation Framework

### 3.1 Performance evaluation
- Data partitioning: [k‑fold CV / temporal holdout / nested CV]; seeds: [values]
- Metrics: [primary and secondary as per task]
- Statistical testing: [bootstrap CIs; DeLong for ROC; McNemar for paired classification; paired t‑test/Wilcoxon for regressors]
- Threshold selection: [Youden’s J; cost‑sensitive utility; operating‑point constraints]
- Robustness: [stress tests; noise injection; subgroup performance]

### 3.2 Explainability evaluation
- Sanity checks: [randomisation test; model/label shuffling]
- Stability: [variation of explanations across seeds/samples]
- Faithfulness: [insertion/deletion metrics; monotonicity checks]
- Human‑grounded eval: [do explanations help novices/experts complete tasks faster/more accurately?]
- Documentation: [Model Card; Explainability report]

### 3.3 Fairness and risk controls
- Group definitions: [sensitive attributes]
- Metrics: [demographic parity Δ; equal opportunity Δ; calibration within groups]
- Mitigations: [reweighing; threshold adjustments; post‑processing]
- Governance: [data protection, DPIA/ethics approvals, audit trail]

### 3.4 User evaluation and feedback
- Participants: [N, recruitment, inclusion criteria]
- Protocol: [tasks; think‑aloud; scenario]
- Instruments: SUS, UMUX‑Lite, Trust in Automation scale, custom Likert items
- Quant KPIs: task success rate, time‑on‑task, error rate, perceived usefulness
- Qual analysis: [thematic coding approach]

### 3.5 Decision criteria
- Promotion rule: [e.g., AUC ≥ 0.85 and ECE ≤ 0.05 and no fairness Δ > 0.05]
- Rollback rule: [conditions triggering rollback]
- Sign‑off: [roles and responsibilities]

---

## 4. Critical Analysis (Discussion structure)

### 4.1 Strengths
- Methodological: [sound evaluation; reproducibility]
- Technical: [robust pipeline; performance gains]
- Process: [effective agile practices; improved DoD]

### 4.2 Limitations
- Data: [size, representativeness, drift]
- Model: [overfitting risks; external validity]
- Explainability: [approximation error; user misinterpretation]
- Tooling: [constraints; compute limits]

### 4.3 Threats to validity
- Internal validity: [confounding, leakage]
- External validity: [generalisation to other settings]
- Construct validity: [metric‑goal alignment]
- Conclusion validity: [statistical power, multiple comparisons]

### 4.4 Ethical and societal considerations
- Privacy and consent: [data governance; minimisation]
- Fairness and bias: [harm analysis; mitigations]
- Transparency and accountability: [documentation; on‑call/runbooks]
- Misuse risks and safeguards: [abuse scenarios; rate‑limiting; monitoring]

### 4.5 Lessons learned and future work
- Architectural and process improvements: [what to change next]
- Research directions: [what to investigate in Sprint Three]

---

## 5. Academic writing templates

### 5.1 Section openers (templates)
- Methods opener: “This chapter details the Sprint Two methodology, mapping Epics to concrete development activities, experimental design, and quality assurance procedures.”
- Results opener: “This section presents empirical results for Sprint Two, including primary performance metrics, explainability analyses (SHAP), and efficiency measurements.”
- Discussion opener: “We critically evaluate the Sprint Two outcomes, considering strengths, limitations, and ethical implications with respect to the research aims.”

### 5.2 Figure and table captions
- Figure: “Figure [number]. [Short description]. Model: [ID]. Data: [split/version]. Error bars denote [CI method].”
- Table: “Table [number]. [Short description]. Bold indicates best result per column.”

### 5.3 Algorithm / equation formatting
- Use displayed math for key formulae (numbered): \[ \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} | \text{acc}(B_m) - \text{conf}(B_m) | \]
- Pseudocode block (if needed):

```
Algorithm 1: [Name]
Input: [inputs]
Output: [outputs]
1: [step]
2: [step]
```

### 5.4 Citation templates (in‑text and BibTeX)
- In‑text (Harvard): “[Tool/Method] (Author Surname, Year)” or “Author Surname (Year)”.
- In‑text (IEEE): “[n]”.

BibTeX skeletons (replace fields as appropriate):

```bibtex
@inproceedings{lundberg2017shap,
  title={A Unified Approach to Interpreting Model Predictions},
  author={Lundberg, Scott M and Lee, Su-In},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A Scalable Tree Boosting System},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={KDD},
  year={2016}
}

@article{pedregosa2011sklearn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, Fabian and others},
  journal={JMLR},
  year={2011}
}

@inproceedings{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and others},
  booktitle={NeurIPS},
  year={2019}
}

@inproceedings{akiba2019optuna,
  title={Optuna: A Next-generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and others},
  booktitle={KDD},
  year={2019}
}

@misc{scrumguide2020,
  title={The Scrum Guide},
  author={Schwaber, Ken and Sutherland, Jeff},
  year={2020},
  howpublished={\url{https://scrumguides.org}}
}
```

Software citation (template):
- “[Software] v[version] ([Year]). [Publisher/Org]. Available at: [URL]. Accessed [Date].”

### 5.5 Writing and style checklist (MSc)
- Academic tone; use past tense for completed methods; present tense for general truths.
- Define research questions/hypotheses explicitly before results.
- Report effect sizes and confidence intervals alongside p‑values.
- State inclusion/exclusion criteria for data and participants.
- Disclose limitations and potential biases.
- Ensure reproducibility: environment, seeds, data versions, code links.

---

## 6. Placeholders to fill before submission
- Replace bracketed placeholders throughout with Sprint Two specifics.
- Insert links to PRs, experiment runs, datasets, and CI artifacts.
- Paste/export SHAP figures and performance plots in the nominated subsections.
- Update decision criteria with final thresholds based on your domain context.

---

## 7. References (examples to adapt)
- Add full references corresponding to in‑text citations and BibTeX above.

