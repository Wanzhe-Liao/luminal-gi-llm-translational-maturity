# Clinical-readiness assessment of large language model research in luminal gastrointestinal cancers: a systematic meta-research review

This repository is a source-only release for reproducible scripting. It contains selected public analysis and build scripts and schemas. Manuscript source text and submission DOCX files are maintained separately and are not redistributed here.

## Study overview

The study examines how much of the current luminal gastrointestinal cancer LLM literature is approaching clinically relevant use. Unlike recent cross-specialty clinical LLM reviews, it focuses on a disease-specific luminal GI oncology corpus, uses a full-report audit of the retained peer-reviewed studies, and asks which use cases fall into external-validation, low-risk human-review-only, or prospective-trial candidate categories under a prespecified readiness framework.

- `1,798` records were identified and `1,623` were screened after deduplication.
- `219` source reports entered full review.
- The primary analysis includes `53` peer-reviewed full-report studies.
- An expanded peer-reviewed comparison set of `155` studies is reported for supplementary sensitivity analyses.
- `27` supplementary source records from publication-status streams were reviewed for comparison, of which `18` remained unique after concept-level overlap matching against the expanded peer-reviewed comparison set.

## Headline findings

- Tier II-III evidence accounts for `32/53` (`60.4%`) of the primary peer-reviewed study set.
- No study reached deployed or randomized real-world use.
- All `53` retained studies were scored for TRIPOD-LLM; mean full 19-item completeness was `8.81/19`, and mean operational core-14 completeness was `8.34/14`.
- Across `97` clinical use scenarios after dual-review adjudication, `38` were not ready for clinical use, `37` were external-validation candidates, `16` were low-risk human-review-only uses, and `6` were prospective-trial candidates.
- Reported performance outcomes remained highly heterogeneous, so the manuscript emphasizes study-level range displays rather than pooled headline effects.

## Repository layout

- `scripts/` contains selected public analysis and build scripts; it is not a mirror of every internal submission-build or QC wrapper.
- `schemas/` contains source schema and configuration definitions used by the scripts.
