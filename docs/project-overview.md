# Project overview

## Background

This project began as a hackathon proof of concept for automated lithology interpretation. The idea was to reduce the time needed to turn well-log measurements into a preliminary lithology prognosis while keeping expert interpretation as the decision boundary.

The original demonstration combined data acquisition experiments, exploratory analysis, missing-data handling, several classification algorithms, and a small interactive interface. This repository preserves the generic workflow without retaining private systems, branded material, credentials, or restricted data.

## Intended use cases

- Explore well-log measurements by well, formation, and depth.
- Identify missing measurements before model training.
- Review the effect of a simple caliper-versus-bit-size quality filter.
- Compare several supervised classification algorithms on labeled records.
- Apply the selected model to another compatible tabular dataset.
- Export predicted lithology labels for offline review.

## Workflow

1. Load labeled CSV or Excel data.
2. Select the wells or formations relevant to the experiment.
3. Review missing values and measurement distributions.
4. Apply optional row filtering where caliper and bit-size measurements support it.
5. Split labeled data into training and test partitions.
6. Fit preprocessing and each classifier together in a pipeline.
7. Compare holdout accuracy and retain the highest-scoring candidate.
8. Predict lithology for data with the same measurement columns.
9. Download the resulting table for review.

## Public reconstruction

The original hackathon material evaluated K-Nearest Neighbors, Support Vector Machines, Random Forest, and gradient-boosted trees. The current demo keeps three scikit-learn classifiers so setup remains small and model preprocessing stays consistent. The historical live-data acquisition and service-side model loading paths are intentionally absent.

Synthetic records provide a reproducible way to exercise the interface. They use fictional wells and formations and do not represent calibrated geological relationships.

## Evaluation boundaries

The model comparison uses one stratified holdout split. That is enough to demonstrate the workflow but not enough to establish production quality. A serious evaluation would need source provenance, domain-reviewed labels, unit definitions, well-level separation between training and validation data, class-specific metrics, calibration checks, and validation on unseen wells.

## Privacy and publication boundary

Do not add credentials, private service addresses, employee information, private screenshots, restricted datasets, serialized models from unknown sources, or documents without redistribution permission. Use synthetic fixtures or datasets with explicit public terms and documented provenance.
