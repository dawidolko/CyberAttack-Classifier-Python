# AGENTS.md

Guide for agents working on the cyber attack classifier. Add a section per area as conventions
emerge — do not pad sections with content that is not established yet.

## What this project is

A supervised multiclass classification study on a Kaggle network-attack
dataset, using a random forest as the primary model with gradient boosting and
k-NN for comparison, and a Streamlit dashboard for the analysis.

Read [README.md](README.md) first — it documents the architecture and the
commands used day to day.

## Language

**UI copy and code comments are English.** Labels, buttons, validation messages,
page titles, empty states and error text are all English. User-supplied data is
rendered exactly as entered and never normalised.

## Comments

Comments explain **why**, not what. The code already says what it does.

- Use multi-line block comments for anything that needs explaining; avoid
  trailing one-line comments tacked onto the end of a statement.
- A comment that restates the code is deleted rather than reworded.
- Document the constraint, the trade-off or the failure mode that made the code
  look the way it does — that is the part a reader cannot recover from the code.

## Experiments

Results must be reproducible. A fixed seed is set once and threaded through
every split and every estimator; changing it silently invalidates the numbers
already written up in the README.

Scaling and encoding happen **inside** the cross-validation pipeline so the
transform never sees the test fold. Fitting a scaler on the full frame before
splitting leaks the test distribution into training and inflates every metric.

Never quote a metric in prose that is not in a generated results file.

## Before finishing

- [ ] The experiment script runs end to end and regenerates `results/`.
- [ ] Every metric quoted in the README comes from a generated results file.
- [ ] The random seed is unchanged, or the README numbers were regenerated.
- [ ] No secrets or dataset credentials committed.
- [ ] Copy is English.
