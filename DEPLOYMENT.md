# Run and deploy EstateEstimate

The frontend and FastAPI backend run together. The browser uses relative `/options` and `/predict` requests; `app.py` is the entrypoint and static files are served at `/assets`.

## Local setup

Use Python 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/` or `/docs`. On Windows, activate with `.venv\Scripts\activate`.

**macOS:** XGBoost requires OpenMP. With Homebrew, use `brew install libomp`. This workspace also supports the copy bundled with scikit-learn. After activating the environment:

```bash
export DYLD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

The alternative path was verified on this Apple Silicon workspace. Vercel uses Linux and the CPU-only XGBoost wheel. [XGBoost installation documentation](https://xgboost.readthedocs.io/en/stable/install.html)

## Vercel deployment

1. Push the changes to GitHub and import the repository through **Add New → Project** on Vercel.
2. Choose your personal **Hobby** account and the **FastAPI** framework preset.
3. Keep **Root Directory** at the repository root, including `app.py`, `Backend`, `Frontend`, `Model`, and `requirements.txt`.
4. Keep build/install/output settings at their framework defaults. Do not enter a Uvicorn start command.
5. Deploy, then verify `/`, `/health`, `/docs`, and a prediction from the form. Check build/runtime logs if a request fails.

The checked-in configuration excludes data, notebooks, training code, the PDF, and evaluation metadata from the function bundle. Mounted static files are supported by Vercel's FastAPI integration. [Vercel FastAPI guide](https://vercel.com/docs/frameworks/backend/fastapi)

Hobby is free for personal, non-commercial use within its quotas; exceeding allowances can pause service. The standard Python bundle limit is currently 500 MB. The downloaded Linux runtime wheels total approximately 276 MB uncompressed before platform/model overhead. This is an estimate, not a verified Vercel build. [Hobby plan](https://vercel.com/docs/plans/hobby) · [Function limits](https://vercel.com/docs/functions/limitations)

## Training folder

The website does not execute `training/`. It loads the saved pipeline from `Model/best_model.pkl`.

- `training/data.py` applies fixed cleaning rules and rebuilds the processed CSVs.
- `training/train.py` compares four models, selects by validation error, evaluates a separate test set, and saves the pipeline, input options, and `Model/metrics.json`.
- `training/__init__.py` lets Python run the folder as a package.

To retrain, install the additional comparison dependency and run:

```bash
pip install catboost==1.2.10
python -m training.train
```

The metrics file includes evaluation scores, split row IDs, package versions and raw/model hashes. Notebooks use these shared functions/results. The six-page report is `technical-report.pdf`; regenerate its reported results if retraining changes them.

The app passed 16 Python regression tests and isolated desktop/mobile browser checks during the repair. Those supporting test files were subsequently removed at the user's request to simplify the repository. No live deployment or cloud account settings have been changed.
