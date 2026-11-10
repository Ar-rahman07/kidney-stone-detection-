# Kidney Stone Detection and Prevention

A student academic project that combines deep-learning image classification and machine-learning analysis of urine parameters to explore kidney-stone detection and prevention support.

## Project overview

The application is designed to:
- Accept a kidney CT scan or ultrasound image.
- Classify images using separate TensorFlow/Keras CNN models.
- Use OpenCV for image preprocessing and visual highlighting.
- Analyse urine parameters using a scikit-learn Logistic Regression model.
- Present prediction results and hydration/risk guidance.
- Generate a PDF summary report.

**Important medical disclaimer:** This is an academic prototype, not a clinically validated diagnostic system. It must not be used to diagnose, treat, or rule out kidney stones. Always consult a qualified healthcare professional.

## Technology stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| TensorFlow / Keras | CNN image classification |
| OpenCV | Image processing |
| NumPy and Pandas | Data handling |
| scikit-learn | Urine-parameter machine learning |
| FPDF | PDF report generation |

## Repository structure

```text
Kidney-Stone-Detection/
├── app.py
├── requirements.txt
├── README.md
├── ct_kidney_stone_model.h5       # add locally; not included here
├── usg_kidney_stone_model.h5      # add locally; not included here
└── kidney_stone_urine_3000.csv    # add locally; not included here
```

The model weights and urine CSV dataset are not included in this repository. The application will need the correct files in the paths expected by `app.py` before all features can run. Do not upload private credentials, API keys, passwords, patient-identifying data, or files such as `firebase_key.json`.

## Installation

Use a Python version supported by your installed TensorFlow release (Python 3.10 or 3.11 is a practical starting point).

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the application

From the directory containing `app.py`:

```bash
python -m streamlit run app.py
```

## Urine dataset columns

The urine-analysis CSV is expected to contain these columns (check your code and dataset match):

- `gravity`
- `ph`
- `urea`
- `calc`
- `osmo`
- `target`

If `app.py` currently points to a computer-specific absolute path, update it to a project-relative path before sharing, for example:

```python
CSV_PATH = "kidney_stone_urine_3000.csv"
```

## Before publishing

- Review the repository for passwords, tokens, private keys, and personal data.
- Keep large trained model files and datasets out of GitHub unless you have checked the current size limits and have permission to share them.
- Update this README if the actual filenames or setup steps in your final code differ.

## Author

Abdur Rahman
