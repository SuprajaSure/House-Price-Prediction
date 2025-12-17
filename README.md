
# House Price Predictor (Flask + scikit-learn)

Mini project that trains a regression model on scikit-learn's California Housing dataset and exposes a simple Flask web UI to predict house prices from user inputs.

## Files
- `train_model.py` - trains the model and saves `model.joblib` and `scaler.joblib`.
- `app.py` - Flask web app that loads the model and serves a form + prediction endpoint.
- `templates/index.html` - HTML form and result area.
- `static/style.css` - basic styling.
- `requirements.txt` - Python dependencies.

## Quick start
1. Create a virtualenv: `python -m venv venv && source venv/bin/activate` (on Windows: `venv\Scripts\activate`).
2. Install requirements: `pip install -r requirements.txt`.
3. Train the model: `python train_model.py` (creates `model.joblib` and `scaler.joblib`).
4. Run app: `python app.py`.
5. Open http://127.0.0.1:5000 in your browser.

## Notes
- Uses scikit-learn's `fetch_california_housing` so no external CSV needed.
- For deployment, replace the model or add better validation and security.

Note: Trained model files are not included in the repository due to size constraints.  
They are generated locally by running `train_model.py`.

