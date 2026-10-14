import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Global model container and diagnostic status
model = None
load_status_message = "Initializing..."

def load_ml_model():
    global model, load_status_message
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check potential names for the pickle file
    candidate_names = [
        "naive_model_b14.pkl",
        "NAIVE.PKL",
        "naive.pkl"
    ]
    
    model_path = None
    for name in candidate_names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            model_path = p
            break

    if not model_path:
        # Search recursively in case the file is inside a subfolder
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower() in ["naive_model_b14.pkl", "naive.pkl"]:
                    model_path = os.path.join(root, file)
                    break
            if model_path:
                break

    if not model_path:
        load_status_message = "File 'naive_model_b14.pkl' not found in project directory."
        print(f"[ERROR] {load_status_message}", file=sys.stderr)
        return

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        load_status_message = f"Model loaded successfully from: {os.path.basename(model_path)}"
        print(f"[INFO] {load_status_message}", file=sys.stdout)
    except Exception as exc:
        load_status_message = f"Failed to unpickle model: {str(exc)}"
        print(f"[ERROR] {load_status_message}", file=sys.stderr)

# Run loader inside a safe try-block so the server doesn't exit with status 1
load_ml_model()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Turnover Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #090d16;
            --surface: #121929;
            --border: #1e293b;
            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --text-heading: #f8fafc;
            --text-muted: #94a3b8;
            --stay-bg: rgba(16, 185, 129, 0.12);
            --stay-border: #10b981;
            --stay-text: #6ee7b7;
            --leave-bg: rgba(239, 68, 68, 0.12);
            --leave-border: #ef4444;
            --leave-text: #fca5a5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-body);
            color: var(--text-heading);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2.5rem 1rem;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 1rem;
            width: 100%;
            max-width: 650px;
            padding: 2.5rem;
            box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.7);
        }

        .card-header {
            margin-bottom: 1.75rem;
            text-align: center;
        }

        .card-header h1 {
            font-size: 1.6rem;
            font-weight: 700;
        }

        .card-header p {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }

        .notice-banner {
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1.25rem;
            font-size: 0.85rem;
            text-align: center;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #fca5a5;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            background-color: var(--bg-body);
            border: 1px solid var(--border);
            color: var(--text-heading);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25);
        }

        select option {
            background-color: var(--bg-body);
            color: var(--text-heading);
        }

        .submit-btn {
            margin-top: 1.5rem;
            width: 100%;
            padding: 0.85rem;
            border-radius: 0.5rem;
            border: none;
            background-color: var(--accent);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        .submit-btn:hover {
            background-color: var(--accent-hover);
        }

        .result-container {
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.05rem;
            font-weight: 600;
        }

        .result-container.leave {
            background: var(--leave-bg);
            border: 1px solid var(--leave-border);
            color: var(--leave-text);
        }

        .result-container.stay {
            background: var(--stay-bg);
            border: 1px solid var(--stay-border);
            color: var(--stay-text);
        }

        .confidence {
            display: block;
            margin-top: 0.35rem;
            font-size: 0.85rem;
            font-weight: 500;
            opacity: 0.9;
        }

        @media (max-width: 600px) {
            .grid {
                grid-template-columns: 1fr;
            }
            .card {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="card-header">
            <h1>Employee Turnover Predictor</h1>
            <p>GaussianNB Inference Model</p>
        </div>

        {% if not is_ready %}
            <div class="notice-banner">
                {{ status_msg }}
            </div>
        {% endif %}

        <form method="POST" action="/">
            <div class="grid">
                <div class="field">
                    <label for="Education">Education</label>
                    <select id="Education" name="Education" required>
                        <option value="0">Bachelors</option>
                        <option value="1">Masters</option>
                        <option value="2">PHD</option>
                    </select>
                </div>

                <div class="field">
                    <label for="JoiningYear">Joining Year</label>
                    <input type="number" id="JoiningYear" name="JoiningYear" min="2000" max="2035" value="2017" required>
                </div>

                <div class="field">
                    <label for="City">City</label>
                    <select id="City" name="City" required>
                        <option value="0">Bangalore</option>
                        <option value="1">Pune</option>
                        <option value="2">New Delhi</option>
                    </select>
                </div>

                <div class="field">
                    <label for="PaymentTier">Payment Tier</label>
                    <select id="PaymentTier" name="PaymentTier" required>
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3" selected>Tier 3</option>
                    </select>
                </div>

                <div class="field">
                    <label for="Age">Age</label>
                    <input type="number" id="Age" name="Age" min="18" max="75" value="28" required>
                </div>

                <div class="field">
                    <label for="Gender">Gender</label>
                    <select id="Gender" name="Gender" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="field">
                    <label for="EverBenched">Ever Benched</label>
                    <select id="EverBenched" name="EverBenched" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="field">
                    <label for="ExperienceInCurrentDomain">Experience (Years)</label>
                    <input type="number" id="ExperienceInCurrentDomain" name="ExperienceInCurrentDomain" min="0" max="40" value="3" required>
                </div>
            </div>

            <button type="submit" class="submit-btn" {% if not is_ready %}disabled style="opacity: 0.5; cursor: not-allowed;"{% endif %}>
                Run Prediction
            </button>
        </form>

        {% if prediction is not none %}
            {% if prediction == 1 %}
                <div class="result-container leave">
                    Prediction: High Risk (Likely to Leave)
                    {% if probability %}
                        <span class="confidence">Confidence: {{ probability }}%</span>
                    {% endif %}
                </div>
            {% else %}
                <div class="result-container stay">
                    Prediction: Low Risk (Likely to Stay)
                    {% if probability %}
                        <span class="confidence">Confidence: {{ probability }}%</span>
                    {% endif %}
                </div>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST" and model is not None:
        feature_order = [
            "Education",
            "JoiningYear",
            "City",
            "PaymentTier",
            "Age",
            "Gender",
            "EverBenched",
            "ExperienceInCurrentDomain"
        ]

        try:
            input_data = [float(request.form.get(col, 0)) for col in feature_order]
            features = np.array([input_data])

            if hasattr(model, "feature_names_in_"):
                features_df = pd.DataFrame(features, columns=model.feature_names_in_)
            else:
                features_df = features

            pred = model.predict(features_df)[0]
            prediction = int(pred)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features_df)[0]
                probability = round(float(probs[prediction]) * 100, 2)
        except Exception as e:
            print(f"[ERROR] Inference error: {e}", file=sys.stderr)

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        probability=probability,
        is_ready=(model is not None),
        status_msg=load_status_message
    )

# Render health check route
@app.route("/healthz")
def health_check():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
