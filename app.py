import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Exact file name as provided
MODEL_FILE = "naive_model_b14.pkl"
model = None

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Churn Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0b0f19;
            --surface: #131b2e;
            --surface-hover: #1c2640;
            --border: #23314e;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.25);
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
            max-width: 680px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }

        .card-header {
            margin-bottom: 2rem;
            text-align: center;
        }

        .card-header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .card-header p {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.4rem;
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
            box-shadow: 0 0 0 3px var(--accent-glow);
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
            transition: opacity 0.2s, transform 0.1s;
        }

        .submit-btn:hover {
            opacity: 0.92;
        }

        .submit-btn:active {
            transform: scale(0.99);
        }

        .result-container {
            margin-top: 1.75rem;
            padding: 1.25rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.1rem;
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
            margin-top: 0.4rem;
            font-size: 0.875rem;
            font-weight: 500;
            opacity: 0.9;
        }

        @media (max-width: 640px) {
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
            <p>GaussianNB Inference Service</p>
        </div>

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
                        <option value="3" selected>
