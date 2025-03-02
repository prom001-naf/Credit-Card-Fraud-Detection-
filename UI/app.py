import pandas as pd
from flask import Flask, render_template, request, jsonify
import plotly.express as px
import folium
import joblib
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

# --- Data Preparation ---
file_name = "credit_card_transactions.csv"
df = pd.read_csv(file_name, parse_dates=["trans_date_trans_time", "dob"])
df["is_fraud"] = df["is_fraud"].astype(int)
df["month"] = df["trans_date_trans_time"].dt.to_period("M").astype(str)
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

# Pre-calculate the default age (mean age from training data)
default_age = df["age"].mean()

# --- Fraud Trend Analysis ---
fraud_counts = df[df["is_fraud"] == 1].groupby("month").size().reset_index(name="counts")
fig_trend = px.line(fraud_counts, x="month", y="counts", title="Fraudulent Transactions Trend", markers=True)
trend_chart = fig_trend.to_html(full_html=False)

# --- Age Distribution ---
fig_age_box = px.box(df, x="is_fraud", y="age", title="Age Distribution by Fraud Status")
age_box_chart = fig_age_box.to_html(full_html=False)

# --- Fraud by Gender ---
if "gender" in df.columns:
    fig_gender_count = px.histogram(df, x="gender", color="is_fraud", barmode="group", title="Fraud Cases by Gender")
    gender_count_chart = fig_gender_count.to_html(full_html=False)
    fraud_rate = df.groupby("gender")["is_fraud"].mean().reset_index(name="fraud_rate")
    fraud_rate["fraud_rate"] *= 100
    fig_gender_rate = px.bar(fraud_rate, x="gender", y="fraud_rate", title="Fraud Rate by Gender (%)")
    gender_rate_chart = fig_gender_rate.to_html(full_html=False)
else:
    gender_count_chart = gender_rate_chart = "<p>No gender data available.</p>"

# --- Folium Map ---
def create_folium_map():
    world_map = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start=4)
    for _, row in df[df["is_fraud"] == 1].iterrows():
        folium.Marker(
            [row["lat"], row["long"]],
            popup=f"<b>City:</b> {row['city']}<br><b>State:</b> {row['state']}",
            icon=folium.Icon(color="red")
        ).add_to(world_map)
    return world_map._repr_html_()

folium_map = create_folium_map()

# --- Load City Population Data ---
with open("city_population.json", "r") as f:
    city_population_dict = json.load(f)

# --- Load Model ---
model = joblib.load("fraud_model.joblib")

# --- Flask Routes ---
@app.route("/")
def index():
    cities = list(city_population_dict.keys())
    return render_template(
        "dashboard.html",
        trend_chart=trend_chart,
        age_box_chart=age_box_chart,
        gender_count_chart=gender_count_chart,
        gender_rate_chart=gender_rate_chart,
        folium_map=folium_map,
        cities=cities
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        amt = float(request.form.get("amt", 0))
        datetime_str = request.form.get("transaction_datetime", "")
        trans_dt = datetime.fromisoformat(datetime_str) if datetime_str else datetime.now()
        day_of_week = trans_dt.weekday()  # 0=Monday, 6=Sunday
        hour_of_day = trans_dt.hour
        age = default_age
        selected_city = request.form.get("city", "")
        city_pop = float(city_population_dict.get(selected_city, 0))
        categories = [
            "category_entertainment", "category_food_dining", "category_gas_transport",
            "category_grocery_net", "category_grocery_pos", "category_health_fitness",
            "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos",
            "category_personal_care", "category_shopping_net", "category_shopping_pos",
            "category_travel"
        ]
        selected_category = request.form.get("selected_category", "")
        category_features = [1 if cat == selected_category else 0 for cat in categories]
        features = [amt, city_pop, age, day_of_week, hour_of_day] + category_features
        feature_array = np.array(features).reshape(1, -1)
        fraud_prob = model.predict_proba(feature_array)[0][1] * 100
        classification = "Fraudulent" if fraud_prob >= 50 else "Not Fraudulent"
        result = {"probability": f"{fraud_prob:.2f}%", "classification": classification}
    except Exception as e:
        result = {"error": str(e)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
