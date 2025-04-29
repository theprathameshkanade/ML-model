from flask import Flask, render_template, request 
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index0.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = {
        "speed_limit": int(request.form['speed_limit']),
        "vehicle_speed": int(request.form['vehicle_speed']),
        "time_of_day": request.form['time_of_day'],
        "weather": request.form['weather'],
        "road_type": request.form['road_type'],
        "traffic_density": float(request.form['traffic_density']),
        "driver_history": request.form['driver_history'],
        "area_sensitivity": float(request.form['area_sensitivity']),
    }

    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Custom message based on probability range
    if probability > 0.7:
        result = f"⚠️ Fine Likely – You were well over the limit!"
    elif 0.5 < probability <= 0.7:
        result = f"❗Warning – Speed nearing fine zone!"
    elif 0.1 < probability <= 0.5:
        result = f"ℹ️ Caution – You were fast, please slow down."
    else:
        result = f"✅ Fine Unlikely"

    return render_template('index0.html', result=result, probability=f"{probability:.2f}")


if __name__ == '__main__':
    app.run(debug=True)
