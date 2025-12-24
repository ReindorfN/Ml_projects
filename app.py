from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the entire pipeline (includes scaling, encoding, and the model)
model = joblib.load('xgb_logistics_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/project-info')
def project_info():
    return render_template('project_info.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture data from the HTML form
    form_data = {
        'price': [float(request.form['price'])],
        'freight_value': [float(request.form['freight_value'])],
        'product_weight_g': [float(request.form['product_weight_g'])],
        'product_volume_cm3': [float(request.form['product_volume_cm3'])],
        'product_density': [float(request.form['product_density'])],
        'delivery_distance_km': [float(request.form['delivery_distance_km'])],
        'purchase_month': [int(request.form['purchase_month'])],
        'purchase_day_of_week': [int(request.form['purchase_day_of_week'])],
        'purchase_hour': [int(request.form['purchase_hour'])],
        'seller_state': [request.form['seller_state']],
        'customer_state': [request.form['customer_state']]
    }

    # 2. Convert to DataFrame (Pipeline requires a DF to match feature names)
    input_df = pd.DataFrame(form_data)

    # 3. Make Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # Chance of being late

    result_text = "LATE" if prediction == 1 else "ON TIME"
    risk_percent = round(probability * 100, 4)

    return render_template('index.html', 
                           prediction_text=f'The order will likely be: {result_text}',
                           risk_text=f'Delay Risk: {risk_percent}%')

if __name__ == "__main__":
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For production (Render)
    # Gunicorn will handle the server
    pass
