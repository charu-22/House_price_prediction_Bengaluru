import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for
app=Flask(__name__)
data=pd.read_csv('Bengaluru_House_Data.csv')
pipe = pickle.load(open('Bengaluru_House_Price_Prediction_Model.pkl','rb'))
@app.route('/')
def home():
    locations = sorted(data['location'].dropna().astype(str).str.strip().unique())
    return render_template('home.html', locations=locations)
@app.route('/predict_price', methods=['POST'])
def predict():
    locations=request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bathrooms = int(request.form.get('bathrooms'))
    sqft = float(request.form.get('sqft'))
    print(locations, bhk, bathrooms, sqft)
    input_data = pd.DataFrame(
    [[locations, sqft, bathrooms, bhk]],
    columns=['location', 'total_sqft', 'bath', 'bhk']
)
    predicted_price = pipe.predict(input_data)[0]

    return str(predicted_price)
if __name__ == '__main__':
    app.run(debug=True,port=5001)