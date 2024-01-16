from flask import Flask,request,render_template
import numpy as np
import pickle


# importing models
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))
sc1 = pickle.load(open('sc1.pkl','rb'))
ms1 = pickle.load(open('ms1.pkl','rb'))

#creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html', result=result)


@app.route('/index2.html')
def index2():
    return render_template('index2.html')

@app.route("/predict1", methods=['POST'])
def predict1():
    temp1 = request.form['Temperature1']
    humidity1 = request.form['Humidity1']
    moisture = request.form['moisture']
    soiltype = request.form['soiltype']
    croptype = request.form['croptype']
    N1 = request.form['Nitrogen1']
    P1 = request.form['Phosporus1']
    K1 = request.form['Potassium1']

    feature_list1 = [temp1,humidity1,moisture,soiltype,croptype,N1,K1,P1]
    single_pred1 = np.array(feature_list1).reshape(1, -1)

    scaled_features1 = ms1.transform(single_pred1)
    final_features1 = sc1.transform(scaled_features1)
    prediction1 = rf.predict(final_features1)

    fert_dict = {0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}

    if prediction1[0] in fert_dict:
        fert = fert_dict[prediction1[0]]
        result = "{} is the best Fertilizer to be used".format(fert)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index2.html', result=result)

#python main
if __name__ == "__main__":
    app.run(debug=True)