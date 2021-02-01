import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

#Read the pickled representation of an object and
#return the reconstituted object heirarchy
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #getting the input features from the form in request.form.values()
    #and converting them to a float numpy array
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    #feeding the input array to the predict function
    predictions=model.predict(final_features)
    return render_template('index.html',prediction=predictions)

if __name__ == "__main__":
    app.run(debug=True)