#importing required libraries
import pickle
from flask import Flask, request, render_template
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
from features import generate_data_set

gbc = pickle.load(open('final_model_all_features_70_30.sav', 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", xx= -1)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        start_time = time.time()
        url = request.form["url"]
        b= url.find('.')
        if not (b>0):
            return render_template('index.html',ab = (b>0),url=url)
        x = np.array(generate_data_set(url)).reshape(1,30) 
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        print("--- %s seconds ---" % (time.time() - start_time))
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)