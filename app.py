from flask import Flask
import pickle
import numpy as np
from flask import render_template, request, redirect, url_for
from flask import flash

app = Flask(__name__)
app.secret_key = "Rafnas"

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')   

@app.route('/predict',methods = ['GET','POST'])
def predictions():
    if request.method == "POST":
        bedrooms = request.form['bedrooms']
        bathrooms = request.form['bathrooms']
        floors = request.form['floors']
        yr_built = request.form['yr_built']
        

        if bedrooms == "" or bathrooms == "":
                flash("Invalid: Every field is required.")
                return redirect(url_for("index"))
        if floors == "" or yr_built == "":
                flash("Invalid: Every field is required.")
                return redirect(url_for("index"))

        arr = np.array([bedrooms,bathrooms,floors,yr_built])
        arr = arr.astype(np.float64)

        prediction = model.predict([arr])

    return render_template('index.html', data=int(prediction)) 

if __name__=='__main__':
    app.run(debug=True)