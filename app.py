
from flask import Flask, render_template, request
import pickle
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import os

app = Flask(__name__)
model = pickle.load(open("model/student_model.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html", result=None, graph=None)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        absences = int(request.form['absences'])

        data = np.array([[studytime, failures, absences]])
        result = model.predict(data)[0]

        bar = go.Figure(data=[go.Bar(
            x=["Studytime", "Failures", "Absences"],
            y=[studytime, failures, absences]
        )])
        graph_html = pyo.plot(bar, output_type='div')

        return render_template("index.html", result=result, graph=graph_html)
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
