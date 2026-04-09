from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

model = joblib.load("salary_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    salary = None

    if request.method == 'POST':
        exp = float(request.form['exp'])

        # prediction
        prediction = model.predict([[exp]])
        salary = f"{prediction[0]:.2f} $"

        # график түзүү
        data = pd.read_csv("Salary_Data.csv")
        X = data[['YearsExperience']]
        y = data['Salary']

        plt.figure()
        plt.scatter(X, y, label='Data')
        plt.plot(X, model.predict(X), linewidth=2, label='Model')

        # колдонуучунун чекити
        plt.scatter(exp, prediction, s=100, label='Your value')

        plt.xlabel('Experience')
        plt.ylabel('Salary')
        plt.legend()

        plt.savefig('static/graph.png')
        plt.close()

    return render_template('index.html', salary=salary)

if __name__ == '__main__':
    app.run(debug=True)
