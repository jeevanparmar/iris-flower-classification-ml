from flask import Flask, render_template,  request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
# model = pickle.load(open('iris_model.pkl', 'rb'))
with open("model/iris_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    flower_map = {
        0: ("Setosa", "setosa.png"),
        1: ("Versicolor", "versicolor.jpg"),
        2: ("Virginica", "virginica.jpg")
    }

    flower_name, flower_image = flower_map[prediction]
    # print(flower_name)

    return jsonify({
        "flower": flower_name,
        "image": flower_image
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



    