from flask import Flask, render_template, url_for, request, jsonify
import helper
import pickle
from tensorflow.keras.models import load_model
model=load_model(r"C:\Users\udays\Desktop\Data Science\projects\quora project\quora\modelnlp.h5")

app = Flask(__name__)




@app.route("/")
def index():
    return render_template('index.html')



@app.route("/result",methods=['POST','GET'])
def result():

    q1 = request.form['question1']
    q2 = request.form['question2']
    print(q1)
    aa = model.predict(helper.query_point_creator(q1, q2))
    print(aa)

    if aa > 0.5:
        r = 'both questions are same'
    else:
        r = 'both questions are not same'

    return render_template('result.html', res = r)


if __name__=="__main__":
    app.run(debug=True, port=5398)