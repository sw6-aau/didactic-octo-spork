from flask import Flask, request, jsonify
from subprocess import Popen, PIPE

import urllib.request
import os
import requests
import pandas as pd 
import numpy as np

import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
import configparser

app = Flask(__name__)

@app.route('/train', methods=['GET'])
def train():
    reader = configparser.RawConfigParser()
    reader.read('information.cnf')
    build_id = request.args.get("build_id")
    build_data = build_id
    build_model = build_id + '.pt'
    url = ("https://storage.googleapis.com/astep-storage/" + str(build_id))
    urllib.request.urlretrieve(url, "/home/LSTnet-demo-production/" + str(build_id))

    connection = mysql.connector.connect(host=reader.get('client', 'host'),
                                        database=reader.get('client', 'db_name'),
                                        user=reader.get('client', 'db_username'),
                                        password=reader.get('client', 'db_password'))
    sql = "SELECT * FROM build_params WHERE build_id=%s"
    val = (str(build_id),)
    cursor = connection.cursor()
    cursor.execute(sql, val)
    result = cursor.fetchone()
    connection.commit()
 
    os.system('python3.8 main.py ' + '--data ' + build_data + ' --save ' + build_model + ' --horizon ' + str(result[1]) +
              ' --dropout ' + str(result[2]) + ' --epochs ' + str(result[4]) + ' --hidRNN ' + str(result[6]) + ' --window ' + str(result[7]) +
              ' --highway_window ' + str(result[8]) + ' --L1Loss False --output_fun None ' + ' --hidCNN ' + str(result[5]))

    url = ('https://up-service-2aeagw7vrq-ew.a.run.app/upload?build_id=' + str(build_model))
    files = {'file': open(str(build_model), 'rb')}

    requests.post(url, files=files)
    return build_id


@app.route('/predict', methods=['GET'])
def predict():
    reader = configparser.RawConfigParser()
    reader.read('information.cnf')

    build_id = request.args.get("build_id")
    datafile_id = request.args.get("datafile_id")
    build_model = build_id + '.pt'
    build_predict = datafile_id + '.predict'
    url_ds = ("https://storage.googleapis.com/astep-storage/" + str(build_id))
    urllib.request.urlretrieve(url_ds, "/home/LSTnet-demo-production/" + str(build_id))
    url_pr = ("https://storage.googleapis.com/astep-storage/" + str(datafile_id))
    urllib.request.urlretrieve(url_pr, "/home/LSTnet-demo-production/" + str(datafile_id))

    connection = mysql.connector.connect(host=reader.get('client', 'host'),
                                    database=reader.get('client', 'db_name'),
                                    user=reader.get('client', 'db_username'),
                                    password=reader.get('client', 'db_password'))
    sql = "SELECT * FROM build_params WHERE build_id=%s"
    val = (str(build_id),)
    cursor = connection.cursor()
    cursor.execute(sql, val)
    result = cursor.fetchone()
    connection.commit()

    os.system('python3.8 predict.py ' + '--data ' + datafile_id + ' --save ' + build_model + ' --horizon ' + str(result[1]) +
            ' --dropout ' + str(result[2]) + ' --epochs ' + str(result[4]) + ' --hidRNN ' + str(result[6]) + ' --window ' + str(result[7]) +
            ' --highway_window ' + str(result[8]) + ' --L1Loss False --output_fun None ' + ' --hidCNN ' + str(result[5]))

    url = ('https://up-service-2aeagw7vrq-ew.a.run.app/upload?build_id=' + str(build_predict))
    files = {'file': open('output.csv', 'rb')}

    requests.post(url, files=files)

    inp = pd.read_csv(datafile_id)
    out = pd.read_csv("output.csv")

    inp = inp.iloc[178:]

    num_cols = len(list(inp))
    rng = range(1, num_cols+1)
    new_cols = ['col' + str(i) for i in rng]

    inp.columns = new_cols[:num_cols]
    out.columns = new_cols[:num_cols] 

    rse = []
    rse2 = []

    for column in inp:
        rse.append(np.power((inp[column].values-out[column].values), 2))

    top_value = np.sqrt(np.sum((rse)))
    all_mean = np.mean(inp.mean())

    for column in inp:
        rse2.append(np.power((inp[column].values-all_mean), 2))

    bot_value = np.sqrt(np.sum((rse2)))
    total_rse = (top_value/bot_value)
    
    return jsonify(
	rse=str(total_rse),
	predictid=datafile_id,
	buildid=build_id)
    #return str(total_rse)

@app.route('/test', methods=['GET'])
def test():
    return "works"

@app.route('/default', methods=['GET'])
def default():
    build_id = "test123"
    os.system('python2 main.py --data exchange_rate.txt --save exchange_rate.pt --hidCNN 50 --hidRNN 50 --epochs 10 --L1Loss False --output_fun None')
    #upload_blob("astep-storage", "exchange_rate.pt", build_id)
    return "done"

if __name__ == '__main__':
    app.run(threaded=True, host='34.105.252.163', port=80)
