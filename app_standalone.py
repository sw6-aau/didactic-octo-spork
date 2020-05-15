from flask import Flask, request, jsonify
from subprocess import Popen, PIPE

import urllib.request
import os
import requests
import pandas as pd 
import numpy as np

app = Flask(__name__)

@app.route('/train', methods=['GET'])
def train():
    build_id = request.args.get("build_id")
    build_data = build_id
    build_model = build_id + '.pt'
    url = ("https://storage.googleapis.com/astep-storage/" + str(build_id))
    urllib.request.urlretrieve(url, "/home/LSTnet-demo-production/" + str(build_id))
    horizon = request.args.get("horizon")
    dropout = request.args.get("dropout")
    skip_rnn = request.args.get("skip_rnn")
    epoch = request.args.get("epoch")
    hid_cnn = request.args.get("hid_cnn")
    hid_rnn = request.args.get("hid_rnn")
    hid_skip_rnn = request.args.get("hid_skip_rnn")# DONT KNOW WHERE IS
    window_rnn = request.args.get("window_rnn")
    window_hw = request.args.get("windows_hw")
    af_output = request.args.get("af_output") # NOT YET IMPLEMENTED; MISSING PARSER ARGS
    af_ae = request.args.get("af_ae") # NOT YET IMPLEMENTED; MISSING PARSER ARGS
 
    os.system('python3.8 main.py ' + '--data ' + build_data + ' --save ' + build_model + ' --horizon ' + horizon +
              ' --dropout ' + dropout + ' --epochs ' + epoch + ' --hidRNN ' + hid_rnn + ' --window ' + window_rnn +
              ' --highway_window ' + window_hw + ' --L1Loss False --output_fun None ' + ' --hidCNN ' + hid_cnn)

    url = ('https://up-service-2aeagw7vrq-ew.a.run.app/upload?build_id=' + str(build_model))
    files = {'file': open(str(build_model), 'rb')}

    requests.post(url, files=files)
    return build_id


@app.route('/predict', methods=['GET'])
def predict():
    build_id = request.args.get("build_id")
    datafile_id = request.args.get("datafile_id")
    build_model = build_id + '.pt'
    build_predict = build_id + '.predict'
    url_ds = ("https://storage.googleapis.com/astep-storage/" + str(build_id))
    urllib.request.urlretrieve(url_ds, "/home/LSTnet-demo-production/" + str(build_id))
    url_pr = ("https://storage.googleapis.com/astep-storage/" + str(datafile_id))
    urllib.request.urlretrieve(url_pr, "/home/LSTnet-demo-production/" + str(datafile_id))

    os.system('python3.8 predict.py' + ' --data ' + datafile_id + ' --save ' + build_model + ' --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None --epochs 1000')

    url = ('https://up-service-2aeagw7vrq-ew.a.run.app/upload?build_id=' + str(build_predict))
    files = {'file': open('output.csv', 'rb')}

    requests.post(url, files=files)

    inp = pd.read_csv(datafile_id)
    out = pd.read_csv("output.csv")

    inp = inp.iloc[179:]

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

    #eturn str(total_rse)
    return jsonify(
        buildid=build_id,
        predictid=datafile_id,
        rse=str(total_rse)
    )

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
