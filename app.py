from flask import Flask, request
from subprocess import Popen, PIPE

app = Flask(__name__)

@app.route('/train')
def train():
    build_id = request.args.get("build_id")
    horizon = request.args.get("horizon")
    dropout = request.args.get("dropout")
    skip_rnn = request.args.get("skip_rnn")
    epoch = request.args.get("epoch")
    ae_hid_cnn = request.args.get("ae_hid_cnn")
    hid_rnn = request.args.get("hid_rnn")
    hid_skip_rnn = request.args.get("hid_skip_rnn")
    window_rnn = request.args.get("window_rnn")
    window_hw = request.args.get("windows_hw")
    af_output = request.args.get("af_output")
    af_ae = request.args.get("af_ae")
    
    process = Popen(['python2 main.py', "--data " + build_id, "--save " + build_id, "--hidCNN " + ae_hid_cnn, "--hidRNN " + hid_rnn, "--L1Loss False", "--Output_fun None"], stdout=PIPE, stderr=PIPE)
    process.wait() # Wait for training to complete.
    
    return build_id


@app.route('/predict')
def predict():
    build_id = request.args.get("build_id")
    
    process = Popen(['python2 predict.py', "--data " + build_id, "--save " + build_id, "--hidCNN " + ae_hid_cnn, "--hidRNN " + hid_rnn, "--L1Loss False", "--Output_fun None"], stdout=PIPE, stderr=PIPE)
    process.wait() # Wait for training to complete.
    
    return build_id