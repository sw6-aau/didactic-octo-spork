from flask import Flask, request
from subprocess import Popen, PIPE
import os

app = Flask(__name__)

@app.route('/train', methods=['GET'])
def train():
    build_id = request.args.get("build_id")
    build_data = request.args.get("build_data")
    build_model = request.args.get("build_model")
    
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
    
    process = Popen(['python2 main.py', "--data " + build_data, "--save " + build_model, "--hidCNN " + ae_hid_cnn, "--hidRNN " + hid_rnn, "--L1Loss False", "--Output_fun None"], stdout=PIPE, stderr=PIPE)
    process.wait() # Wait for training to complete.
    
    return build_id


@app.route('/predict', methods=['GET'])
def predict():
    build_id = request.args.get("build_id")
    
    #os.system('python2 predict.py', "--data " + build_id, "--save " + build_id, "--L1Loss False", "--Output_fun None")
   # process.wait() # Wait for training to complete.
    
    return build_id

@app.route('/test', methods=['GET'])
def test():
    return "works"

@app.route('/default', methods=['GET'])
def default():
    os.system('python2 main.py --data exchange_rate.txt --save exchange_rate.pt --hidCNN 50 --hidRNN 50 --epochs 10 --L1Loss False --output_fun None')
    return "done"

port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port)