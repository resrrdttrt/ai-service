from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    with open("ddos_knn_classifier_model.pkl", "rb") as file:
        ddos_model_1 = joblib.load(file)
except Exception as e:
    print("Error loading the model:", str(e))
    ddos_model_1 = None


label=["Benign","DDoS-ACK","DDoS-PSH-ACK"]
data2={
        "tcp.srcport": 2413,
        "tcp.dstport": 8000,
        "ip.proto": 6,
        "frame.len": 54,
        "tcp.flags.syn": 0,
        "tcp.flags.reset": 0,
        "tcp.flags.push": 1,
        "tcp.flags.ack": 1,
        "ip.flags.mf": 0,
        "ip.flags.df": 0,
        "ip.flags.rb": 0,
        "tcp.seq": 1,
        "tcp.ack": 1,
        "Packets": 10,
        "Bytes": 540,
        "Tx Packets": 5,
        "Tx Bytes": 270,
        "Rx Packets": 5,
        "Rx Bytes": 270
    }
data1 = {
    "tcp.srcport": 2412,
    "tcp.dstport": 8000,
    "ip.proto": 6,
    "frame.len": 54,
    "tcp.flags.syn": 0,
    "tcp.flags.reset": 0,
    "tcp.flags.push": 1,
    "tcp.flags.ack": 1,
    "ip.flags.mf": 0,
    "ip.flags.df": 0,
    "ip.flags.rb": 0,
    "tcp.seq": 1,
    "tcp.ack": 1,
    "Packets": 8,
    "Bytes": 432,
    "Tx Packets": 4,
    "Tx Bytes": 216,
    "Rx Packets": 4,
    "Rx Bytes": 216
}


# Convert the dictionary to a DataFrame
df = pd.DataFrame([data1])
print(label[ddos_model_1.predict(df)[0]])