import joblib
import pandas as pd

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
data12 = {
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

data1 = {
    "tcp_srcport": 2412,
    "tcp_dstport": 8000,
    "ip_proto": 6,
    "frame_len": 54,
    "tcp_flags_syn": 0,
    "tcp_flags_reset": 0,
    "tcp_flags_push": 1,
    "tcp_flags_ack": 1,
    "ip_flags_mf": 0,
    "ip_flags_df": 0,
    "ip_flags_rb": 0,
    "tcp_seq": 1,
    "tcp_ack": 1,
    "packets": 8,
    "bytes": 432,
    "tx_packets": 4,
    "tx_bytes": 216,
    "rx_packets": 4,
    "rx_bytes": 216
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame([data1])
print(df)
print(label[ddos_model_1.predict(df)[0]])