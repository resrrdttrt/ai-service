from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    with open("fuel_random_forest_model_pca.pkl", "rb") as file:
        fuel_model_1 = joblib.load(file)
except Exception as e:
    print("Error loading the model:", str(e))
    fuel_model_1 = None

try:
    with open("fuel_decision_tree_model_pca.pkl", "rb") as file:
        fuel_model_2 = joblib.load(file)
except Exception as e:
    print("Error loading the model:", str(e))
    fuel_model_2 = None

try:
    with open("ddos_knn_classifier_model.pkl", "rb") as file:
        ddos_model_1 = joblib.load(file)
except Exception as e:
    print("Error loading the model:", str(e))
    ddos_model_1 = None



scaler=PCA(n_components=5)
try:
    X_train = pd.read_csv("fuel_x_sample.csv")
    X_train.columns = ['distance', 'speed', 'temp_inside', 'gas_type', 'AC', 'rain', 'sun']
    scaler = PCA(n_components=5)
    scaled_X_train = scaler.fit_transform(X_train)
except Exception as e:
    print("Error loading training data:", str(e))
    scaled_X_train = None

label=["Benign","DDoS-ACK","DDoS-PSH-ACK"]

app = Flask("AI Service")

def fuel_predict(data):
    try:
        if not fuel_model_1 or not fuel_model_2 or scaled_X_train is None:   
            return "Model or training data is not available"
        input_data = np.array(
            [
                [
                    data["distance"],
                    data["speed"],
                    data["temp_inside"],
                    data["gas_type"],
                    data["AC"],
                    data["rain"],
                    data["sun"],
                ]
            ]
        )
        input_data = pd.DataFrame(input_data, columns=['distance', 'speed', 'temp_inside', 'gas_type', 'AC', 'rain', 'sun'])
        # Scale the input data using the same scaler used during training
        scaled_input_data = scaler.transform(input_data)

        # Make predictions using the loaded model
        output1 = fuel_model_1.predict(scaled_input_data)
        output2 = fuel_model_2.predict(scaled_input_data)

        result = [output1[0]/100*float(data["distance"]),output2[0]/100*float(data["distance"])]
        return result
    except Exception as e:
        return "Error making predictions: " + str(e)


# Post endpoint
@app.route("/fuel", methods=["POST"])
def FuelHandler():
    data = request.get_json()
    print(">>>ReceivedData: ", data)

    # Ensure that the required fields are present in the JSON
    required_fields = [
        "distance",
        "speed",
        "temp_inside",
        "gas_type",
        "AC",
        "rain",
        "sun",
    ]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # # Use the classifier to make predictions
    result = fuel_predict(data)
    print(result)
    return jsonify({"RF": f'{result[0]}',"DT":f'{result[1]}'}), 200

@app.route("/ddos", methods=["POST"])
def DDoSHandler():
    data = request.get_json()
    print(">>>ReceivedData: ", data)

    # Ensure that the required fields are present in the JSON
    required_fields = [
        "tcp_srcport",
        "tcp_dstport",
        "ip_proto",
        "frame_len",
        "tcp_flags_syn",
        "tcp_flags_reset",
        "tcp_flags_push",
        "tcp_flags_ack",
        "ip_flags_mf",
        "ip_flags_df",
        "ip_flags_rb",
        "tcp_seq",
        "tcp_ack",
        "packets",
        "bytes",
        "tx_packets",
        "tx_bytes",
        "rx_packets",
        "rx_bytes"
    ]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    df = pd.DataFrame([data])
    # # Use the classifier to make predictions
    result = label[ddos_model_1.predict(df)[0]]
    return jsonify({"result": f'{result}'}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)