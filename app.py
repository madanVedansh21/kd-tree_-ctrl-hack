from flask import Flask, request, jsonify
import pandas as pd
import os
from kd_tree_working_3 import RobustMultimessengerCorrelator
from flask_cors import CORS
from flask import send_file, after_this_request
import uuid

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load correlator with default data directory (can be changed per request)
def get_correlator(csv_directory="./data"):
    correlator = RobustMultimessengerCorrelator(csv_directory=csv_directory)
    correlator.load_csv_files()
    return correlator

@app.route("/correlate", methods=["POST"])
def correlate():
    try:
        print("CWD:", os.getcwd())
        # Use a unique temp directory per request
        req_id = str(uuid.uuid4())
        data_dir = f"./data_{req_id}"
        os.makedirs(data_dir, exist_ok=True)
        # Remove old files in data dir
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                os.remove(os.path.join(data_dir, f))

        files_received = []
        if 'files' in request.files:
            files = request.files.getlist('files')
            if len(files) < 2:
                return jsonify({"error": "At least 2 CSV files are required."}), 400
            for f in files:
                path = os.path.join(data_dir, f.filename)
                f.save(path)
                files_received.append(path)
        elif request.is_json:
            data = request.get_json()
            if not isinstance(data, dict) or len(data) < 2:
                return jsonify({"error": "At least 2 CSV files are required in JSON."}), 400
            for fname, csv_str in data.items():
                path = os.path.join(data_dir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(csv_str)
                files_received.append(path)
        else:
            return jsonify({"error": "No valid input provided. Send at least 2 files or JSON."}), 400

        correlator = RobustMultimessengerCorrelator(csv_directory=data_dir)
        correlator.load_csv_files()
        # Save all correlations to CSV
        results = correlator.find_correlations(target_top_n=1000000, output_file="multimessenger_correlations.csv")
        if results is None or results.empty:
            return jsonify({"results": []})

        # Read performance metrics from hackathon_technical_report.txt (lines 9-12)
        perf_metrics = {}
        try:
            with open("hackathon_technical_report.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().startswith("Total Events Processed"):
                        perf_metrics["Total Events Processed"] = int(line.split(":")[-1].strip())
                    elif line.strip().startswith("Datasets Successfully Loaded"):
                        perf_metrics["Datasets Successfully Loaded"] = int(line.split(":")[-1].strip())
                    elif line.strip().startswith("Valid Correlations Found"):
                        perf_metrics["Valid Correlations Found"] = int(line.split(":")[-1].strip())
        except Exception as e:
            perf_metrics = {"error": f"Could not read performance metrics: {str(e)}"}

        # Prepare response: send CSV file and JSON metrics
        @after_this_request
        def cleanup(response):
            # Optionally clean up uploaded files if needed
            return response

        return send_file(
            "multimessenger_correlations.csv",
            mimetype="text/csv",
            as_attachment=True,
            download_name="multimessenger_correlations.csv",
            etag=False
        ), 200, {"X-Performance-Metrics": jsonify(perf_metrics).get_data(as_text=True)}
    except Exception as e:
        # Log the error only once
        print(f"Error in /correlate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
