import os
import json
import ast
import pandas as pd

LOCAL_FILE = r"d:\RL\dataset\borg_traces_data.csv"

def calibrate():
    print(f"Reading and parsing data from {LOCAL_FILE}...")
    
    # Read a subset of rows to save time and memory. The 2019 trace has an 'average_usage' column.
    df = pd.read_csv(
        LOCAL_FILE, 
        usecols=["average_usage"],
        nrows=100000 
    )
    
    print("Computing statistics...")
    df = df.dropna()
    
    cpu_usages = []
    ram_usages = []
    
    for valStr in df["average_usage"]:
        try:
            # valStr is like "{'cpus': 0.00466156005859375, 'memory': 0.00592041015625}"
            val = ast.literal_eval(valStr)
            if 'cpus' in val and val['cpus'] is not None:
                cpu_usages.append(float(val['cpus']))
            if 'memory' in val and val['memory'] is not None:
                ram_usages.append(float(val['memory']))
        except Exception:
            continue
            
    cpu_series = pd.Series(cpu_usages)
    ram_series = pd.Series(ram_usages)
    
    stats = {
        "cpu": {
            "mean": float(cpu_series.mean()),
            "std": float(cpu_series.std()),
            "min": float(cpu_series.min()),
            "max": float(cpu_series.max()),
            "p95": float(cpu_series.quantile(0.95))
        },
        "ram": {
            "mean": float(ram_series.mean()),
            "std": float(ram_series.std()),
            "min": float(ram_series.min()),
            "max": float(ram_series.max()),
            "p95": float(ram_series.quantile(0.95))
        },
        "sample_size": len(cpu_usages),
        "source": LOCAL_FILE,
        "note": "Extracted from Google Cluster Workload Traces 2019 'average_usage' column."
    }
    
    # For bursty traffic simulation logic
    stats["arrival_rate"] = {
        "lambda_steady": 10.0,
        "lambda_bursty": 50.0,
        "spike_prob": 0.05
    }
    
    summary_path = os.path.join(os.path.dirname(__file__), "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Calibration complete. Results saved to {summary_path}")
    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    calibrate()
