import os 
import sys 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) 
sys.path.append(PROJECT_ROOT) 

json_dir =  '/project/hnguyen2/mvu9/datasets/TCGA-labels/clinical.project-tcga-kirc.2025-05-08.json'
import os
import sys
import json
import pandas as pd

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

# Path to the JSON file
json_dir = '/project/hnguyen2/mvu9/datasets/TCGA-labels/clinical.project-tcga-kirc.2025-05-08.json'

# Output CSV path
output_csv = '/project/hnguyen2/mvu9/datasets/TCGA-labels/kirc_diagnosis_labels.csv'

def extract_tcga_diagnoses(json_path, output_csv):
    with open(json_path, "r") as f:
        data = json.load(f)

    records = []
    for case in data:
        case_id = case.get("submitter_id", "Unknown")
        diagnoses = case.get("diagnoses", [])

        # Extract diagnosis info if available
        if diagnoses:
            diag = diagnoses[-1]
            primary_diagnosis = diag.get("primary_diagnosis", "Unknown")
            stage = diag.get("ajcc_pathologic_stage", "Unknown")
            morphology = diag.get("morphology", "Unknown")
            age = diag.get("age_at_diagnosis", None)
            if age is not None:
                age = int(age / 365.25)  # convert days to years
        else:
            primary_diagnosis = "Unknown"
            stage = "Unknown"
            morphology = "Unknown"
            age = "Unknown"

        records.append({
            "case_id": case_id,
            "primary_diagnosis": primary_diagnosis,
            "ajcc_pathologic_stage": stage,
            "morphology": morphology,
            "age_at_diagnosis_years": age
        })

    df = pd.DataFrame(records)
    # After you've built the DataFrame
    unique_diagnoses = df["primary_diagnosis"].value_counts()
    print("[INFO] Unique primary diagnoses and their counts:")
    print(unique_diagnoses)
    
    # df.to_csv(output_csv, index=False)
    # print(f"[INFO] Saved: {output_csv}")

# Run
if __name__ == "__main__":
    extract_tcga_diagnoses(json_dir, output_csv)
    
