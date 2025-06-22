import os
import wfdb
import pandas as pd


def load_raw_wfdb_data(base_path="../data/WFDBRecords"):
    # Reads all ECG signals from the WFDB dataset in base_path.
    # Returns a DataFrame with signals per record.

    data = []

    with open(os.path.join(base_path, "RECORDS.txt")) as f:
        record_paths = [line.strip() for line in f if line.strip()]

    for dir_raiz in os.listdir(base_path):
        print(f"Loading: {dir_raiz} ...")
        raiz_path = os.path.join(base_path, dir_raiz)

        if not os.path.isdir(raiz_path):
            continue

        for subpasta in os.listdir(raiz_path):
            sub_path = os.path.join(raiz_path, subpasta)
            records_file = os.path.join(sub_path, "RECORDS")

            if not os.path.exists(records_file):
                continue

            with open(records_file) as f:
                registos = [linha.strip() for linha in f if linha.strip()]

            for nome_registo in registos:
                full_path = os.path.join(sub_path, nome_registo)

                try:
                    record = wfdb.rdrecord(full_path)
                    sinal = record.p_signal[:, 0]  # canal 0
                    data.append({
                        "record_id": nome_registo,
                        "signal": sinal.tolist()
                    })
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")

    df_ecg = pd.DataFrame(data)
    print(df_ecg.head())
    print(f"**** Total records loaded: {len(df_ecg)} ****")
    return df_ecg


def enrich_metadata(df_ecg, base_path="../data/WFDBRecords"):
    # Add columns age, sex, and diagnoses to the DataFrame from the .hea headers.
    # 1. Read the mapping from SNOMED codes to acronyms
    df_map = pd.read_csv(os.path.join(base_path, "ConditionNames_SNOMED-CT.csv"))
    snomed_to_acronym = dict(zip(df_map["Snomed_CT"].astype(str), df_map["Acronym Name"]))

    # 2. Creat empty lists
    age, sex, diagnostics_acronyms = [], [], []

    # 3. Create a map record_id → path .hea
    hea_paths = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".hea"):
                record_id = file.replace(".hea", "")
                hea_paths[record_id] = os.path.join(root, record_id)

    # 4. Extract metadata from .hea files
    for idx, row in df_ecg.iterrows():
        record_id = row["record_id"]
        path = hea_paths.get(record_id)

        if path:
            try:
                header = wfdb.rdheader(path)
                com = {c.split(":")[0].strip(): c.split(":")[1].strip() for c in header.comments if ":" in c}

                age.append(int(com.get("Age", -1)))
                sex.append(com.get("Sex", None))

                dx_codes = com.get("Dx", "").split(",") if "Dx" in com else []
                acronyms = [snomed_to_acronym.get(code.strip(), f"UNKNOWN_{code.strip()}") for code in dx_codes]
                diagnostics_acronyms.append(acronyms)

            except Exception as e:
                print(f"Erro ao ler header de {record_id}: {e}")
                print(f"Error reading header from {record_id}: {e}")
                age.append(None)
                sex.append(None)
                diagnostics_acronyms.append([])
        else:
            age.append(None)
            sex.append(None)
            diagnostics_acronyms.append([])

    # 5. Add columns to the DataFrame
    df_ecg["age"] = age
    df_ecg["sex"] = sex
    df_ecg["diagnosticos"] = diagnostics_acronyms

    # 6. Convert sex to binary and drop original column
    df_ecg["male"] = df_ecg["sex"].apply(lambda x: 1 if x == "Male" else 0)
    df_ecg = df_ecg.drop(columns=["sex"])

    print("**** Metadata added to DataFrame ****")

    return df_ecg
