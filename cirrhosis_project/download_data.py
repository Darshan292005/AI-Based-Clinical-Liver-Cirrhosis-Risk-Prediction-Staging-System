import os
import pandas as pd


def download_dataset():
    print("Fetching Mayo Clinic PBC dataset from a known reliable mirror...")
    try:
        url = "https://raw.githubusercontent.com/KaziRamisaRifa/Liver-Cirrhosis-Detection/main/cirrhosis.csv"
        df = pd.read_csv(url)
        print("Columns found:", df.columns.tolist())

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/raw_cirrhosis.csv", index=False)
        print("Dataset successfully saved to data/raw_cirrhosis.csv")
    except Exception as e:
        print(f"Error downloading dataset: {e}")


if __name__ == "__main__":
    download_dataset()

