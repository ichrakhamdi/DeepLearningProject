import os
import glob
import pandas as pd

class CICIDataLoader:
    def __init__(self, input_base_path='data/raw', output_base_path='data/concatenated'):
        self.input_base_path = input_base_path
        self.output_base_path = output_base_path

        os.makedirs(self.output_base_path, exist_ok=True)

        self.label_19_to_6 = {
            "ARP_spoofing": "SPOOFING",
            "Ping_sweep": "RECON",
            "Recon_VulScan": "RECON",
            "OS_scan": "RECON",
            "Port_scan": "RECON",
            "Malformed_data": "MQTT",
            "DoS_connect_flood": "MQTT",
            "DDoS_publish_flood": "MQTT",
            "DoS_publish_flood": "MQTT",
            "DDoS_connect_flood": "MQTT",
            "DoS_TCP": "DoS",
            "DoS_ICMP": "DoS",
            "DoS_SYN": "DoS",
            "DoS_UDP": "DoS",
            "DDoS_SYN": "DDoS",
            "DDoS_TCP": "DDoS",
            "DDoS_ICMP": "DDoS",
            "DDoS_UDP": "DDoS",
            "BENIGN": "BENIGN"
        }

    def _extract_labels(self, filename):
        label_19 = "UNKNOWN"

        if "Benign" in filename:
            label_19 = "BENIGN"
        elif "ARP_Spoofing" in filename:
            label_19 = "ARP_spoofing"
        elif "Ping_Sweep" in filename:
            label_19 = "Ping_sweep"
        elif "Port_Scan" in filename:
            label_19 = "Port_scan"
        elif "OS_Scan" in filename:
            label_19 = "OS_scan"
        elif "VulScan" in filename:
            label_19 = "Recon_VulScan"
        elif "Malformed_Data" in filename:
            label_19 = "Malformed_data"
        elif "DDoS-Connect_Flood" in filename:
            label_19 = "DDoS_connect_flood"
        elif "DDoS-Publish_Flood" in filename:
            label_19 = "DDoS_publish_flood"
        elif "DDoS-SYN" in filename:
            label_19 = "DDoS_SYN"
        elif "DDoS-TCP" in filename:
            label_19 = "DDoS_TCP"
        elif "DDoS-ICMP" in filename:
            label_19 = "DDoS_ICMP"
        elif "DDoS-UDP" in filename:
            label_19 = "DDoS_UDP"
        elif "DoS-Connect_Flood" in filename:
            label_19 = "DoS_connect_flood"
        elif "DoS-Publish_Flood" in filename:
            label_19 = "DoS_publish_flood"
        elif "DoS-SYN" in filename:
            label_19 = "DoS_SYN"
        elif "DoS-TCP" in filename:
            label_19 = "DoS_TCP"
        elif "DoS-ICMP" in filename:
            label_19 = "DoS_ICMP"
        elif "DoS-UDP" in filename:
            label_19 = "DoS_UDP"

        label_6 = self.label_19_to_6.get(label_19, "UNKNOWN")
        label_2 = "BENIGN" if label_6 == "BENIGN" else "ATTACK"

        return label_2, label_6, label_19

    def load_and_concat(self, split):
        """Load and label data from a split (train/test)"""
        data = []
        input_folder = os.path.join(self.input_base_path, split)
        for file in glob.glob(f"{input_folder}/*.csv"):
            try:
                df = pd.read_csv(file)
                label_2, label_6, label_19 = self._extract_labels(file)
                df["label_2"] = label_2
                df["label_6"] = label_6
                df["label_19"] = label_19
                data.append(df)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        return pd.concat(data, ignore_index=True)

    def save_concatenated(self, df, split):
        output_path = os.path.join(self.output_base_path, f"{split}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved concatenated {split} data to: {output_path}")

    def run_full_preparation(self):
        """Load, label, and save both train and test splits"""
        for split in ["train", "test"]:
            print(f"\nProcessing {split} split...")
            df = self.load_and_concat(split)
            self.save_concatenated(df, split)
