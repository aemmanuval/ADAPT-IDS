#!/usr/bin/env python3
"""ADAPT-IDS Live Network Monitor

A lightweight network packet sniffer + IDS that:
  1. Captures live packets from your network interface (like Wireshark)
  2. Aggregates packets into network flows in real-time
  3. Classifies each flow as BENIGN or ATTACK using the trained model
  4. Logs attacks to MongoDB for persistence
  5. Monitors for concept drift on prediction confidence

Requirements:
  - Root/sudo access for packet capture
  - MongoDB running locally (optional — falls back to console logging)
  - Trained model in results/

Usage:
    sudo python scripts/live_monitor.py                     # auto-detect interface
    sudo python scripts/live_monitor.py --interface en0      # specific interface
    sudo python scripts/live_monitor.py --interface eth0 --duration 300  # 5 min capture
    sudo python scripts/live_monitor.py --mongo mongodb://localhost:27017
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.utils.logging import setup_logging, get_logger

try:
    from scapy.all import sniff, IP, TCP, UDP, conf
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False


FLOW_TIMEOUT = 30.0
FLOW_EXPORT_INTERVAL = 5.0


class LiveFlowAggregator:
    """Aggregates raw packets into bidirectional flows in real-time."""

    def __init__(self):
        self.flows: dict[str, dict] = {}
        self.completed_flows: list[dict] = []

    def _flow_key(self, pkt) -> str | None:
        if not pkt.haslayer(IP):
            return None
        ip = pkt[IP]
        src, dst = ip.src, ip.dst
        sport, dport, proto = 0, 0, ip.proto

        if pkt.haslayer(TCP):
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        if (src, sport) > (dst, dport):
            return f"{src}:{sport}-{dst}:{dport}-{proto}"
        return f"{dst}:{dport}-{src}:{sport}-{proto}"

    def process_packet(self, pkt) -> None:
        key = self._flow_key(pkt)
        if key is None:
            return

        ts = float(pkt.time)
        length = len(pkt)
        ip = pkt[IP]

        if key not in self.flows:
            self.flows[key] = {
                "key": key,
                "src_ip": ip.src,
                "dst_ip": ip.dst,
                "src_port": 0,
                "dst_port": 0,
                "protocol": ip.proto,
                "start_time": ts,
                "last_time": ts,
                "packets": [],
                "fwd_packets": 0,
                "bwd_packets": 0,
                "fwd_bytes": 0,
                "bwd_bytes": 0,
                "syn_count": 0,
                "ack_count": 0,
                "fin_count": 0,
                "rst_count": 0,
                "psh_count": 0,
            }
            if pkt.haslayer(TCP):
                self.flows[key]["src_port"] = pkt[TCP].sport
                self.flows[key]["dst_port"] = pkt[TCP].dport
            elif pkt.haslayer(UDP):
                self.flows[key]["src_port"] = pkt[UDP].sport
                self.flows[key]["dst_port"] = pkt[UDP].dport

        flow = self.flows[key]
        flow["last_time"] = ts
        flow["packets"].append({"time": ts, "length": length})

        if ip.src == flow["src_ip"]:
            flow["fwd_packets"] += 1
            flow["fwd_bytes"] += length
        else:
            flow["bwd_packets"] += 1
            flow["bwd_bytes"] += length

        if pkt.haslayer(TCP):
            flags = pkt[TCP].flags
            if flags & 0x02: flow["syn_count"] += 1
            if flags & 0x10: flow["ack_count"] += 1
            if flags & 0x01: flow["fin_count"] += 1
            if flags & 0x04: flow["rst_count"] += 1
            if flags & 0x08: flow["psh_count"] += 1

    def export_completed(self) -> list[dict]:
        """Export flows that have timed out or completed."""
        now = time.time()
        completed = []
        expired_keys = []

        for key, flow in self.flows.items():
            if now - flow["last_time"] > FLOW_TIMEOUT:
                features = self._compute_features(flow)
                completed.append(features)
                expired_keys.append(key)

        for key in expired_keys:
            del self.flows[key]

        return completed

    def _compute_features(self, flow: dict) -> dict:
        """Convert raw flow into CIC-IDS2017-compatible features."""
        packets = flow["packets"]
        n = len(packets)
        times = [p["time"] for p in packets]
        lengths = [p["length"] for p in packets]

        duration = (times[-1] - times[0]) * 1e6 if n > 1 else 0
        iat = np.diff(times) * 1e6 if n > 1 else np.array([0.0])
        lengths_arr = np.array(lengths, dtype=float)

        return {
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "protocol": flow["protocol"],
            "Flow Duration": duration,
            "Total Fwd Packets": flow["fwd_packets"],
            "Total Backward Packets": flow["bwd_packets"],
            "Total Length of Fwd Packets": float(flow["fwd_bytes"]),
            "Total Length of Bwd Packets": float(flow["bwd_bytes"]),
            "Fwd Packet Length Max": float(lengths_arr.max()) if n > 0 else 0,
            "Fwd Packet Length Min": float(lengths_arr.min()) if n > 0 else 0,
            "Fwd Packet Length Mean": float(lengths_arr.mean()) if n > 0 else 0,
            "Fwd Packet Length Std": float(lengths_arr.std()) if n > 1 else 0,
            "Flow Bytes/s": float(sum(lengths) / (duration / 1e6)) if duration > 0 else 0,
            "Flow Packets/s": float(n / (duration / 1e6)) if duration > 0 else 0,
            "Flow IAT Mean": float(iat.mean()) if len(iat) > 0 else 0,
            "Flow IAT Std": float(iat.std()) if len(iat) > 1 else 0,
            "Flow IAT Max": float(iat.max()) if len(iat) > 0 else 0,
            "Flow IAT Min": float(iat.min()) if len(iat) > 0 else 0,
            "Fwd IAT Total": float(iat.sum()),
            "Fwd IAT Mean": float(iat.mean()) if len(iat) > 0 else 0,
            "Fwd IAT Std": float(iat.std()) if len(iat) > 1 else 0,
            "Fwd IAT Max": float(iat.max()) if len(iat) > 0 else 0,
            "Fwd IAT Min": float(iat.min()) if len(iat) > 0 else 0,
            "Min Packet Length": float(lengths_arr.min()) if n > 0 else 0,
            "Max Packet Length": float(lengths_arr.max()) if n > 0 else 0,
            "Packet Length Mean": float(lengths_arr.mean()) if n > 0 else 0,
            "Packet Length Std": float(lengths_arr.std()) if n > 1 else 0,
            "Packet Length Variance": float(lengths_arr.var()) if n > 1 else 0,
            "Average Packet Size": float(lengths_arr.mean()) if n > 0 else 0,
            "SYN Flag Count": flow["syn_count"],
            "ACK Flag Count": flow["ack_count"],
            "FIN Flag Count": flow["fin_count"],
            "RST Flag Count": flow["rst_count"],
            "PSH Flag Count": flow["psh_count"],
            "Destination Port": flow["dst_port"],
            "Source Port": flow["src_port"],
            "Protocol": flow["protocol"],
            "n_packets": n,
            "start_time": flow["start_time"],
        }


def parse_args():
    parser = argparse.ArgumentParser(description="ADAPT-IDS Live Network Monitor")
    parser.add_argument("--interface", "-i", type=str, default=None, help="Network interface (e.g., en0, eth0, wlan0)")
    parser.add_argument("--duration", "-d", type=int, default=0, help="Capture duration in seconds (0=indefinite)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (.joblib)")
    parser.add_argument("--mongo", type=str, default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument("--no-mongo", action="store_true", help="Disable MongoDB logging")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging("INFO")
    logger = get_logger("live_monitor")

    if not HAS_SCAPY:
        print("ERROR: scapy not installed. Run: pip install scapy")
        sys.exit(1)

    if os.geteuid() != 0 and sys.platform != "win32":
        print("WARNING: Packet capture usually requires root/sudo.")
        print("Run with: sudo python scripts/live_monitor.py")

    root = Path(__file__).resolve().parents[1]

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  —  Live Network Monitor")
    print("  Real-time packet capture → flow classification → attack logging")
    print("=" * 60 + "\n")

    # Load model
    from adaptive_ids.models.baseline import BaselineIDS
    model_path = args.model
    if not model_path:
        for candidate in [
            root / "results" / "cross_dataset" / "lightgbm_combined.joblib",
            root / "results" / "temporal" / "lightgbm" / "lightgbm_temporal.joblib",
            root / "results" / "baseline" / "lightgbm" / "lightgbm_baseline.joblib",
        ]:
            if candidate.exists():
                model_path = str(candidate)
                break

    if not model_path or not Path(model_path).exists():
        print("ERROR: No trained model found. Run training scripts first.")
        sys.exit(1)

    model = BaselineIDS.load(model_path)
    print(f"[+] Model loaded: {Path(model_path).name}")

    # MongoDB
    storage = None
    if not args.no_mongo:
        try:
            from adaptive_ids.storage.mongo import MongoStorage
            storage = MongoStorage(uri=args.mongo)
            if storage.connect():
                print(f"[+] MongoDB connected: {args.mongo}")
            else:
                print("[!] MongoDB not available — logging to console only")
                storage = None
        except Exception as e:
            print(f"[!] MongoDB error: {e} — logging to console only")
            storage = None

    # Drift detector (unsupervised — no labels needed)
    from adaptive_ids.drift.detectors import UnsupervisedDriftDetector
    detector = UnsupervisedDriftDetector(delta=0.002)

    # Flow aggregator
    aggregator = LiveFlowAggregator()

    # Stats
    stats = {"packets": 0, "flows": 0, "attacks": 0, "benign": 0, "drift_events": 0}

    interface = args.interface or conf.iface
    print(f"[+] Capturing on interface: {interface}")
    print(f"[+] Flow timeout: {FLOW_TIMEOUT}s")
    print(f"[+] Press Ctrl+C to stop\n")

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n[!] Stopping capture...")

    signal.signal(signal.SIGINT, signal_handler)

    def process_packet(pkt):
        nonlocal stats
        stats["packets"] += 1
        aggregator.process_packet(pkt)

        if stats["packets"] % 100 == 0:
            completed = aggregator.export_completed()
            for flow_features in completed:
                classify_flow(flow_features, model, detector, storage, stats)

    def classify_flow(flow, model, detector, storage, stats):
        feature_names = [
            "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
            "Fwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
            "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Min Packet Length", "Max Packet Length", "Packet Length Mean",
            "Packet Length Std", "Packet Length Variance", "Average Packet Size",
            "SYN Flag Count", "ACK Flag Count", "FIN Flag Count",
            "RST Flag Count", "PSH Flag Count",
            "Destination Port", "Source Port", "Protocol",
        ]

        X = np.array([[flow.get(f, 0) for f in feature_names]], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)
            confidence = float(proba.max())
        except Exception:
            prediction = "UNKNOWN"
            confidence = 0.0

        detector.update(confidence)
        drift = detector.drift_detected()
        if drift:
            stats["drift_events"] += 1
            if storage:
                storage.log_drift_event(detector.name, stats["flows"], detector.get_state())

        stats["flows"] += 1
        ts = datetime.now().strftime("%H:%M:%S")
        src = f"{flow['src_ip']}:{flow['src_port']}"
        dst = f"{flow['dst_ip']}:{flow['dst_port']}"

        if prediction == "ATTACK":
            stats["attacks"] += 1
            severity = "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
            print(f"  [{ts}] \U0001f6a8 ATTACK  {src} → {dst}  (conf={confidence:.2f}, {flow['n_packets']} pkts, severity={severity})")

            if storage:
                storage.log_attack(
                    confidence=confidence,
                    source_ip=flow["src_ip"],
                    dest_ip=flow["dst_ip"],
                    dest_port=flow["dst_port"],
                    protocol=str(flow["protocol"]),
                    severity=severity,
                    features={f: flow.get(f, 0) for f in feature_names[:10]},
                )
        else:
            stats["benign"] += 1
            if stats["flows"] % 20 == 0:
                print(f"  [{ts}]    OK     {src} → {dst}  (conf={confidence:.2f}, {flow['n_packets']} pkts)")

        if storage:
            storage.log_prediction(
                prediction=prediction,
                confidence=confidence,
                source_ip=flow["src_ip"],
                dest_ip=flow["dst_ip"],
                dest_port=flow["dst_port"],
                protocol=str(flow["protocol"]),
            )

        if drift:
            print(f"  [{ts}] \u26a0\ufe0f  DRIFT DETECTED — model confidence distribution changed")

    try:
        start = time.time()
        while running:
            timeout = min(FLOW_EXPORT_INTERVAL, args.duration - (time.time() - start)) if args.duration > 0 else FLOW_EXPORT_INTERVAL
            if timeout <= 0:
                break

            sniff(
                iface=interface,
                prn=process_packet,
                timeout=timeout,
                store=False,
            )

            completed = aggregator.export_completed()
            for flow_features in completed:
                classify_flow(flow_features, model, detector, storage, stats)

    except PermissionError:
        print("\nERROR: Permission denied. Run with sudo:")
        print(f"  sudo {sys.executable} scripts/live_monitor.py")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 60)
    print("  CAPTURE SUMMARY")
    print("=" * 60)
    print(f"  Packets captured:  {stats['packets']:,}")
    print(f"  Flows classified:  {stats['flows']:,}")
    print(f"  Attacks detected:  {stats['attacks']:,}")
    print(f"  Benign flows:      {stats['benign']:,}")
    print(f"  Drift events:      {stats['drift_events']}")
    if storage and storage.connected:
        db_stats = storage.get_attack_stats()
        print(f"\n  MongoDB totals:")
        print(f"    Total attacks:     {db_stats['total_attacks']}")
        print(f"    Total predictions: {db_stats['total_predictions']}")
        storage.close()
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
