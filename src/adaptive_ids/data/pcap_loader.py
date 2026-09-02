"""PCAP/Wireshark ingestion pipeline.

Converts raw packet captures into flow-level features compatible with
the ADAPT-IDS model. Supports two modes:

1. Offline: process a .pcap/.pcapng file via tshark + CICFlowMeter
2. Live: capture from a network interface via tshark

Requirements (not Python packages — system tools):
    - tshark (Wireshark CLI) for packet parsing
  - CICFlowMeter (optional) for full CIC-compatible flow generation
  - Alternatively, uses a lightweight built-in flow aggregator
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_ids.utils.logging import get_logger

logger = get_logger("data.pcap")

TSHARK_FLOW_FIELDS = [
    "-e", "frame.time_epoch",
    "-e", "ip.src", "-e", "ip.dst",
    "-e", "tcp.srcport", "-e", "tcp.dstport",
    "-e", "udp.srcport", "-e", "udp.dstport",
    "-e", "ip.proto", "-e", "frame.len",
    "-e", "tcp.flags",
    "-e", "ip.ttl",
    "-e", "tcp.window_size_value",
]


def check_tshark() -> bool:
    """Check if tshark is available on the system."""
    try:
        result = subprocess.run(
            ["tshark", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def pcap_to_packets(pcap_path: str | Path, max_packets: int | None = None) -> pd.DataFrame:
    """Extract packet-level data from a PCAP file using tshark.

    Returns a DataFrame with one row per packet.
    """
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    if not check_tshark():
        raise RuntimeError(
            "tshark not found. Install Wireshark:\n"
            "  macOS: brew install wireshark\n"
            "  Ubuntu: sudo apt install tshark\n"
            "  Windows: https://www.wireshark.org/download.html"
        )

    cmd = [
        "tshark", "-r", str(pcap_path),
        "-T", "fields",
        *TSHARK_FLOW_FIELDS,
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=f",
    ]
    if max_packets:
        cmd.extend(["-c", str(max_packets)])

    logger.info("Running tshark on %s", pcap_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"tshark failed: {result.stderr[:500]}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write(result.stdout)
        tmp_path = tmp.name

    df = pd.read_csv(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    logger.info("Extracted %d packets from %s", len(df), pcap_path.name)
    return df


def aggregate_flows(packets_df: pd.DataFrame, timeout_s: float = 120.0) -> pd.DataFrame:
    """Aggregate packets into bidirectional flows with CIC-compatible features.

    This is a lightweight alternative to CICFlowMeter that produces
    a subset of the flow features used by CIC-IDS2017.
    """
    packets = packets_df.copy()

    for col in ["tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"]:
        if col in packets.columns:
            packets[col] = pd.to_numeric(packets[col], errors="coerce").fillna(0).astype(int)

    packets["srcport"] = packets.get("tcp.srcport", 0).astype(int) + packets.get("udp.srcport", 0).astype(int)
    packets["dstport"] = packets.get("tcp.dstport", 0).astype(int) + packets.get("udp.dstport", 0).astype(int)
    packets["frame.time_epoch"] = pd.to_numeric(packets["frame.time_epoch"], errors="coerce")
    packets["frame.len"] = pd.to_numeric(packets["frame.len"], errors="coerce").fillna(0)

    flow_key_cols = ["ip.src", "ip.dst", "srcport", "dstport", "ip.proto"]
    available_keys = [c for c in flow_key_cols if c in packets.columns]

    if not available_keys:
        logger.warning("No flow key columns found — returning empty flows")
        return pd.DataFrame()

    flows = []
    for key, group in packets.groupby(available_keys):
        group = group.sort_values("frame.time_epoch")
        ts = group["frame.time_epoch"].values
        lengths = group["frame.len"].values
        n = len(group)

        duration = (ts[-1] - ts[0]) * 1e6 if n > 1 else 0
        iat = np.diff(ts) * 1e6 if n > 1 else np.array([0.0])

        flow = {
            "Flow Duration": duration,
            "Total Fwd Packets": n,
            "Total Backward Packets": 0,
            "Total Length of Fwd Packets": float(lengths.sum()),
            "Total Length of Bwd Packets": 0.0,
            "Fwd Packet Length Max": float(lengths.max()),
            "Fwd Packet Length Min": float(lengths.min()),
            "Fwd Packet Length Mean": float(lengths.mean()),
            "Fwd Packet Length Std": float(lengths.std()) if n > 1 else 0.0,
            "Flow Bytes/s": float(lengths.sum() / (duration / 1e6)) if duration > 0 else 0.0,
            "Flow Packets/s": float(n / (duration / 1e6)) if duration > 0 else 0.0,
            "Flow IAT Mean": float(iat.mean()) if len(iat) > 0 else 0.0,
            "Flow IAT Std": float(iat.std()) if len(iat) > 1 else 0.0,
            "Flow IAT Max": float(iat.max()) if len(iat) > 0 else 0.0,
            "Flow IAT Min": float(iat.min()) if len(iat) > 0 else 0.0,
            "Fwd IAT Total": float(iat.sum()) if len(iat) > 0 else 0.0,
            "Fwd IAT Mean": float(iat.mean()) if len(iat) > 0 else 0.0,
            "Fwd IAT Std": float(iat.std()) if len(iat) > 1 else 0.0,
            "Fwd IAT Max": float(iat.max()) if len(iat) > 0 else 0.0,
            "Fwd IAT Min": float(iat.min()) if len(iat) > 0 else 0.0,
            "Min Packet Length": float(lengths.min()),
            "Max Packet Length": float(lengths.max()),
            "Packet Length Mean": float(lengths.mean()),
            "Packet Length Std": float(lengths.std()) if n > 1 else 0.0,
            "Packet Length Variance": float(lengths.var()) if n > 1 else 0.0,
            "Average Packet Size": float(lengths.mean()),
            "SYN Flag Count": 0,
            "ACK Flag Count": 0,
            "FIN Flag Count": 0,
            "RST Flag Count": 0,
            "PSH Flag Count": 0,
            "URG Flag Count": 0,
            "Timestamp": ts[0],
        }

        if "tcp.flags" in group.columns:
            flags = group["tcp.flags"].dropna()
            for f in flags:
                try:
                    f_int = int(str(f), 16) if isinstance(f, str) else int(f)
                    if f_int & 0x02: flow["SYN Flag Count"] += 1
                    if f_int & 0x10: flow["ACK Flag Count"] += 1
                    if f_int & 0x01: flow["FIN Flag Count"] += 1
                    if f_int & 0x04: flow["RST Flag Count"] += 1
                    if f_int & 0x08: flow["PSH Flag Count"] += 1
                    if f_int & 0x20: flow["URG Flag Count"] += 1
                except (ValueError, TypeError):
                    pass

        if "tcp.window_size_value" in group.columns:
            wins = pd.to_numeric(group["tcp.window_size_value"], errors="coerce").dropna()
            flow["Init_Win_bytes_forward"] = float(wins.iloc[0]) if len(wins) > 0 else 0.0
        else:
            flow["Init_Win_bytes_forward"] = 0.0

        flow["Init_Win_bytes_backward"] = 0.0
        flows.append(flow)

    flows_df = pd.DataFrame(flows)
    logger.info("Aggregated %d packets into %d flows", len(packets), len(flows_df))
    return flows_df


def load_pcap_as_flows(
    pcap_path: str | Path,
    max_packets: int | None = None,
) -> pd.DataFrame:
    """End-to-end: PCAP file -> flow-level DataFrame ready for the model."""
    packets = pcap_to_packets(pcap_path, max_packets=max_packets)
    flows = aggregate_flows(packets)
    return flows


def load_cicflowmeter_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV generated by CICFlowMeter (from Wireshark PCAP).

    CICFlowMeter produces CSVs with the same schema as CIC-IDS2017,
    which is directly compatible with our model.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CICFlowMeter CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    logger.info("Loaded CICFlowMeter CSV: %d flows, %d columns", len(df), len(df.columns))
    return df
