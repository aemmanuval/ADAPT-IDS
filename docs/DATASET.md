# ADAPT-IDS — Dataset Documentation

## Primary Dataset: CIC-IDS2017

### Source

Canadian Institute for Cybersecurity / University of New Brunswick
<https://www.unb.ca/cic/datasets/ids-2017.html>

### Description

CIC-IDS2017 contains benign and attack network traffic captured over five
working days (Monday–Friday, July 3–7, 2017). Raw PCAPs were processed
into flow-level features using CICFlowMeter, producing CSV files with
approximately 80 features per flow.

### Files

| File | Day | Attack Types |
|------|-----|-------------|
| Monday-WorkingHours.pcap_ISCX.csv | Mon Jul 3 | Benign only |
| Tuesday-WorkingHours.pcap_ISCX.csv | Tue Jul 4 | Brute Force (FTP-Patator, SSH-Patator) |
| Wednesday-WorkingHours.pcap_ISCX.csv | Wed Jul 5 | DoS (Slowloris, SlowHTTPTest, Hulk, GoldenEye), Heartbleed |
| Thursday-WorkingHours.pcap_ISCX.csv | Thu Jul 6 | Web Attacks (XSS, SQL Injection, Brute Force), Infiltration |
| Friday-WorkingHours.pcap_ISCX.csv | Fri Jul 7 | Botnet, PortScan, DDoS |

### Approximate Size

- ~2.8 million flows total across all files
- CSV files total ~1–2 GB uncompressed

### Known Issues

1. Some CSVs have inconsistent column-name whitespace (leading spaces).
2. Certain features contain infinity and NaN values.
3. Timestamp format may vary between files.
4. Class imbalance: benign flows heavily outnumber individual attack types.
5. Potential labelling errors in the original dataset (known in literature).

### Preprocessing Applied

1. Strip whitespace from column headers.
2. Replace +/- infinity with NaN, then fill NaN with column median.
3. Drop identifier columns (Flow ID, IPs, Ports, Protocol) to prevent
   leakage.
4. Drop constant or near-constant features (>99% single value).
5. Remove exact duplicate rows.
6. Map labels: binary (BENIGN / ATTACK) or multiclass.

See `configs/default.yaml` for full preprocessing configuration.

### Data Leakage Prevention

| Risk | Mitigation |
|------|------------|
| Random shuffle before temporal split | Temporal split preserves chronological order |
| Scaler fitted on full dataset | Scaler fitted only on training portion |
| Feature selection on test data | Feature columns determined from training data only |
| Future labels influencing past | Streaming evaluation processes data sequentially |
| Test data in training | Automated test verifies `max(train_ts) < min(test_ts)` |

### Licensing

The dataset is released by UNB/CIC for research purposes. Cite the
original authors in any publication:

> Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani,
> "Toward Generating a New Intrusion Detection Dataset and Intrusion
> Traffic Characterization", 4th International Conference on Information
> Systems Security and Privacy (ICISSP), 2018.

### Configuration

```yaml
dataset:
  name: cic_ids_2017
  raw_dir: data/raw
  processed_dir: data/processed
  sample_fraction: 1.0
  max_rows: null
```

Reduce for development:
```yaml
  sample_fraction: 0.1
  # or
  max_rows: 100000
```
