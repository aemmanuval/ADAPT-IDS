# Dataset: CIC-IDS2017

## Source

Canadian Institute for Cybersecurity / University of New Brunswick

**Official page:** <https://www.unb.ca/cic/datasets/ids-2017.html>

## Description

CIC-IDS2017 contains benign and the most common up-to-date attacks at the
time of collection (2017). Traffic was captured over five days (Monday–Friday)
and processed into flow-level CSV files using CICFlowMeter.

## Files expected

Place CSV files in `data/raw/`:

| File | Day | Traffic |
|------|-----|---------|
| Monday-WorkingHours.pcap_ISCX.csv | Mon | Benign only |
| Tuesday-WorkingHours.pcap_ISCX.csv | Tue | Benign + BruteForce (FTP/SSH) |
| Wednesday-WorkingHours.pcap_ISCX.csv | Wed | Benign + DoS/Heartbleed |
| Thursday-WorkingHours.pcap_ISCX.csv | Thu | Benign + Web Attacks + Infiltration |
| Friday-WorkingHours.pcap_ISCX.csv | Fri | Benign + Botnet + PortScan + DDoS |

## Download

1. Visit the official page above
2. Download the "MachineLearningCSV" zip archive
3. Extract CSVs into `data/raw/`

The pipeline will detect and load all CSVs automatically.

## Licensing

The dataset is published for research purposes by UNB/CIC.
Cite the original authors in any publication.

## Size warning

The full dataset is approximately 2.8 million flows across all days.
Use `sample_fraction` or `max_rows` in `configs/default.yaml` to work
with a subset during development.
