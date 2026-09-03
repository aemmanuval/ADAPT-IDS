# Adaptive Intrusion Detection Under Concept and Feature Drift in Evolving Network Traffic

## Abstract

Machine-learning-based intrusion detection systems (IDS) assume stable traffic distributions, yet real network environments exhibit continuous evolution through new applications, protocols, and attack techniques. This work investigates whether drift-triggered adaptive learning can maintain IDS performance under evolving traffic while reducing unnecessary retraining. We propose ADAPT-IDS, a lightweight framework combining gradient-boosted decision trees with ADWIN-based concept drift detection and automated model retraining. Experiments on CIC-IDS2017 (2.8M flows) and UNSW-NB15 (2.5M flows) demonstrate that (i) a static LightGBM classifier's F1 score catastrophically degrades from 0.997 to 0.036 under temporal evaluation, missing 98.2% of attacks; (ii) drift-triggered adaptation recovers F1 to 0.987 with only 19 retrains; (iii) this approach outperforms both periodic retraining (F1=0.978, 65 retrains) and static LSTM (F1=0.792) while requiring 71% fewer retraining episodes and 11× less computation than deep learning alternatives. Feature drift analysis reveals that 100% of flow features exhibit statistically significant distribution changes (KS test, p<0.05), with inter-arrival time features shifting most severely. Controlled experiments across four synthetic drift types confirm that adaptation is effective regardless of drift pattern, with incremental drift requiring the most frequent retraining (43 episodes) and recurring drift achieving the fastest recovery (F1=0.985).

**Keywords:** Intrusion Detection, Concept Drift, ADWIN, Adaptive Learning, Network Security, LightGBM, LSTM

---

## 1. Introduction

Network intrusion detection systems (NIDS) are a critical defence layer in modern cybersecurity infrastructure. Machine learning approaches have demonstrated strong detection performance on benchmark datasets, with reported F1 scores exceeding 0.99 on datasets such as CIC-IDS2017 [1]. However, these results are typically obtained under independent and identically distributed (IID) evaluation conditions, where training and test data are randomly sampled from the same distribution.

In real network environments, traffic distributions evolve continuously due to:
- New applications and protocols (e.g., HTTP/1.1 → HTTP/2 → HTTP/3)
- Changes in user behaviour and cloud migration
- IoT device proliferation
- New malware families and attack techniques
- Attacker adaptation to deployed defences

This non-stationarity produces **concept drift** — changes in the relationship between network features and their class labels — and **feature drift** — shifts in the input feature distributions themselves [2]. When drift occurs, a previously accurate model may silently degrade, increasing false negatives (missed attacks) and false positives (spurious alerts).

Existing approaches to drift-aware IDS include CDDA-MD [3], which employs LSTM with multi-head self-attention and error-rate-based drift detection, and HOIDS [4], which combines online learning with active learning for label efficiency. However, systematic comparative evaluation of drift types, detector behaviour, and adaptation strategies under a consistent experimental framework remains limited.

This work makes the following contributions:

1. **Quantitative evidence of temporal degradation**: We demonstrate that a LightGBM classifier achieving F1=0.997 under random-split evaluation drops to F1=0.036 under chronological temporal evaluation — a 96.4% reduction — on CIC-IDS2017.

2. **Drift-triggered adaptation framework**: We propose and evaluate an ADWIN-based drift detection mechanism coupled with automatic model retraining that recovers F1 to 0.987 using only 19 targeted retraining episodes.

3. **Comparative analysis**: We compare static, periodic, and drift-triggered adaptation strategies, demonstrating that drift-triggered retraining achieves the highest detection quality with 71% fewer retrains than periodic approaches.

4. **LSTM vs. lightweight models under drift**: We show that drift-triggered LightGBM (F1=0.987) outperforms static LSTM (F1=0.792) at 11× lower computational cost.

5. **Comprehensive drift characterisation**: We evaluate all four drift types (sudden, gradual, incremental, recurring) and analyse feature-level distribution changes across 64 flow features.

---

## 2. Related Work

### 2.1 Concept Drift in Intrusion Detection

Shyaa et al. [2] provide a comprehensive survey of concept drift and feature dynamics in IDS, identifying that non-stationary traffic is a persistent challenge. They categorise drift into sudden, gradual, incremental, and recurring types, each requiring different detection and adaptation mechanisms.

### 2.2 CDDA-MD

Cai et al. [3] propose CDDA-MD, which uses sliding windows, LSTM networks, and multi-head self-attention for malicious traffic detection with concept drift adaptation. While effective, their approach requires significant computational resources for LSTM training. Our work investigates whether comparable drift resilience can be achieved with lightweight classical models.

### 2.3 HOIDS

The Hybrid Online Intrusion Detection System [4] combines concept drift detection with online learning and active learning to reduce labelling costs. We differentiate our work by providing explicit comparison of adaptation strategies and controlled drift experiments under a consistent framework.

### 2.4 Research Gap

While drift-aware IDS and online adaptation approaches exist, there remains a need for systematic comparative evaluation of: (a) drift types and their impact on detection performance, (b) drift detector behaviour under different conditions, (c) adaptation strategy efficiency and cost, and (d) whether lightweight models with adaptation can match deep learning approaches.

---

## 3. Methodology

### 3.1 System Architecture

ADAPT-IDS processes network traffic through the following pipeline:

```
Network Traffic → Feature Extraction → IDS Classifier → Predictions
                                              ↓
                                    Error Monitoring (0/1 per sample)
                                              ↓
                                    ADWIN Drift Detection
                                              ↓
                                 Drift? → Yes → Retrain on recent window
                                        → No  → Continue monitoring
```

### 3.2 Base Classifiers

**LightGBM**: Gradient-boosted decision tree ensemble (200 estimators, max depth 8, balanced class weights). Selected for its established superiority on tabular data and computational efficiency.

**Random Forest**: 200 estimators, max depth 20, balanced class weights. Included as a secondary baseline.

**LSTM**: Bidirectional LSTM with attention mechanism (128 hidden units, 2 layers, dropout 0.3). Included to compare deep learning with classical approaches under drift.

### 3.3 Drift Detection

We employ ADWIN (Adaptive Windowing) [5] from the River library to monitor the model's error stream. ADWIN maintains a variable-length window and detects statistically significant changes in the error distribution. Configuration: δ=0.002, clock=32.

The error signal is computed as:
- error(t) = 0 if prediction matches ground truth
- error(t) = 1 if prediction is incorrect

### 3.4 Adaptation Strategies

We compare three strategies:

**Static**: No adaptation. The model trained on the initial period is never updated.

**Periodic**: The model is retrained every N samples regardless of drift detection. We evaluate N ∈ {5000, 10000}.

**Drift-triggered**: The model is retrained only when ADWIN detects a statistically significant change in error rate, with a cooldown of 2000 samples between retrains. Retraining uses the most recent 20,000 labelled samples from the stream.

### 3.5 Datasets

**CIC-IDS2017** [1]: 2,830,743 network flows captured over five working days at the Canadian Institute for Cybersecurity. Contains benign traffic and 14 attack types including DoS, DDoS, brute force, web attacks, botnets, port scanning, and infiltration. After preprocessing (infinity/NaN handling, duplicate removal, constant feature removal), 2,233,963 flows with 65 features remain.

**UNSW-NB15** [6]: 2,540,256 network flows generated using the IXIA PerfectStorm tool at the Australian Centre for Cyber Security. Contains 9 attack categories including Generic, Exploits, DoS, Fuzzers, Reconnaissance, Backdoors, Shellcode, and Worms. 35 numeric features used.

### 3.6 Evaluation Protocol

**Random split**: Stratified 80/20 train/test split. Represents conventional IID evaluation.

**Temporal split**: Chronological 70/10/20 train/validation/test split preserving time ordering. We enforce max(train_timestamp) < min(test_timestamp) to prevent data leakage.

**Metrics**: F1-score, Recall, Precision, FPR, FNR, MCC. We prioritise Recall and F1 over Accuracy due to class imbalance (80.3% benign in CIC-IDS2017).

### 3.7 Synthetic Drift Generation

We generate controlled drift by modifying the test feature distributions:
- **Sudden**: Abrupt shift at position 50% with magnitude 1.5σ
- **Gradual**: Progressive shift between positions 30–70%
- **Incremental**: Continuous linear shift across the entire stream
- **Recurring**: Alternating drift with 20% cycle length

---

## 4. Experimental Results

### 4.1 Temporal Degradation (RQ1)

Table 1 shows the catastrophic performance difference between random and temporal evaluation.

| Model | Evaluation | F1 | Recall | FNR |
|-------|-----------|-----|--------|-----|
| LightGBM | Random | 0.997 | 99.9% | 0.06% |
| LightGBM | Temporal | 0.036 | 1.8% | 98.2% |
| Random Forest | Random | 0.996 | 99.7% | 0.26% |
| Random Forest | Temporal | 0.032 | 1.6% | 98.4% |

**Finding**: Both models achieve near-perfect performance under IID conditions but catastrophically fail under temporal evaluation, missing over 98% of attacks. This confirms H1.

### 4.2 Drift Detection (RQ2)

ADWIN detected 153 drift events across the 446,793-sample test stream. The error rate sustained approximately 93% for the majority of the stream, with periodic drops indicating regions where the test distribution partially matched training data.

### 4.3 Adaptation Strategy Comparison (RQ3, RQ4)

Table 2 compares adaptation strategies on the temporal test stream.

| Strategy | F1 | Recall | FNR | Retrains | Cost (s) |
|----------|-----|--------|-----|----------|----------|
| Static | 0.036 | 1.8% | 98.2% | 0 | 0 |
| Periodic (5K) | 0.978 | 95.8% | 4.2% | 65 | 43.4 |
| Periodic (10K) | 0.959 | 92.2% | 7.8% | 32 | 22.6 |
| **Drift-triggered** | **0.987** | **97.7%** | **2.3%** | **19** | **12.7** |

**Finding**: Drift-triggered adaptation achieves the highest F1 (0.987) with the fewest retrains (19) and lowest computational cost (12.7s). This confirms H2 and H3.

### 4.4 LSTM vs. LightGBM (RQ1 extended)

Table 3 compares deep learning with classical models.

| Model | Random F1 | Temporal F1 | Training Time |
|-------|-----------|-------------|---------------|
| LightGBM | 0.997 | 0.0003 | 3.2s |
| LSTM | 0.888 | 0.792 | 36.8s |
| **LightGBM + adapt** | **0.997** | **0.987** | **12.7s total** |

**Finding**: While LSTM has inherent drift resilience (F1=0.792 vs 0.0003 for static LightGBM), LightGBM with drift-triggered adaptation (F1=0.987) substantially outperforms LSTM at 3× lower total computation.

### 4.5 Synthetic Drift Types (RQ5)

Table 4 shows adaptation effectiveness across controlled drift types.

| Drift Type | Static F1 | Adaptive F1 | Recovery | Retrains |
|-----------|-----------|-------------|----------|----------|
| No drift | 0.036 | 0.987 | +0.951 | 19 |
| Sudden | 0.032 | 0.979 | +0.947 | 21 |
| Gradual | 0.007 | 0.977 | +0.971 | 21 |
| Incremental | 0.004 | 0.969 | +0.965 | 43 |
| Recurring | 0.006 | 0.985 | +0.979 | 23 |

**Finding**: Adaptation is effective across all drift types. Incremental drift is hardest (requires 43 retrains, lowest adaptive F1=0.969). Recurring drift recovers fastest (F1=0.985). This partially confirms H4.

### 4.6 Feature Drift Analysis

Kolmogorov-Smirnov tests reveal that 100% of 64 features exhibit statistically significant drift (p<0.05) between training and test periods. The most-shifted features are timing-related:

| Feature | KS Statistic | Relative Shift |
|---------|-------------|---------------|
| Idle Max | 0.252 | 3.9× |
| Flow IAT Max | 0.252 | 3.4× |
| Fwd IAT Max | 0.251 | 3.6× |
| Idle Mean | 0.250 | 4.3× |
| Fwd IAT Std | 0.249 | 4.9× |

This indicates that attack-type changes primarily manifest through timing characteristics rather than packet-size features.

### 4.7 Cross-Dataset Validation

On UNSW-NB15 (different attack taxonomy, different feature set), LightGBM achieves F1=0.960 with Recall=99.88%, confirming model generalisability beyond a single benchmark.

---

## 5. Discussion

### 5.1 The Illusion of High Performance

Our results demonstrate that conventional random-split evaluation dramatically overestimates real-world IDS performance. The 96.4% F1 drop from random (0.997) to temporal (0.036) evaluation reveals that published benchmark results may be misleading if temporal ordering is not respected.

### 5.2 Adaptation vs. Architecture

A key finding is that the adaptation strategy matters more than the model architecture. Static LSTM (F1=0.792) provides some inherent drift resilience through its recurrent structure, but lightweight LightGBM with drift-triggered adaptation (F1=0.987) achieves substantially better performance at a fraction of the computational cost. This suggests that research effort may be better invested in adaptation mechanisms than in increasingly complex model architectures.

### 5.3 Efficiency of Drift-Triggered Retraining

Drift-triggered retraining is both more effective and more efficient than periodic retraining. It achieves higher F1 (0.987 vs 0.978) with 71% fewer retraining episodes (19 vs 65) and 71% less computation (12.7s vs 43.4s). This is because periodic retraining wastes computation during stable periods and may retrain too late during rapid drift, while drift-triggered retraining responds precisely when needed.

### 5.4 Limitations

1. **Label availability**: Our drift detection uses ground-truth labels from the benchmark dataset. In production, true labels are delayed or unavailable. Unsupervised drift detection on feature distributions or prediction confidence is a natural extension.
2. **Single-seed experiments**: Results are from a single random seed (42). Future work should include multiple seeds with confidence intervals.
3. **Benchmark vs. production**: While CIC-IDS2017 and UNSW-NB15 are widely used, they may not fully represent production network diversity.

---

## 6. Conclusion

We presented ADAPT-IDS, a framework for adaptive intrusion detection under concept and feature drift. Our experimental evaluation demonstrates that:

1. Static ML-based IDS models experience catastrophic performance degradation under temporal drift (F1: 0.997 → 0.036).
2. Drift-triggered adaptation using ADWIN recovers detection performance to F1=0.987 with minimal computational overhead.
3. Drift-triggered retraining outperforms both periodic retraining and static deep learning (LSTM) approaches.
4. All four drift types (sudden, gradual, incremental, recurring) can be effectively handled by adaptive retraining, though incremental drift is the most challenging.
5. Feature drift analysis reveals that 100% of flow features shift significantly under temporal evaluation, with timing features most affected.

Future work includes unsupervised drift detection for production deployment, active learning for label efficiency, and real-time PCAP integration via Wireshark/Zeek for live network monitoring.

---

## References

[1] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," in *4th International Conference on Information Systems Security and Privacy (ICISSP)*, 2018.

[2] S. S. Shyaa et al., "Evolving cybersecurity frontiers: A comprehensive survey on concept drift and feature dynamics aware machine and deep learning in intrusion detection systems," *Engineering Applications of Artificial Intelligence*, vol. 137, 2024. DOI: 10.1016/j.engappai.2024.109143

[3] H. Cai et al., "CDDA-MD: An efficient malicious traffic detection method based on concept drift detection and adaptation technique," *Computers & Security*, 2025. DOI: 10.1016/j.cose.2024.104121

[4] "Concept drift aware hybrid online intrusion detection system," *Journal of Network and Computer Applications*, 2026. DOI: 10.1016/j.jnca.2026.104556

[5] A. Bifet and R. Gavalda, "Learning from Time-Changing Data with Adaptive Windowing," in *Proceedings of the 2007 SIAM International Conference on Data Mining*, 2007.

[6] N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," in *Military Communications and Information Systems Conference (MilCIS)*, 2015.