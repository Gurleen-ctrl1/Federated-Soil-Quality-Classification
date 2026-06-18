# Hybrid Winsorized Federated Averaging for Robust Soil Quality Classification Across Distributed Agricultural IoT Networks

## Overview

This repository presents a Federated Learning framework for robust soil quality classification in distributed agricultural environments. The project addresses one of the key challenges in federated systems: handling heterogeneous client distributions and noisy local updates while preserving privacy and enabling collaborative learning.

The proposed framework investigates Winsorization-based aggregation strategies integrated with Federated Averaging (FedAvg) to improve convergence stability, client fairness, and overall model performance in agricultural IoT networks.

Unlike conventional centralized learning approaches, the system enables multiple geographically distributed farms to collaboratively train a global model without sharing raw data, thereby preserving data ownership and privacy.

---

## Research Motivation

Modern precision agriculture relies on large-scale data collected from distributed sensors, edge devices, and IoT infrastructures deployed across multiple farms. While centralized machine learning can leverage such data effectively, it introduces challenges related to privacy, communication costs, scalability, and data governance.

Federated Learning provides an alternative paradigm by allowing local model training at each client while sharing only model parameters with a central server.

However, standard Federated Averaging (FedAvg) often struggles when:

* Client data distributions are highly heterogeneous (non-IID)
* Local updates contain noisy or extreme parameter values
* Certain clients dominate the aggregation process
* Global convergence becomes unstable

This work explores robust aggregation mechanisms based on Winsorization to mitigate the influence of extreme updates while maintaining participation from all clients.

---

## Key Contributions

* Design and implementation of a distributed Federated Learning framework for soil quality classification.
* Investigation of heterogeneous client environments representative of real-world agricultural deployments.
* Development of Winsorization-based aggregation strategies for robust model training.
* Comparative evaluation of multiple aggregation approaches under distributed settings.
* Analysis of convergence behaviour, client-level fairness, and global model performance.
* Simulation of collaborative learning across multiple agricultural clients without centralizing raw data.

---

## System Architecture

The framework follows a client-server Federated Learning architecture.

```text
                    Global Aggregation Server
                                |
        -----------------------------------------------------
        |            |            |            |            |
      Client 1    Client 2    Client 3    Client 4    Client 5
        |            |            |            |            |
      Local        Local        Local        Local        Local
     Training     Training     Training     Training     Training
        |            |            |            |            |
        -----------------------------------------------------
                                |
                    Updated Global Model
```

### Workflow

1. Global model initialization.
2. Distribution of model parameters to participating clients.
3. Local training on client-specific data.
4. Collection of local model updates.
5. Aggregation at the central server.
6. Global model update.
7. Repetition across multiple communication rounds until convergence.

At no stage is raw client data shared with the server.

---

## Methodology

The project evaluates multiple federated aggregation strategies.

### Setup 1: Baseline FedAvg

The standard Federated Averaging algorithm is used throughout all communication rounds.

### Setup 2: Uniform Winsorization

Following an initial warm-up phase, Winsorization is applied to client updates prior to aggregation to reduce the impact of extreme parameter values.

### Setup 3: Threshold-Based Winsorization

A selective Winsorization strategy is employed in which client updates are monitored using a predefined threshold. Divergent updates are adjusted before aggregation, improving robustness while preserving client participation.

These experimental configurations enable analysis of the impact of robust aggregation techniques on distributed learning performance.

---

## Experimental Objectives

The study evaluates:

* Global model accuracy
* Client-wise performance
* Convergence behaviour
* Communication-round efficiency
* Stability under heterogeneous client distributions
* Fairness across participating clients

The framework is designed to investigate how robust aggregation strategies influence collaborative model training in practical federated environments.

---

## Repository Structure

```text
federated_learning_soil_project/

├── data/
│   ├── client_data/
│   └── client_data_uneven/
│
├── figures/
│   ├── plots/
│   ├── plots_final/
│   └── plots_uneven/
│
├── results/
│   ├── federated_training_log.csv
│   ├── federated_training_log_final.csv
│   └── federated_uneven.csv
│
├── src/
│   ├── client.py
│   ├── edge_server.py
│   ├── federated_main.py
│   ├── model.py
│   ├── server.py
│   └── utils.py
│
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Gurleen-ctrl1/federated_learning_soil_project.git

cd federated_learning_soil_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the federated learning pipeline:

```bash
python src/federated_main.py
```

The framework will:

* Initialize participating clients
* Train local models
* Perform federated aggregation
* Track convergence metrics
* Generate performance logs
* Produce evaluation visualizations

---

## Results

Experimental evaluation demonstrates that Winsorization-based aggregation improves learning stability and mitigates the influence of noisy client updates compared to standard Federated Averaging.

The framework provides:

* Improved convergence behaviour
* Better client-level consistency
* Enhanced robustness under heterogeneous client environments
* More stable global model performance across communication rounds

Detailed logs and visualizations are available in the `results` and `figures` directories.

---

## Applications

* Precision Agriculture
* Agricultural IoT Systems
* Edge AI
* Federated Learning
* Distributed Machine Learning
* Privacy-Preserving Analytics
* Smart Farming Infrastructure

---

## Future Work

* Weighted Winsorized Aggregation (WWA)
* Metadata-aware client trust scoring
* Adaptive aggregation mechanisms
* Real-world IoT deployment using LoRaWAN
* Edge deployment on NVIDIA Jetson platforms
* Large-scale federated agricultural ecosystems
* Integration of advanced deep learning architectures such as TabNet

---

## Authors

**Gurleen Kaur Bhatia**
Department of Computer Science and Engineering
Maulana Azad National Institute of Technology (MANIT), Bhopal

---
