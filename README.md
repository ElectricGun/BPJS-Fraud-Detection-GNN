# Grafana: Integrasi Graph Database untuk Fraud Detection dengan Graph Neural Networks (GraphSAGE-GAT & XGBoost Ensemble) & Algoritma Louvain

<div align="center">

<table style="border: none; margin: 0 auto; padding: 0; border-collapse: collapse;">
<tr>
<td align="center" style="vertical-align: middle; padding: 10px; border: none; width: 250px;">
  <img src="img/grafana_logo.png" alt="GRAFANA Logo" width="200"/>
</td>
<td align="left" style="vertical-align: middle; padding: 10px 0 10px 30px; border: none;">
  <pre style="font-family: 'Courier New', monospace; font-size: 16px; color: #0EA5E9; margin: 0; padding: 0; text-shadow: 0 0 10px #0EA5E9, 0 0 20px rgba(14,165,233,0.5); line-height: 1.2; transform: skew(-1deg, 0deg); display: block;">

░██████╗░██████╗░░█████╗░███████╗░█████╗░███╗░░██╗░█████╗░
██╔════╝░██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗░██║██╔══██╗
██║░░██╗░██████╔╝███████║█████╗░░███████║██╔██╗██║███████║
██║░░╚██╗██╔══██╗██╔══██║██╔══╝░░██╔══██║██║╚████║██╔══██║
╚██████╔╝██║░░██║██║░░██║██║░░░░░██║░░██║██║░╚███║██║░░██║
░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝
  </pre>
</td>
</tr>
</table>

<p>
  <img src="https://img.shields.io/badge/Neo4j-GraphDB-00d9ff?style=for-the-badge&logo=neo4j&logoColor=white"/>
  <img src="https://img.shields.io/badge/GDS-Graph_Data_Science-4ecdc4?style=for-the-badge&logo=protodotio&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-ETL_Scripts-f39c12?style=for-the-badge&logo=python&logoColor=white"/>
</p>

<div align="center">
<a href="https://trendshift.io/repositories/14665" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14665" alt="Grafana Team" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center" style="width: 100%; height: 2px; margin: 20px 0; background: linear-gradient(90deg, transparent, #00d9ff, transparent);"></div>
</div>

> **GRAFANA** (Graph Fraud Analytics) adalah sistem deteksi fraud cerdas yang menggabungkan kekuatan **Neo4j Graph Database** dengan arsitektur Deep Learning **Hybrid GNN (GraphSAGE + GAT)** dan **XGBoost Ensemble**.
>
> Sistem ini tidak hanya memetakan hubungan pasien-klaim, tetapi juga mempelajari pola struktural (embedding) untuk memprediksi anomali dengan akurasi tinggi, divisualisasikan langsung melalui **Neo4j Bloom**.
---

## 📑 **Table of Contents**

* [✨ Features](#-features)
* [🏗️ Architecture](#️-architecture)
* [⚙️ Setup Environment](#️-setup-environment)
* [📥 Data Loading (ETL)](#-data-loading-etl)
* [🧠 Graph Projection + Louvain](#-graph-projection--louvain)
* [🌐 Visualizations](#-visualizations)
* [📁 Export for GNN](#-export-for-gnn)
* [📄 License](#-license)

---

## ✨ **Features**

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="33%" align="center" style="padding: 20px;">
<h3>🔗 Knowledge Graph Construction</h3>
<img src="https://img.shields.io/badge/Neo4j-Graph_Modeling-00d9ff?style=for-the-badge&logo=neo4j" />
<p>Mengubah data tabular mentah menjadi graf cerdas yang menghubungkan entitas <b>Patient, Claim, Doctor,</b> dan <b>Hospital</b> untuk mengungkap relasi tersembunyi.</p>
</td>
<td width="33%" align="center" style="padding: 20px;">
<h3>🧬 Structural Feature Engineering</h3>
<img src="https://img.shields.io/badge/Algo-Louvain_&_Node2Vec-4ecdc4?style=for-the-badge" />
<p>Mengekstraksi fitur graf tingkat lanjut menggunakan algoritma <b>Louvain Community Detection</b> dan <b>Node2Vec Embeddings</b> untuk menangkap konteks komunitas fraud.</p>
</td>
<td width="33%" align="center" style="padding: 20px;">
<h3>🤖 Hybrid AI Prediction</h3>
<img src="https://img.shields.io/badge/Model-GraphSAGE_+_GAT_+_XGBoost-f39c12?style=for-the-badge&logo=pytorch" />
<p>Model ensemble yang menggabungkan kekuatan induktif <b>GraphSAGE</b>, mekanisme atensi <b>GAT</b>, dan boosting <b>XGBoost</b> untuk klasifikasi risiko tinggi.</p>
</td>
</tr>
</table>

---

## 🏗️ **Architecture & Pipeline**

```
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 520" width="100%">
.box{fill:#ffffff;stroke:#3a7;stroke-width:2;rx:8;}
.small{font:14px/1.1 Arial, sans-serif;}
.title{font:700 18px/1.1 Arial, sans-serif;}
.emoji{font-size:20px}
.arrow{fill:none;stroke:#7a7a7a;stroke-width:2;marker-end:url(#arr)}
</style>
<defs>
<marker id="arr" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto">
<path d="M0 0 L10 5 L0 10 z" fill="#7a7a7a" />
</marker>
</defs>


<!-- Data Ingestion -->
<rect x="20" y="30" width="210" height="70" class="box"/>
<text x="30" y="55" class="title">📄 Raw CSV Data</text>
<text x="30" y="75" class="small">ETL: load_data.py</text>


<!-- Neo4j -->
<rect x="270" y="20" width="200" height="100" class="box"/>
<text x="285" y="55" class="title">🍃 Neo4j Database</text>


<!-- GDS -->
<rect x="520" y="20" width="220" height="100" class="box"/>
<text x="540" y="55" class="title">⚙️ Neo4j GDS</text>
<text x="540" y="75" class="small">Projection · Embedding · Detection</text>


<!-- AI Core -->
<rect x="820" y="10" width="260" height="170" class="box"/>
<text x="840" y="35" class="title">🧠 Hybrid AI Engine</text>
<text x="840" y="60" class="small">Louvain → Node2Vec → GraphSAGE + GAT</text>


<!-- XGBoost / Result -->
<rect x="550" y="160" width="220" height="90" class="box"/>
<text x="570" y="190" class="title">XGBoost Classifier</text>
<text x="570" y="210" class="small">Risk Score & Explanations</text>


<rect x="820" y="200" width="220" height="70" class="box"/>
<text x="840" y="235" class="title">📄 Final Report CSV</text>


<!-- Bloom -->
<rect x="270" y="170" width="200" height="80" class="box"/>
<text x="285" y="200" class="title">🌸 Neo4j Bloom</text>
<text x="285" y="220" class="small">Visual Investigation</text>


<!-- Arrows -->
<path class="arrow" d="M230 65 L270 65" />
<path class="arrow" d="M470 65 L520 65" />
<path class="arrow" d="M740 65 L820 65" />


<path class="arrow" d="M640 160 L640 120 L740 120 L740 130" />
<path class="arrow" d="M740 205 L740 235 L820 235" />


<path class="arrow" d="M470 200 L520 200" />
<path class="arrow" d="M470 200 L300 200" />
<path class="arrow" d="M370 200 L270 200" />


<!-- Labels near arrows -->
<text x="125" y="50" class="small">ETL: load_data.py →</text>
<text x="390" y="40" class="small">Graph Projection →</text>
<text x="680" y="40" class="small">Export Features →</text>


</svg>
```

# ⚙️ Setup Environment

Panduan ini menjelaskan seluruh instalasi dari nol hingga siap menjalankan pipeline GRAFANA.

## 🧱 1. System Requirements

* Python ≥ 3.10
* Neo4j Desktop / Neo4j AuraDB
* CUDA (opsional, untuk training GNN)
* Pip & Virtualenv

---

## 🐍 2. Create Virtual Environment

```bash
git clone https://github.com/username/GRAFANA
cd GRAFANA
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Library inti:

* `neo4j`
* `pandas`, `numpy`
* `networkx`
* `torch`, `pyg` (PyTorch Geometric)
* `matplotlib`

---

# 🏗️ 4. Neo4j Setup

## 4.1 Instalasi Neo4j Desktop

Download: [https://neo4j.com/download/](https://neo4j.com/download/)

Setelah instalasi:

1. Buat database baru
2. Gunakan password: `neo4j` (atau custom)
3. Jalankan database

## 4.2 Import Data

Gunakan file `etl/claims.csv`, `etl/providers.csv`, dll.

Contoh import (Neo4j Browser):

```cypher
LOAD CSV WITH HEADERS FROM 'file:///claims.csv' AS row
CREATE (:Claim {
    claim_id: row.claim_id,
    amount: toFloat(row.amount),
    date: row.date
});
```

---

# 🔗 5. Graph Model Design

## Node Types

* **Claim**
* **Patient**
* **Provider**
* **Hospital**

## Relationship Types

* `(:Patient)-[:SUBMITTED]->(:Claim)`
* `(:Provider)-[:HANDLED]->(:Claim)`
* `(:Provider)-[:WORKS_AT]->(:Hospital)`

Diagram:

```
Patient ---SUBMITTED---> Claim <---HANDLED--- Provider ---WORKS_AT---> Hospital
```

---

# 🔄 6. ETL Pipeline

File: `etl/extract_to_neo4j.py`

### 6.1 Extract

```python
import pandas as pd
claims = pd.read_csv('data/claims.csv')
```

### 6.2 Transform

```python
claims['amount_norm'] = (claims['amount'] - claims['amount'].mean()) / claims['amount'].std()
```

### 6.3 Load to Neo4j

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(URI, auth=(USER, PASS))
```

---

# 👁️ 7. Graph Visualization

## 7.1 Neo4j Browser

Gunakan:

```cypher
MATCH (c:Claim)-[r]-(n)
RETURN * LIMIT 50;
```

## 7.2 Python Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt
```

---

# 🧠 8. GNN Training

Menggunakan PyTorch Geometric.

## 8.1 Convert Neo4j → PyG

File: `gnn/neo4j_to_pyg.py`

Pipeline:

1. Query nodes & relationships
2. Encode categorical entities
3. Build `edge_index`
4. Build `node_features`

## 8.2 Train Model

File: `gnn/train.py`

Model: GraphSAGE / GAT

```python
model = GraphSAGE(hidden_channels=64)
```

## 8.3 Evaluate

```python
accuracy, f1 = evaluate(model, loader)
```

---

# 📁 9. Project Structure

```
GRAFANA/
│── etl/
│   ├── extract_to_neo4j.py
│   ├── claims.csv
│   └── providers.csv
│
│── gnn/
│   ├── neo4j_to_pyg.py
│   ├── train.py
│   └── model.py
│
│── assets/
│── README.md
│── requirements.txt
```

---

# 🚀 10. Quick Start

```bash
python etl/extract_to_neo4j.py
python gnn/neo4j_to_pyg.py
python gnn/train.py
```

## 📄 **License**

MIT License
