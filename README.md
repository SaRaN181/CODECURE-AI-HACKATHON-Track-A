# 🧪 CodeCure: AI Pharmacology Dashboard

![CodeCure Dashboard](https://img.icons8.com/fluency/96/000000/microscope.png)

**CodeCure** is an AI-powered diagnostic engine that analyzes molecular structures (SMILES strings) to perform a multi-pathway toxicity profile across 12 unique biological receptors (Tox21 Assays). This project is designed to aid researchers in rapidly screening chemical compounds for potential toxicological risks.

---

## 🚀 Deliverables

- [x] **GitHub Repository**: Clean, well-structured codebase with comprehensive documentation.
- [x] **Machine Learning Model**: Multi-pathway toxicity predictor trained on Tox21 data (`Model/multi_pathway_toxicity_system.pkl`).
- [x] **Feature Importance Analysis**: Interactive UI displaying the global impact of Structural vs. Physicochemical properties on AI predictions.
- [x] **Visualizations**: Property Toxicity map detailing LogP and QED compared to ZINC250k Safe Benchmarks.
- [x] **Simple Prediction Tool**: Interactive Streamlit Dashboard for easy input and analysis.

---

## 🛠️ Technology Stack
- **Python 3.12+**
- **Machine Learning**: `xgboost`, `scikit-learn`, `pandas`, `numpy`
- **Chemoinformatics**: `rdkit`
- **Frontend / UI**: `streamlit`, `plotly`

---

## ⚙️ Local Setup and Execution

### 1. Requirements
Ensure you have Python installed. The required packages are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 2. Running the Dashboard
To launch the AI Pharmacology Dashboard locally, run the following Streamlit command:
```bash
streamlit run app.py
```

### 3. Usage Guide
1. Obtain a SMILES string for your molecule (e.g., `CCOc1ccc2nc(S(N)(=O)=O)sc2c1`).
2. Paste it into the `Molecular Input` sidebar on the dashboard.
3. Review the **2D Visualization**, the **AI Diagnostic Summary**, and the exhaustive 12-Pathway risk chart in the `Dashboard` tab.
4. Switch to the **Feature Analysis** tab to see how structural features vs. physicochemical descriptors influenced the risk scoring, and view the Property Toxicity Map.

---

## 🔬 Scientific Methodology

The AI architecture is composed of 12 independent `XGBClassifier` models, each rigorously trained and tested on binary assays from the Tox21 challenge. 
We extracted 2,051 features per molecule:
- **2048-bit Morgan Fingerprints** (Radius=2) reflecting topological substructures.
- **LogP** (Lipophilicity/Solubility).
- **Molecular Weight** (MW).
- **QED** (Quantitative Estimate of Drug-likeness).

The prediction confidence is evaluated by generating probability scores across all 12 pathways. The results are compared against average thresholds compiled from the **ZINC250k** (Safe) drug benchmark to deliver dynamic and comprehensive interpretability.
