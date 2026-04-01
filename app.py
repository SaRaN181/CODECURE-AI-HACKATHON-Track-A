import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, QED, rdFingerprintGenerator

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


weights = {
    # --- Nuclear Receptors (binding risk) ---
    'NR-AR': 0.9,
    'NR-AR-LBD': 0.9,
    'NR-AhR': 1.1,
    'NR-Aromatase': 1.0,
    'NR-ER': 1.1,
    'NR-ER-LBD': 1.0,
    'NR-PPAR-gamma': 0.9,

    # --- Stress Response (actual toxicity) ---
    'SR-ARE': 1.3,
    'SR-ATAD5': 1.5,
    'SR-HSE': 1.2,
    'SR-MMP': 1.3,
    'SR-p53': 1.6
}

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="CodeCure | Multi-Pathway Toxicity AI", layout="wide", page_icon="🧪")
# Updated CSS to fix visibility in both Dark and Light modes
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: #ffffff; padding: 15px; border-radius: 10px;
        border: 1px solid #e1e4e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetric"] label, [data-testid="stMetric"] div { color: #1a1a1a !important; }
    [data-testid="stMetricDelta"] div { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 CodeCure: AI Pharmacology Dashboard")
st.subheader("Multi-Pathway Toxicity Profile (12-Assay Diagnostic Engine)")

MODEL_SYSTEM_PATH = "Model/multi_pathway_toxicity_system.pkl"

@st.cache_resource
def load_toxicity_system():
    return joblib.load(MODEL_SYSTEM_PATH)

try:
    models_dict = load_toxicity_system()
except:
    st.error(f"⚠️ Multi-model system '{MODEL_SYSTEM_PATH}' not found. Please ensure the training script has finished.")
    st.stop()

def extract_features_advanced(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    # 1. Fingerprint
    fp = morgan_gen.GetFingerprint(mol)
    fp_array = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    
    # 2. Basic Properties (LogP, MW)
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    
    # 3. QED (Drug-likeness)
    qed_score = QED.qed(mol) 
    
    return np.append(fp_array, [logp, mw, qed_score])


# --- 2. SIDEBAR ---
st.sidebar.image("https://img.icons8.com/fluency/96/000000/microscope.png")
st.sidebar.header("Molecular Input")
smiles_input = st.sidebar.text_area("Paste SMILES String:", height=100)
st.sidebar.markdown("---")
st.sidebar.write("**Methodology:**")
st.sidebar.caption("1. 12-Pathway Multi-Label Ensemble\n2. ZINC250k Safe Benchmarking\n3. Morgan Fingerprint Fragments")

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        user_logp = Descriptors.MolLogP(mol)
        user_qed = QED.qed(mol)
        user_mw = Descriptors.MolWt(mol)
        ZINC_LOGP, ZINC_QED = 2.4571, 0.7283
        
        # PREDICTION ACROSS ALL 12 PATHWAYS
        features = extract_features_advanced(smiles_input)
        if features is None:
            st.error("❌ Feature extraction failed. Invalid or unsupported molecule.")
            st.stop()
        pathway_risks = {}
        weighted_risks = {}
        for name, model in models_dict.items():
            prob = model.predict_proba([features])[0][1]
            
            pathway_risks[name] = prob
            weighted_risks[name] = min(prob * weights.get(name, 1.0), 1.0)
        
        # Sort risks for visualization
        sorted_risks = dict(sorted(weighted_risks.items(), key=lambda item: item[1], reverse=True))
        top_3 = list(sorted_risks.items())[:3]

        # UI LAYOUT
        tab1, tab2, tab3 = st.tabs(["🧬 Dashboard", "📊 Feature Analysis", "📜 History"])
        
        with tab1:
            top_col1, top_col2 = st.columns([1, 2])
            
            with top_col1:
                st.markdown("### 2D Visualization")
                st.image(Draw.MolToImage(mol, size=(400, 400)), width='stretch')
                
            with top_col2:
                st.markdown("### AI Diagnostic Summary")


                # --- Toxicity Decision Logic ---
                nr_paths = [k for k in pathway_risks if k.startswith("NR")]
                sr_paths = [k for k in pathway_risks if k.startswith("SR")]

                nr_vals = [weighted_risks[k] for k in nr_paths]
                sr_vals = [weighted_risks[k] for k in sr_paths]

                nr_max = max(nr_vals)
                sr_max = max(sr_vals)

                # --- Decision Tree ---
                if sr_max > 0.55 and nr_max > 0.55:
                    final_label = "🚨 SEVERE TOXICITY (Receptor + Cellular Damage)"
                    final_color = "error"

                elif sr_max > 0.55:
                    final_label = "🔴 HIGH TOXICITY (Cell Stress / Damage)"
                    final_color = "error"

                elif nr_max > 0.55:
                    final_label = "🟡 MODERATE RISK (Receptor Activation)"
                    final_color = "warning"

                else:
                    final_label = "🟢 LOW RISK"
                    final_color = "success"

                # --- Save to History ---
                if 'last_smiles' not in st.session_state:
                    st.session_state.last_smiles = None
                
                if st.session_state.last_smiles != smiles_input:
                    history_file = "history.csv"
                    new_entry = pd.DataFrame([{
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "SMILES": smiles_input,
                        "Prediction": final_label
                    }])
                    write_header = not os.path.exists(history_file) or os.path.getsize(history_file) == 0
                    new_entry.to_csv(history_file, mode='a', header=write_header, index=False)
                    st.session_state.last_smiles = smiles_input
                
                # --- UI Output ---
                if final_color == "error":
                    st.error(f"## {final_label}")
                elif final_color == "warning":
                    st.warning(f"## {final_label}")
                else:
                    st.success(f"## {final_label}")
                
                st.caption(f"Max NR Activity: {nr_max:.2f} | Max SR Activity: {sr_max:.2f}")
                st.write("---")
                
                risk_values = list(weighted_risks.values())
                confidence = np.std(risk_values)
                st.metric("Prediction Confidence", f"{confidence:.3f}")

                if confidence < 0.1:
                    st.caption("⚠️ Low confidence: pathway signals are similar")

                m1, m2, m3 = st.columns(3)
                m1.metric("LogP (Solubility)", f"{user_logp:.2f}", f"{user_logp - ZINC_LOGP:.2f} vs ZINC", delta_color="inverse")
                m2.metric("Drug-likeness (QED)", f"{user_qed:.2f}", f"{user_qed - ZINC_QED:.2f} vs ZINC")
                m3.metric("Mol. Weight", f"{user_mw:.1f}")

            # 12-PATHWAY BAR CHART
            st.write("---")
            st.subheader("🧬 Multi-Pathway Toxicity Profile")
            st.caption("Detailed risk probability across 12 unique biological receptors (Tox21 Assays)")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(sorted_risks.keys()), 
                    y=list(sorted_risks.values()),
                    marker_color=['#d62728' if v > 0.6 else '#1f77b4' for v in sorted_risks.values()]
                )
            ])
            threshold = 0.6 if sr_max > 0.6 else 0.5
            fig.add_hline(y=threshold, line_dash="dash", line_color="orange", annotation_text="Dynamic Risk Threshold")
            fig.update_layout(yaxis_title="Probability", xaxis_tickangle=-45, height=400, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, width='stretch')

            # COMPARATIVE ANALYSIS
            st.write("---")
            st.subheader("📊 ZINC250k Benchmark Analysis")
            comp_fig = go.Figure(data=[
                go.Bar(name='Input Molecule', x=['LogP', 'QED'], y=[user_logp, user_qed], marker_color='#1f77b4'),
                go.Bar(name='ZINC Average (Safe)', x=['LogP', 'QED'], y=[ZINC_LOGP, ZINC_QED], marker_color='#d62728')
            ])
            comp_fig.update_layout(barmode='group', height=300)
            st.plotly_chart(comp_fig, width='stretch')

            # DYNAMIC INTERPRETABILITY
            with st.expander("🔬 Scientific Decision Logic"):

                qed_status = "Superior" if user_qed > ZINC_QED else "Sub-optimal"

                st.markdown("**Top Contributing Pathways:**")

                for name, val in top_3:
                    st.write(f"- {name}: {val:.2f} (weighted)")

                top_pathways = [name for name, val in top_3]

                if sr_max > 0.55 and nr_max > 0.55:
                    st.error(f"""
                    **Toxicity Insight:** The compound activates both receptor-mediated pathways 
                    and stress-response mechanisms. Key pathways include {', '.join(top_pathways)}. 
                    This suggests a complete toxicological profile involving both molecular initiation 
                    and downstream cellular damage.
                    """)

                elif sr_max > 0.55:
                    st.warning(f"""
                    **Toxicity Insight:** Strong activation of stress-response pathways 
                    (notably {', '.join(top_pathways)}) indicates potential cellular damage such as 
                    oxidative stress, DNA damage, or mitochondrial dysfunction.
                    """)

                elif nr_max > 0.4:   
                    st.info(f"""
                    **Toxicity Insight:** The compound shows moderate receptor-level interaction, 
                    particularly via {', '.join(top_pathways)}. This suggests potential molecular 
                    initiating events without strong downstream cellular damage.
                    """)

                else:
                    st.success("""
                    **Toxicity Insight:** No significant receptor or stress pathway activation detected. 
                    The molecule aligns with a safe pharmacological profile.
                    """)

                high_nr = [k for k in nr_paths if weighted_risks[k] > 0.6]
                high_sr = [k for k in sr_paths if weighted_risks[k] > 0.6]

                if high_nr:
                    st.write("**Activated NR Pathways:**", ", ".join(high_nr))

                if high_sr:
                    st.write("**Activated SR Pathways:**", ", ".join(high_sr))

        with tab3:
            st.markdown("### 📜 Prediction History")
            st.caption("A record of all molecules evaluated in this session or previously saved locally.")
            history_file = "history.csv"
            if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        has_header = "Timestamp" in f.readline()
                    
                    if has_header:
                        history_df = pd.read_csv(history_file)
                    else:
                        history_df = pd.read_csv(history_file, names=["Timestamp", "SMILES", "Prediction"])
                    
                    st.dataframe(history_df, width='stretch')
                    
                    if st.button("🗑️ Clear History"):
                        os.remove(history_file)
                        st.session_state.last_smiles = None
                        st.rerun()
                except Exception as e:
                    st.info("No prediction history available or error loading it.")
            else:
                st.info("No prediction history found. Start by entering a SMILES string!")

        with tab2:
            st.markdown("### 🔍 Feature Importance Analysis")
            st.caption("How different molecular properties and structural features contribute to the AI's predictions.")
            
            # Feature extraction from models
            all_importances = []
            for name, model in models_dict.items():
                if hasattr(model, 'feature_importances_'):
                    all_importances.append(model.feature_importances_)
            
            if all_importances:
                avg_importances = np.mean(all_importances, axis=0)
                
                # Sum of all Morgan Fingerprint bits vs global properties
                morgan_imp = np.sum(avg_importances[:2048])
                logp_imp = avg_importances[2048]
                mw_imp = avg_importances[2049]
                qed_imp = avg_importances[2050]
                
                # Normalize
                total_imp = morgan_imp + logp_imp + mw_imp + qed_imp
                if total_imp > 0:
                    morgan_imp /= total_imp
                    logp_imp /= total_imp
                    mw_imp /= total_imp
                    qed_imp /= total_imp
                
                # Visual 1: Structural vs Property Importance
                st.subheader("Structural vs. Physicochemical Impact")
                pie_fig = go.Figure(data=[go.Pie(
                    labels=['Structural Features (Morgan FP)', 'Lipophilicity (LogP)', 'Molecular Weight', 'Drug-likeness (QED)'],
                    values=[morgan_imp, logp_imp, mw_imp, qed_imp],
                    hole=.3,
                    marker_colors=['#8c564b', '#2ca02c', '#d62728', '#1f77b4']
                )])
                pie_fig.update_layout(height=400, margin=dict(l=20,r=20,t=20,b=20))
                st.plotly_chart(pie_fig, width='stretch')
                
                # Visual 2: Relationship between Properties and Toxicity
                st.write("---")
                st.subheader("🧪 Property Toxicity Map")
                st.caption("Positions the current molecule against Safe vs. Toxic property zones.")
                
                # Note: creating a generic toxicity map based on LogP and QED.
                # Usually high LogP (>5) is toxic/poor solubility, low QED (<0.4) is non-drug-like.
                map_fig = go.Figure()

                # Add safe zone background
                map_fig.add_shape(type="rect", x0=-2, y0=0.5, x1=5, y1=1.1,
                                  fillcolor="LightGreen", opacity=0.3, layer="below", line_width=0)
                
                # Add current molecule
                map_fig.add_trace(go.Scatter(
                    x=[user_logp], y=[user_qed],
                    mode='markers+text',
                    marker=dict(size=15, color='red' if sr_max > 0.55 or nr_max > 0.55 else 'blue'),
                    name='This Molecule',
                    text=['<br>Input'], textposition="top center"
                ))
                
                # Add ZINC average
                map_fig.add_trace(go.Scatter(
                    x=[ZINC_LOGP], y=[ZINC_QED],
                    mode='markers+text',
                    marker=dict(size=12, color='green', symbol='star'),
                    name='ZINC Average (Safe)',
                    text=['<br>Avg Safe'], textposition="bottom center"
                ))
                
                map_fig.update_layout(
                    xaxis_title="LogP (Lipophilicity) - Ideally < 5",
                    yaxis_title="QED (Drug-likeness) - Ideally > 0.5",
                    xaxis=dict(range=[-2, 8]),
                    yaxis=dict(range=[0, 1.1]),
                    height=400, margin=dict(l=20,r=20,t=20,b=20)
                )
                
                # Add thresholds lines
                map_fig.add_vline(x=5, line_dash="dash", line_color="red")
                map_fig.add_hline(y=0.5, line_dash="dash", line_color="red")
                
                st.plotly_chart(map_fig, width='stretch')
            else:
                st.info("Feature importances not available for this model type.")

    else:
        st.error("❌ Invalid SMILES string.")

else:
    st.info("👈 Paste a SMILES string in the sidebar to begin analysis.")
    st.markdown("### 📜 Prediction History")
    history_file = "history.csv"
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                has_header = "Timestamp" in f.readline()
            
            if has_header:
                history_df = pd.read_csv(history_file)
            else:
                history_df = pd.read_csv(history_file, names=["Timestamp", "SMILES", "Prediction"])
                
            st.dataframe(history_df, width='stretch')
            
            if st.button("🗑️ Clear History", key="clear_empty"):
                os.remove(history_file)
                st.rerun()
        except Exception as e:
            st.info("No prediction history available or error loading it.")
