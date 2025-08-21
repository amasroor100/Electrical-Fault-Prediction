import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Electrical Fault Prediction", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for a modern, clean look ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        font-size: 20px;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        border-radius: 12px;
        border: none;
        padding: 12px 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .st-emotion-cache-1wv9v3w p {
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 8px solid;
    }
    .prediction-container.no-fault {
        border-color: #28a745;
    }
    .prediction-container.fault {
        border-color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Load all models and the label encoder ---
@st.cache_resource
def load_models():
    """Loads all models and the LabelEncoder from their pickle files."""
    models = {}
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        with open('xgboost_model.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
        with open('decision_tree_model.pkl', 'rb') as f:
            models['Decision Tree'] = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Error: One or more model files were not found. Please ensure all required .pkl files are in the same directory.")
        st.stop()
    return models, le

models, le = load_models()


# --- Sidebar for user inputs ---
st.sidebar.title("⚡ Control Panel")
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a Prediction Model",
    list(models.keys())
)
selected_model = models[selected_model_name]

st.sidebar.header("Input Electrical Parameters")
st.sidebar.markdown("Adjust the sliders to simulate system conditions.")

st.sidebar.subheader("Currents (Amperes)")
Ia = st.sidebar.slider("Phase A Current (Ia)", -1000.0, 1000.0, 10.0)
Ib = st.sidebar.slider("Phase B Current (Ib)", -1000.0, 1000.0, 10.0)
Ic = st.sidebar.slider("Phase C Current (Ic)", -1000.0, 1000.0, 10.0)

st.sidebar.subheader("Voltages (pu)")
Va = st.sidebar.slider("Phase A Voltage (Va)", -1.0, 1.0, 1.0)
Vb = st.sidebar.slider("Phase B Voltage (Vb)", -1.0, 1.0, 1.0)
Vc = st.sidebar.slider("Phase C Voltage (Vc)", -1.0, 1.0, 1.0)

# --- Main Page Content ---
st.title("⚡ Electrical Fault Prediction Dashboard")
st.markdown("### Powered by Machine Learning")

# Main content columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Real-Time System Data Visualization")
    
    # Create a DataFrame for plotting
    data_points = {
        'Phase': ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'],
        'Value': [Ia, Ib, Ic, Va, Vb, Vc],
        'Type': ['Current', 'Current', 'Current', 'Voltage', 'Voltage', 'Voltage']
    }
    df_plot = pd.DataFrame(data_points)
    
    # Plot current and voltage graphs
    fig_current = px.bar(df_plot[df_plot['Type'] == 'Current'], x='Phase', y='Value', 
                         title='Current Measurements', color='Phase',
                         labels={'Value': 'Current (A)'},
                         color_discrete_map={'Ia': '#E91E63', 'Ib': '#2196F3', 'Ic': '#FFC107'})
    fig_current.update_layout(height=400, title_x=0.5)

    fig_voltage = px.bar(df_plot[df_plot['Type'] == 'Voltage'], x='Phase', y='Value', 
                         title='Voltage Measurements', color='Phase',
                         labels={'Value': 'Voltage (pu)'},
                         color_discrete_map={'Va': '#E91E63', 'Vb': '#2196F3', 'Vc': '#FFC107'})
    fig_voltage.update_layout(height=400, title_x=0.5)
    
    st.plotly_chart(fig_current, use_container_width=True)
    st.plotly_chart(fig_voltage, use_container_width=True)
    
with col2:
    st.subheader(f"Predict with {selected_model_name}")
    st.markdown("---")
    if st.button("Predict Fault"):
        input_features = np.array([[Ia, Ib, Ic, Va, Vb, Vc]])

        # Make prediction based on selected model
        prediction_class = selected_model.predict(input_features)[0]

        # Handle XGBoost's specific label encoding
        if selected_model_name == 'XGBoost':
            # The label encoder is used to transform the numerical prediction back to a string label
            prediction_class = le.inverse_transform([prediction_class])[0]

        # Map the prediction class to the fault name
        fault_mapping = {
            0: "No Fault Detected",
            1: "Single Line A Fault",
            2: "Line C to Ground (CG)",
            3: "Line B to Ground (BG)",
            4: "Line A to Ground (AG)",
            5: "Line A to C (AC)",
            6: "Line B to C (BC)",
            7: "Line A to B (AB)",
            8: "Line C, A to Ground (CAG)",
            9: "Line B, C to Ground (BCG)",
            10: "Line A, B to Ground (ABG)",
            11: "Line A, B, C (ABC)",
            12: "Line A, B, C to Ground (ABCG)",
            13: "Single Line B Fault",
            14: "Single Line C Fault",
            15: "Single Ground Fault (G)"
        }
        
        fault_name = fault_mapping.get(prediction_class, "Unknown Fault Type")

        if prediction_class != 0:
            st.markdown('<div class="prediction-container fault">', unsafe_allow_html=True)
            st.error(f"🚨 FAULT DETECTED! 🚨", icon="⚠️")
            st.markdown(f"<p class='big-font'>**Fault Type:** {fault_name}</p>", unsafe_allow_html=True)
            st.warning("Immediate action is required to diagnose and resolve the issue.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-container no-fault">', unsafe_allow_html=True)
            st.success("✅ NO FAULT DETECTED ✅", icon="✅")
            st.markdown(f"<p class='big-font'>The system is operating normally.</p>", unsafe_allow_html=True)
            st.info("The electrical system appears to be stable and healthy.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- Final description ---
st.markdown("---")
st.markdown("This dashboard was built using a machine learning model trained on a dataset of electrical parameters to predict the presence and type of electrical faults.")
