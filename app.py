import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd

# --- Page Configuration (MUST be the first Streamlit command) ---
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
        # Note: These files must exist in the same directory as this script.
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

# Load the models and label encoder
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
Ia_val = st.sidebar.slider("Phase A Current (Ia)", -1000.0, 1000.0, 10.0)
Ib_val = st.sidebar.slider("Phase B Current (Ib)", -1000.0, 1000.0, 10.0)
Ic_val = st.sidebar.slider("Phase C Current (Ic)", -1000.0, 1000.0, 10.0)

st.sidebar.subheader("Voltages (pu)")
Va_val = st.sidebar.slider("Phase A Voltage (Va)", -1.0, 1.0, 1.0)
Vb_val = st.sidebar.slider("Phase B Voltage (Vb)", -1.0, 1.0, 1.0)
Vc_val = st.sidebar.slider("Phase C Voltage (Vc)", -1.0, 1.0, 1.0)

# Function to generate sinusoidal data
def generate_sinusoidal_data(amplitude_a, amplitude_b, amplitude_c, prediction_class, is_current=True):
    """
    Generates time-series data for sinusoidal waves with optional fault simulation.
    
    Args:
        amplitude_a (float): The base amplitude for phase A.
        amplitude_b (float): The base amplitude for phase B.
        amplitude_c (float): The base amplitude for phase C.
        prediction_class (int): The numerical class of the predicted fault.
        is_current (bool): True if generating current data, False for voltage.

    Returns:
        pd.DataFrame: A DataFrame with 'Time', 'Phase', and 'Value' columns.
    """
    time = np.linspace(0, 0.05, 500)  # Simulate 3 cycles for a 60 Hz system
    data = []
    
    # Define phase shifts for a balanced three-phase system (120 degrees apart)
    phase_shift_a = 0
    phase_shift_b = -2 * np.pi / 3  # -120 degrees
    phase_shift_c = 2 * np.pi / 3   # +120 degrees

    # Apply fault effects to the waves
    fault_multiplier = 0.5  # Example: 50% voltage drop or 2x current spike
    
    # Adjust amplitudes based on fault type
    # For a simplified visual representation of the fault
    if prediction_class != 0:
        if is_current:
            # Current spikes in faulted phases
            if prediction_class in [1, 4, 7, 10, 11, 12]: # A-phase faults
                amplitude_a *= (1 + fault_multiplier)
            if prediction_class in [2, 8, 9, 11, 12]: # C-phase faults
                amplitude_c *= (1 + fault_multiplier)
            if prediction_class in [3, 6, 9, 10, 11, 12, 13]: # B-phase faults
                amplitude_b *= (1 + fault_multiplier)
        else:
            # Voltage dips in faulted phases
            if prediction_class in [1, 4, 7, 10, 11, 12]: # A-phase faults
                amplitude_a *= (1 - fault_multiplier)
            if prediction_class in [2, 8, 9, 11, 12]: # C-phase faults
                amplitude_c *= (1 - fault_multiplier)
            if prediction_class in [3, 6, 9, 10, 11, 12, 13]: # B-phase faults
                amplitude_b *= (1 - fault_multiplier)

    # Generate data points
    data.append(pd.DataFrame({'Time': time, 'Phase': 'A', 'Value': amplitude_a * np.sin(2 * np.pi * 60 * time + phase_shift_a)}))
    data.append(pd.DataFrame({'Time': time, 'Phase': 'B', 'Value': amplitude_b * np.sin(2 * np.pi * 60 * time + phase_shift_b)}))
    data.append(pd.DataFrame({'Time': time, 'Phase': 'C', 'Value': amplitude_c * np.sin(2 * np.pi * 60 * time + phase_shift_c)}))

    return pd.concat(data)

# --- Main Page Content ---
st.title("Electrical Fault Prediction Dashboard")

# Main content columns
col1, col2 = st.columns([2, 1])

# Initialize prediction_class to 0 (No Fault) for initial display
prediction_class = 0

with col1:
    st.subheader("Real-Time System Data Visualization")
    
    # Generate and plot sinusoidal waves based on current prediction
    current_data = generate_sinusoidal_data(Ia_val, Ib_val, Ic_val, prediction_class, is_current=True)
    voltage_data = generate_sinusoidal_data(Va_val, Vb_val, Vc_val, prediction_class, is_current=False)

    fig_current = px.line(current_data, x='Time', y='Value', color='Phase',
                          title='Current Measurements (Amperes)',
                          labels={'Value': 'Current (A)'},
                          color_discrete_map={'A': '#E91E63', 'B': '#2196F3', 'C': '#FFC107'})
    fig_current.update_layout(height=400, title_x=0.5)

    fig_voltage = px.line(voltage_data, x='Time', y='Value', color='Phase',
                          title='Voltage Measurements (pu)',
                          labels={'Value': 'Voltage (pu)'},
                          color_discrete_map={'A': '#E91E63', 'B': '#2196F3', 'C': '#FFC107'})
    fig_voltage.update_layout(height=400, title_x=0.5)

    st.plotly_chart(fig_current, use_container_width=True)
    st.plotly_chart(fig_voltage, use_container_width=True)

    # Calculate and display power metrics
    st.subheader("Calculated Power Values")
    st.markdown("---")
    
    # Base voltage for per-unit conversion
    base_voltage_V = 11000  # 11 kV base voltage in Volts

    # Calculate Apparent Power (S = V_actual * I)
    S_A = (Va_val * base_voltage_V) * Ia_val
    S_B = (Vb_val * base_voltage_V) * Ib_val
    S_C = (Vc_val * base_voltage_V) * Ic_val
    
    total_S = S_A + S_B + S_C

    # Display the results in columns in kVA
    power_col1, power_col2, power_col3 = st.columns(3)
    
    with power_col1:
        st.metric(label="Apparent Power Phase A", value=f"{S_A / 1000:.2f} kVA")
    with power_col2:
        st.metric(label="Apparent Power Phase B", value=f"{S_B / 1000:.2f} kVA")
    with power_col3:
        st.metric(label="Apparent Power Phase C", value=f"{S_C / 1000:.2f} kVA")

    st.markdown("<br>", unsafe_allow_html=True)
    st.metric(label="Total Apparent Power", value=f"{total_S / 1000:.2f} kVA")


with col2:
    st.subheader(f"Predict with {selected_model_name}")
    st.markdown("---")
    if st.button("Predict Fault"):
        input_features = np.array([[Ia_val, Ib_val, Ic_val, Va_val, Vb_val, Vc_val]])

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
            st.success("NO FAULT DETECTED ✅", icon="✅")
            st.markdown(f"<p class='big-font'>The system is operating normally.</p>", unsafe_allow_html=True)
            st.info("The electrical system appears to be stable and healthy.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- Final description ---
st.markdown("---")
st.markdown("This dashboard was built using a machine learning model trained on a dataset of electrical parameters to predict the presence and type of electrical faults.")
