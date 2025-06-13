import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Software Bug Prediction",
    page_icon="🐛",
    layout="wide"
)

# Title and description
st.title("🐛 Software Bug Prediction")
st.markdown("""
This application predicts the likelihood of bugs in software code based on various code metrics.
The model was trained on data from multiple open-source projects including Eclipse JDT Core, Eclipse PDE UI,
Equinox Framework, Lucene, and Mylyn.
""")

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except:
        return None, None

# Create input fields for all metrics
st.header("Input Code Metrics")

col1, col2 = st.columns(2)

with col1:
    cbo = st.number_input("Coupling Between Objects (CBO)", min_value=0)
    dit = st.number_input("Depth of Inheritance Tree (DIT)", min_value=0)
    fan_in = st.number_input("Fan In", min_value=0)
    fan_out = st.number_input("Fan Out", min_value=0)
    lcom = st.number_input("Lack of Cohesion of Methods (LCOM)", min_value=0)
    noc = st.number_input("Number of Children (NOC)", min_value=0)
    num_attributes = st.number_input("Number of Attributes", min_value=0)
    num_attributes_inherited = st.number_input("Number of Inherited Attributes", min_value=0)

with col2:
    num_lines = st.number_input("Number of Lines of Code", min_value=0)
    num_methods = st.number_input("Number of Methods", min_value=0)
    num_methods_inherited = st.number_input("Number of Inherited Methods", min_value=0)
    num_private_attributes = st.number_input("Number of Private Attributes", min_value=0)
    num_private_methods = st.number_input("Number of Private Methods", min_value=0)
    num_public_attributes = st.number_input("Number of Public Attributes", min_value=0)
    num_public_methods = st.number_input("Number of Public Methods", min_value=0)
    rfc = st.number_input("Response For Class (RFC)", min_value=0)
    wmc = st.number_input("Weighted Methods per Class (WMC)", min_value=0)

# Create a button for prediction
if st.button("Predict Bug Probability"):
    # Create input array
    input_data = np.array([[
        cbo, dit, fan_in, fan_out, lcom, noc, num_attributes,
        num_attributes_inherited, num_lines, num_methods,
        num_methods_inherited, num_private_attributes,
        num_private_methods, num_public_attributes,
        num_public_methods, rfc, wmc
    ]])
    
    # Load model and scaler
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.error("Model files not found. Please ensure model.joblib and scaler.joblib are in the correct location.")
    else:
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict_proba(input_scaled)
        
        # Display results
        st.header("Prediction Results")
        
        # Create a gauge chart for bug probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction[0][1] * 100,
            title={'text': "Bug Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        st.plotly_chart(fig)
        
        # Display detailed probabilities
        st.subheader("Detailed Probabilities")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("No Bugs", f"{prediction[0][0]*100:.2f}%")
        with col2:
            st.metric("Has Bugs", f"{prediction[0][1]*100:.2f}%")

# Add information about the metrics
st.header("About the Metrics")
st.markdown("""
The following metrics are used to predict the likelihood of bugs in the code:

- **CBO (Coupling Between Objects)**: Measures the coupling between classes
- **DIT (Depth of Inheritance Tree)**: The length of the maximum path from the class to the root class
- **Fan In**: Number of classes that depend on this class
- **Fan Out**: Number of classes this class depends on
- **LCOM (Lack of Cohesion of Methods)**: Measures the lack of cohesion in methods
- **NOC (Number of Children)**: Number of immediate subclasses
- **RFC (Response For Class)**: Number of methods that can be executed in response to a message
- **WMC (Weighted Methods per Class)**: Sum of the complexity of all methods in a class
""")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 