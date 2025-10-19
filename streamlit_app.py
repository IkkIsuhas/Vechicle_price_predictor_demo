import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Page configuration
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Fix selectbox styling for better visibility */
    .stSelectbox > div > div {
        background-color: white !important;
        color: #262730 !important;
    }
    
    .stSelectbox > div > div > div {
        color: #262730 !important;
        background-color: white !important;
    }
    
    /* Dropdown options styling */
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: white !important;
        color: #262730 !important;
    }
    
    /* Selected value styling */
    .stSelectbox [data-baseweb="select"] span {
        color: #262730 !important;
        font-weight: 500;
    }
    
    /* Dropdown menu styling */
    .stSelectbox [role="listbox"] {
        background-color: white !important;
    }
    
    .stSelectbox [role="option"] {
        color: #262730 !important;
        background-color: white !important;
    }
    
    .stSelectbox [role="option"]:hover {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        color: #262730 !important;
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model = joblib.load('vehicle_price_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'vehicle_price_model.pkl' not found. Please run the training script first.")
        return None

@st.cache_data
def load_dataset():
    """Load and preprocess the dataset to get unique values"""
    try:
        df = pd.read_csv('dataset.csv')
        df = df[(df['price'] > 1000) & (df['price'] < 200000)]
        df = df[df['mileage'] >= 0]
        return df
    except FileNotFoundError:
        st.error("Dataset file 'dataset.csv' not found.")
        return None

def get_unique_values(df, column):
    """Get unique values for a column, handling NaN values"""
    unique_vals = df[column].dropna().unique()
    return sorted([str(val) for val in unique_vals if pd.notna(val)])

def create_input_form(df):
    """Create the input form with all required fields"""
    
    # Define column mappings
    num_cols = ['year', 'cylinders', 'mileage', 'doors']
    cat_cols = ['name', 'make', 'model', 'engine', 'fuel', 'transmission', 'trim', 'body', 'exterior_color', 'interior_color', 'drivetrain']
    
    st.markdown('<h2 class="main-header">🚗 Vehicle Price Predictor</h2>', unsafe_allow_html=True)
    st.markdown("### Enter Vehicle Details")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        
        # Numerical inputs
        year = st.number_input(
            "Year", 
            min_value=1990, 
            max_value=2025, 
            value=2023,
            help="Manufacturing year of the vehicle"
        )
        
        cylinders = st.number_input(
            "Number of Cylinders", 
            min_value=2, 
            max_value=12, 
            value=6,
            help="Number of engine cylinders"
        )
        
        mileage = st.number_input(
            "Mileage", 
            min_value=0, 
            max_value=500000, 
            value=50000,
            help="Vehicle mileage in miles"
        )
        
        doors = st.number_input(
            "Number of Doors", 
            min_value=2, 
            max_value=6, 
            value=4,
            help="Number of doors"
        )
        
        # Categorical inputs - first column
        make = st.selectbox(
            "Make", 
            options=get_unique_values(df, 'make'),
            help="Vehicle manufacturer"
        )
        
        model = st.selectbox(
            "Model", 
            options=get_unique_values(df, 'model'),
            help="Vehicle model"
        )
        
        fuel = st.selectbox(
            "Fuel Type", 
            options=get_unique_values(df, 'fuel'),
            help="Type of fuel used"
        )
        
        transmission = st.selectbox(
            "Transmission", 
            options=get_unique_values(df, 'transmission'),
            help="Transmission type"
        )
    
    with col2:
        st.markdown("#### Additional Details")
        
        # Categorical inputs - second column
        name = st.selectbox(
            "Vehicle Name", 
            options=get_unique_values(df, 'name'),
            help="Full vehicle name/trim"
        )
        
        engine = st.selectbox(
            "Engine", 
            options=get_unique_values(df, 'engine'),
            help="Engine specification"
        )
        
        trim = st.selectbox(
            "Trim Level", 
            options=get_unique_values(df, 'trim'),
            help="Vehicle trim level"
        )
        
        body = st.selectbox(
            "Body Type", 
            options=get_unique_values(df, 'body'),
            help="Vehicle body style"
        )
        
        exterior_color = st.selectbox(
            "Exterior Color", 
            options=get_unique_values(df, 'exterior_color'),
            help="Exterior color"
        )
        
        interior_color = st.selectbox(
            "Interior Color", 
            options=get_unique_values(df, 'interior_color'),
            help="Interior color"
        )
        
        drivetrain = st.selectbox(
            "Drivetrain", 
            options=get_unique_values(df, 'drivetrain'),
            help="Drive system type"
        )
    
    # Create input dictionary
    input_data = {
        'year': year,
        'cylinders': cylinders,
        'mileage': mileage,
        'doors': doors,
        'name': name,
        'make': make,
        'model': model,
        'engine': engine,
        'fuel': fuel,
        'transmission': transmission,
        'trim': trim,
        'body': body,
        'exterior_color': exterior_color,
        'interior_color': interior_color,
        'drivetrain': drivetrain
    }
    
    return input_data

def predict_price(model, input_data):
    """Make prediction using the loaded model"""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def display_prediction(prediction):
    """Display the prediction result in a nice format"""
    if prediction is not None:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Predicted Price")
        
        # Format the price
        formatted_price = f"${prediction:,.2f}"
        
        # Display with different styling based on price range
        if prediction < 20000:
            st.markdown(f'<h1 style="color: #28a745; text-align: center;">{formatted_price}</h1>', unsafe_allow_html=True)
            st.markdown("**Price Range:** Budget-friendly")
        elif prediction < 50000:
            st.markdown(f'<h1 style="color: #ffc107; text-align: center;">{formatted_price}</h1>', unsafe_allow_html=True)
            st.markdown("**Price Range:** Mid-range")
        else:
            st.markdown(f'<h1 style="color: #dc3545; text-align: center;">{formatted_price}</h1>', unsafe_allow_html=True)
            st.markdown("**Price Range:** Premium")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Load model and dataset
    model = load_model()
    df = load_dataset()
    
    if model is None or df is None:
        st.stop()
    
    # Create input form
    input_data = create_input_form(df)
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                prediction = predict_price(model, input_data)
                display_prediction(prediction)
    
    # Sidebar information
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.markdown("""
        **Model Type:** Random Forest Regressor
        
        **Features Used:**
        - Year, Cylinders, Mileage, Doors
        - Make, Model, Engine, Fuel Type
        - Transmission, Trim, Body Type
        - Exterior & Interior Colors
        - Drivetrain
        
        **Training Data:** Vehicle dataset with price filtering ($1K - $200K)
        """)
        
        st.markdown("## 💡 Tips")
        st.markdown("""
        - Select realistic combinations
        - Higher mileage = Lower price
        - Newer vehicles = Higher price
        - Premium brands = Higher price
        """)
        
        # Display some statistics
        st.markdown("## 📈 Dataset Statistics")
        st.metric("Total Vehicles", f"{len(df):,}")
        st.metric("Average Price", f"${df['price'].mean():,.0f}")
        st.metric("Price Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        
        # Show unique counts
        st.markdown("## 🔢 Unique Values")
        for col in ['make', 'model', 'fuel', 'body', 'drivetrain']:
            st.metric(f"{col.title()}", len(get_unique_values(df, col)))

if __name__ == "__main__":
    main()
