import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('vehicle_price_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'vehicle_price_model.pkl' is in the same directory.")
        return None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found! Please ensure 'dataset.csv' is in the same directory.")
        return None

model = load_model()
df = load_dataset()

st.title("🚗 Vehicle Price Predictor")
st.markdown("### Predict the price of a vehicle based on its features")
st.divider()

if model is not None and df is not None:
    year_min = int(df['year'].min()) if 'year' in df.columns else 1990
    year_max = int(df['year'].max()) if 'year' in df.columns else 2024
    year = st.selectbox(
        "Year",
        options=sorted(range(year_min, year_max + 1), reverse=True),
        index=0
    )
    
    names = sorted(df['name'].dropna().unique().tolist()) if 'name' in df.columns else []
    name = st.selectbox("Vehicle Name", options=names)
    
    makes = sorted(df['make'].dropna().unique().tolist()) if 'make' in df.columns else []
    make = st.selectbox("Make", options=makes)
    
    if make:
        models = sorted(df[df['make'] == make]['model'].dropna().unique().tolist())
    else:
        models = sorted(df['model'].dropna().unique().tolist()) if 'model' in df.columns else []
    model_name = st.selectbox("Model", options=models)
    
    fuel_types = sorted(df['fuel'].dropna().unique().tolist()) if 'fuel' in df.columns else []
    fuel = st.selectbox("Fuel Type", options=fuel_types)
    
    mileage = st.number_input(
        "Mileage (miles)",
        min_value=0,
        max_value=500000,
        value=50000,
        step=1000
    )
    
    st.divider()
    
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        cylinders = df['cylinders'].mode()[0] if 'cylinders' in df.columns and not df['cylinders'].empty else 4
        doors = df['doors'].mode()[0] if 'doors' in df.columns and not df['doors'].empty else 4
        engine = df['engine'].mode()[0] if 'engine' in df.columns and not df['engine'].empty else None
        transmission = df['transmission'].mode()[0] if 'transmission' in df.columns and not df['transmission'].empty else None
        trim = df['trim'].mode()[0] if 'trim' in df.columns and not df['trim'].empty else None
        body = df['body'].mode()[0] if 'body' in df.columns and not df['body'].empty else None
        exterior_color = df['exterior_color'].mode()[0] if 'exterior_color' in df.columns and not df['exterior_color'].empty else None
        interior_color = df['interior_color'].mode()[0] if 'interior_color' in df.columns and not df['interior_color'].empty else None
        drivetrain = df['drivetrain'].mode()[0] if 'drivetrain' in df.columns and not df['drivetrain'].empty else None
        
        input_data = pd.DataFrame({
            'year': [year],
            'cylinders': [cylinders],
            'mileage': [mileage],
            'doors': [doors],
            'name': [name],
            'make': [make],
            'model': [model_name],
            'engine': [engine],
            'fuel': [fuel],
            'transmission': [transmission],
            'trim': [trim],
            'body': [body],
            'exterior_color': [exterior_color],
            'interior_color': [interior_color],
            'drivetrain': [drivetrain]
        })
        
        try:
            prediction = model.predict(input_data)[0]
            
            st.success("### Prediction Complete!")
            st.metric(
                label="Estimated Vehicle Price",
                value=f"${prediction:,.2f}"
            )
            
            with st.expander("📋 Input Summary"):
                st.write(f"**Year:** {year}")
                st.write(f"**Vehicle Name:** {name}")
                st.write(f"**Make:** {make}")
                st.write(f"**Model:** {model_name}")
                st.write(f"**Fuel:** {fuel}")
                st.write(f"**Mileage:** {mileage:,} miles")
                    
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all required fields are filled correctly.")

else:
    st.warning("⚠️ Unable to load model or dataset. Please ensure the required files are available.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Vehicle Price Predictor | Powered by Machine Learning 🤖"
    "</div>",
    unsafe_allow_html=True
)