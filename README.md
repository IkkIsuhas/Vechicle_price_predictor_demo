# Vehicle Price Predictor - Streamlit App

A machine learning web application that predicts vehicle prices based on various features using a Random Forest model.

## Features

- **Interactive UI**: Easy-to-use interface with dropdown selections for all vehicle features
- **Real-time Predictions**: Instant price predictions as you select vehicle options
- **Comprehensive Inputs**: 
  - Basic info: Year, cylinders, mileage, doors
  - Vehicle details: Make, model, engine, fuel type, transmission
  - Appearance: Trim, body type, exterior/interior colors, drivetrain
- **Visual Feedback**: Color-coded price ranges and formatted predictions
- **Model Information**: Sidebar with dataset statistics and model details

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (if not already done):
   ```bash
   python vehicle.py
   ```
   This will create `vehicle_price_model.pkl`

3. **Run the Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open in Browser**: The app will automatically open at `http://localhost:8501`

## Usage

1. Select vehicle features using the dropdown menus
2. Click "🔮 Predict Price" button
3. View the predicted price with color-coded formatting
4. Check sidebar for additional information and statistics

## Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: 15 input features (4 numerical, 11 categorical)
- **Preprocessing**: StandardScaler for numerical features, OrdinalEncoder for categorical features
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold cross-validation

## File Structure

```
├── vehicle.py              # Model training script
├── streamlit_app.py        # Streamlit web application
├── dataset.csv             # Vehicle dataset
├── vehicle_price_model.pkl  # Trained model (created after training)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- NumPy