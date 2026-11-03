"""
config.py
---------
Central configuration for the Real Estate Investment Advisor project.
Every script (preprocessing, training, evaluation, Streamlit app) imports
paths and constants from here so that nothing is hard-coded twice.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (all relative to the project root, so the project runs on any machine)
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "india_housing_prices.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned_housing_data.csv"
CITY_STATS_PATH = ROOT_DIR / "data" / "processed" / "city_reference_stats.csv"

MODELS_DIR = ROOT_DIR / "models"
CLASSIFIER_PATH = MODELS_DIR / "investment_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "future_price_regressor.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"

# MLflow: a local SQLite backend is required for the Model Registry
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_DIR = ROOT_DIR / "mlruns"
MLFLOW_EXPERIMENT = "Real_Estate_Investment_Advisor"
REGISTERED_CLASSIFIER = "RealEstate_GoodInvestment_Classifier"
REGISTERED_REGRESSOR = "RealEstate_FuturePrice_Regressor"

# ----------------------------------------------------------------------------
# Domain constants
# ----------------------------------------------------------------------------
REFERENCE_YEAR = 2025          # dataset's Age_of_Property = 2025 - Year_Built
FORECAST_YEARS = 5             # horizon for the future-price target
BASE_GROWTH_RATE = 0.08        # 8 % p.a. base appreciation (from project brief)
RANDOM_STATE = 42
TEST_SIZE = 0.20

# City tiers (used for location-based growth assumptions)
TIER_1_CITIES = {
    "Mumbai", "New Delhi", "Dwarka", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Gurgaon", "Noida",
}
TIER_2_CITIES = {
    "Jaipur", "Lucknow", "Surat", "Nagpur", "Indore", "Bhopal", "Kochi",
    "Coimbatore", "Vishakhapatnam", "Bhubaneswar", "Chandigarh", "Ludhiana",
    "Faridabad", "Patna", "Vijayawada", "Mysore", "Trivandrum", "Guwahati",
    "Dehradun", "Raipur", "Ranchi", "Amritsar", "Mangalore", "Jodhpur",
}
# Every other city in the data is treated as Tier 3.

# Approximate city coordinates (for the location map in the app / EDA)
CITY_COORDS = {
    "Ahmedabad": (23.02, 72.57), "Amritsar": (31.63, 74.87), "Bangalore": (12.97, 77.59),
    "Bhopal": (23.26, 77.41), "Bhubaneswar": (20.30, 85.82), "Bilaspur": (22.08, 82.15),
    "Chennai": (13.08, 80.27), "Coimbatore": (11.02, 76.96), "Cuttack": (20.46, 85.88),
    "Dehradun": (30.32, 78.03), "Durgapur": (23.52, 87.31), "Dwarka": (28.59, 77.05),
    "Faridabad": (28.41, 77.32), "Gaya": (24.79, 85.00), "Gurgaon": (28.46, 77.03),
    "Guwahati": (26.14, 91.74), "Haridwar": (29.95, 78.16), "Hyderabad": (17.39, 78.49),
    "Indore": (22.72, 75.86), "Jaipur": (26.91, 75.79), "Jamshedpur": (22.80, 86.20),
    "Jodhpur": (26.24, 73.02), "Kochi": (9.93, 76.27), "Kolkata": (22.57, 88.36),
    "Lucknow": (26.85, 80.95), "Ludhiana": (30.90, 75.86), "Mangalore": (12.91, 74.86),
    "Mumbai": (19.08, 72.88), "Mysore": (12.30, 76.64), "Nagpur": (21.15, 79.09),
    "New Delhi": (28.61, 77.21), "Noida": (28.54, 77.39), "Patna": (25.59, 85.14),
    "Pune": (18.52, 73.86), "Raipur": (21.25, 81.63), "Ranchi": (23.34, 85.31),
    "Silchar": (24.83, 92.78), "Surat": (21.17, 72.83), "Trivandrum": (8.52, 76.94),
    "Vijayawada": (16.51, 80.65), "Vishakhapatnam": (17.69, 83.22), "Warangal": (17.97, 79.59),
}

AMENITY_LIST = ["Gym", "Pool", "Garden", "Clubhouse", "Playground"]

# ----------------------------------------------------------------------------
# Model feature lists (shared by training script and Streamlit app)
# ----------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rs",
    "Age_of_Property", "Floor_No", "Total_Floors", "Nearby_Schools",
    "Nearby_Hospitals", "Amenities_Count", "Infrastructure_Score",
    "School_Density_Score", "PPSF_vs_City_Median", "Price_vs_City_Median",
]
BINARY_FEATURES = [
    "Parking_Flag", "Security_Flag", "Ready_To_Move_Flag",
] + [f"Has_{a}" for a in AMENITY_LIST]
CATEGORICAL_FEATURES = [
    "City", "City_Tier", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Facing", "Owner_Type",
]
MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

CLASSIFICATION_TARGET = "Good_Investment"
REGRESSION_TARGET = "Future_Price_5Y_Lakhs"
