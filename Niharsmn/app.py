# ------------------------------
# Universal and Render-safe Streamlit app
# ------------------------------

import os
os.environ["PORT"] = os.getenv("PORT", "10000")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Streamlit page config
st.set_page_config(page_title="Airbnb Listings & Price Prediction", layout="wide")


# ----- Load model and feature metadata -----
@st.cache_data
def load_model():
    base_path = os.path.dirname(__file__)

    # Try to find files in current directory
    model_path = os.path.join(base_path, "model.pkl")
    feature_path = os.path.join(base_path, "feature_list.pkl")

    # If not found (Render deploys app in root), look inside Niharsmn/
    if not os.path.exists(model_path):
        model_path = os.path.join(base_path, "Niharsmn", "model.pkl")
    if not os.path.exists(feature_path):
        feature_path = os.path.join(base_path, "Niharsmn", "feature_list.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    if not os.path.exists(feature_path):
        st.error(f"Feature list file not found at {feature_path}")
        st.stop()

    model = joblib.load(model_path)
    feature_list = joblib.load(feature_path)
    return model, feature_list


# Load model + feature list once
model, feature_list = load_model()


# ----- Load dataset -----
base_path = os.path.dirname(__file__)
df_path = os.path.join(base_path, "new_york_listings_2024.csv")

if not os.path.exists(df_path):
    df_path = os.path.join(base_path, "Niharsmn", "new_york_listings_2024.csv")

if not os.path.exists(df_path):
    st.error("Dataset not found. Please ensure 'new_york_listings_2024.csv' is available.")
    st.stop()

df = pd.read_csv(df_path)


# Sidebar navigation
st.sidebar.title("🏠 Airbnb Project")
page = st.sidebar.radio("Select a section", [
    "Home / Dataset Overview",
    "Outliers in Price",
    "Price Distribution",
    "Minimum Nights Distribution",
    "Number of Reviews Distribution",
    "Availability 365 Distribution",
    "Correlation Heatmap",
    "Number of Reviews vs Price",
    "Room Type vs Average Price",
    "Price Dependency on Neighbourhood",
    "Geographical Distribution of Listings",
    "Pairplot Overview",
    "Predict Price"
])

# ------------------------------
# HOME / DATA OVERVIEW
# ------------------------------
if page == "Home / Dataset Overview":
    st.title("📊 Airbnb Dataset Overview")
    st.write("This section gives an overview of the dataset used for Airbnb price prediction.")

    if st.checkbox("Show first 10 rows of dataset"):
        st.dataframe(df.head(10))

    st.subheader("Basic Info")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    st.subheader("Column Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.write(df.describe())

# ------------------------------
# OUTLIERS
# ------------------------------
elif page == "Outliers in Price":
    st.title(" Identifying Outliers in Price")
    df_filtered = df[df['price'] < 1500]
    fig, ax = plt.subplots()
    sns.boxplot(data=df_filtered, x='price', ax=ax)
    ax.set_title("Boxplot - Price Outliers")
    st.pyplot(fig)
    st.info("Boxplot helps identify extreme price values that might be outliers in the dataset.")


# ------------------------------
# PRICE DISTRIBUTION
# ------------------------------
elif page == "Price Distribution":
    st.title(" Price Distribution")
    df_filtered = df[df['price'] < 1500]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df_filtered, x='price', bins=100, kde=False, ax=ax)
    ax.set_title("Distribution of Airbnb Prices")
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Price')
    st.pyplot(fig)
    st.info("This graph shows how Airbnb listing prices are distributed, with most prices falling in the lower range.")


# ------------------------------
# MINIMUM NIGHTS
# ------------------------------
elif page == "Minimum Nights Distribution":
    st.title("Minimum Nights Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["minimum_nights"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Minimum Nights")
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.info("Shows how many nights guests typically stay in Airbnb listings.")


# ------------------------------
# NUMBER OF REVIEWS
# ------------------------------
elif page == "Number of Reviews Distribution":
    st.title(" Number of Reviews Distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(data=df, x='number_of_reviews', bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Number of Reviews")
    st.pyplot(fig)
    st.info("Most listings receive a smaller number of reviews, while few have high review counts.")


# ------------------------------
# AVAILABILITY 365
# ------------------------------
elif page == "Availability 365 Distribution":
    st.title(" Availability 365 Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["availability_365"], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Availability Throughout the Year")
    st.pyplot(fig)
    st.info("Shows how many days per year listings are available for booking.")


# ------------------------------
# CORRELATION HEATMAP
# ------------------------------
elif page == "Correlation Heatmap":
    st.title(" Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.info("Displays correlation between numerical columns such as price, reviews, and availability.")


# ------------------------------
# REVIEWS VS PRICE
# ------------------------------
elif page == "Number of Reviews vs Price":
    st.title("Locality and Reviews Dependency")
    df_filtered = df[df['price'] < 1500]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_filtered, x="number_of_reviews", y="price", hue='neighbourhood_group', s=50)
    ax.set_title("Locality and Reviews Dependency")
    st.pyplot(fig)
    st.info("Shows whether listings with more reviews tend to have higher or lower prices.")


# ------------------------------
# ROOM TYPE VS PRICE
# ------------------------------
elif page == "Room Type vs Average Price":
    st.title(" Room Type vs Average Price")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="room_type", y="price", ax=ax)
    ax.set_title("Average Price by Room Type")
    st.pyplot(fig)
    st.info("Compares average Airbnb prices for different room types.")


# ------------------------------
# PRICE DEPENDENCY ON NEIGHBOURHOOD
# ------------------------------
elif page == "Price Dependency on Neighbourhood":
    st.title(" Price Dependency on Neighbourhood Group")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="neighbourhood_group", y="price", hue='room_type', ax=ax)
    ax.set_title("Average Price across Neighbourhood Groups")
    st.pyplot(fig)
    st.info("Shows how average prices vary between different neighbourhood groups.")


# ------------------------------
# GEOGRAPHICAL DISTRIBUTION
# ------------------------------
elif page == "Geographical Distribution of Listings":
    st.title(" Geographical Distribution of Airbnb Listings")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df, x="longitude", y="latitude", hue="room_type", alpha=0.6)
    ax.set_title("Geographical Distribution by Neighbourhood Group")
    st.pyplot(fig)
    st.info("Displays where Airbnb listings are located across different neighbourhoods of New York.")


# ------------------------------
# PAIRPLOT OVERVIEW
# ------------------------------
elif page == "Pairplot Overview":
    st.title("Pairplot Overview of Key Variables")
    st.info("This pairplot shows relationships among price, minimum nights, number of reviews, and availability — filtered to remove extreme price outliers for better readability.")
    df_filtered = df[df['price'] < 1500]
    fig = sns.pairplot(
        data=df_filtered,
        vars=['price', 'minimum_nights', 'number_of_reviews', 'availability_365'],
        hue='room_type'
    )
    st.pyplot(fig)


# ------------------------------
# PREDICT PRICE
# ------------------------------
elif page == "Predict Price":
    st.title("🏠 Predict Airbnb Listing Price")
    st.write("Use this form to predict the price of a listing based on its characteristics.")

    # Input fields
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=2)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
    reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=0.5)
    availability_365 = st.number_input("Availability (Days per Year)", min_value=0, max_value=365, value=150)
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Hotel room", "Private room", "Shared room"])
    neighbourhood_group = st.selectbox("Neighbourhood Group", ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"])

    # Create base input dict
    data = {
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'availability_365': availability_365,
        'room_type_Entire home/apt': 1 if room_type == "Entire home/apt" else 0,
        'room_type_Hotel room': 1 if room_type == "Hotel room" else 0,
        'room_type_Private room': 1 if room_type == "Private room" else 0,
        'room_type_Shared room': 1 if room_type == "Shared room" else 0,
        'neighbourhood_group_Bronx': 1 if neighbourhood_group == "Bronx" else 0,
        'neighbourhood_group_Brooklyn': 1 if neighbourhood_group == "Brooklyn" else 0,
        'neighbourhood_group_Manhattan': 1 if neighbourhood_group == "Manhattan" else 0,
        'neighbourhood_group_Queens': 1 if neighbourhood_group == "Queens" else 0,
        'neighbourhood_group_Staten Island': 1 if neighbourhood_group == "Staten Island" else 0
    }

    input_df = pd.DataFrame([data])

    # --- Key Section: Align columns with model’s expected features ---
    try:
        feature_list = model.feature_names_in_.tolist()

        for col in feature_list:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_list]

        if st.button(" Predict Price"):
            predicted_price = model.predict(input_df)[0]
            st.success(f"Predicted Airbnb Listing Price: **${predicted_price:.2f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

