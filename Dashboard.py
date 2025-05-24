import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Hydroponics Growth Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with top navigation bar
st.markdown(f"""
<style>
    /* Main app styling */
    .stApp {{
        background: linear-gradient(rgba(245, 247, 250, 0.7), rgba(228, 240, 242, 0.7)), 
                    url("https://media.istockphoto.com/id/591811738/photo/irrigation-system-in-function.jpg?s=612x612&w=0&k=20&c=jWKwu8Di5G958OYj2UDH14FfBM7AJ6MzroXcPiURHS8=");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
        color: black !important;
    }}
            
      </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    import os
    dataset_path = os.path.join(os.path.dirname(__file__), "hydroponics_system_dataset.csv")
    return pd.read_csv(dataset_path)

    

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
plant_types = df['plant_type'].unique()
selected_plants = st.sidebar.multiselect(
    "Select plant types", 
    plant_types, 
    default=plant_types
)

# Filter data based on selection
filtered_df = df[df['plant_type'].isin(selected_plants)]

# Main dashboard
st.title("🌱 Hydroponics Plant Growth Dashboard")
st.markdown("Analyzing plant growth patterns and optimal conditions in hydroponic systems")

# Overview metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered_df))
col2.metric("Unique Plant Types", filtered_df['plant_type'].nunique())
col3.metric("Average Growth Rate", f"{filtered_df['growth_rate'].mean():.2f}")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs([
    "Growth Analysis", 
    "Parameter Correlations", 
    "ML Models"
])

# Growth Analysis Tab
with tab1:
    st.header("Growth Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 📊 Boxplot showing the distribution of growth rate across plant types.
        # Helps identify variance, outliers, and how different plants respond to hydroponic conditions.
        st.subheader("Growth Rate Distribution by Plant Type")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='plant_type', y='growth_rate', data=filtered_df)
        plt.xticks(rotation=45)
        plt.xlabel('Plant Type')
        plt.ylabel('Growth Rate')
        st.pyplot(plt)

        # Explanation Section
        st.subheader("📝 Graph Explanation")
        
        st.markdown("""
        - **Box (rectangle)**: Represents the middle 50% of the plants' growth rate.
        - **Line inside box**: Represents the median growth rate.
        - **Whiskers (lines extending from box)**: Spread of the main data range.
        - **Circles (dots)**: Outliers — plants that grew much faster or slower.
        
        ### 🌟 Plant-wise Observations:
        - **Watercress**: High median growth (~0.95) and consistent performance.
        - **Tomato**: Slightly lower median, with some slow-growing outliers.
        - **Cilantro**: Wide variation; a few plants grew poorly (~0.4 growth rate).
        - **Kale, Mint, Chard**: Consistent but slightly variable growth.
        - **Spinach**: Lower median growth than Watercress.
        - **Basil, Lettuce, Arugula**: Highly consistent growers, ideal for stable yield.
        - **Strawberry, Cucumber**: Moderate but wider spread — room for optimization.
        
        ### 🚀 Insights:
        - **Best performers**: Watercress, Bell Pepper, Basil.
        - **Plants needing optimization**: Cilantro, Spinach, Mint (due to variability).
        - **Outliers**: Require checking nutrient, light, or pH conditions.
        
        """)
        
        # Conclusion
        st.subheader("📌 Conclusion")
        st.info("""
        - Plants like **Watercress** and **Basil** are recommended for quick, consistent hydroponic growth.
        - Attention needed for crops showing high variation (like **Cilantro** and **Spinach**).
        - Outliers should be monitored closely for environmental or nutrient issues.
        """)
    
with col2:
    # 📈 Line plot showing how growth rate changes over time per plant type.
    # Useful for identifying trends, cycles, or inconsistencies over time.
    st.subheader("Growth Rate Over Time")
    if 'timestamp' in filtered_df.columns:
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        filtered_df['date'] = filtered_df['timestamp'].dt.date
        daily_growth = filtered_df.groupby(['date', 'plant_type'])['growth_rate'].mean().unstack()
        plt.figure(figsize=(10, 6))
        for plant in daily_growth.columns:
            plt.plot(daily_growth.index, daily_growth[plant], label=plant)
        plt.xlabel('Date')
        plt.ylabel('Average Growth Rate')
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Timestamp data not available for temporal analysis")

    # Explanation
    st.subheader("📝 Graph Explanation")
    st.markdown("""
    - **X-axis**: Date range (May 2024 to May 2025).
    - **Y-axis**: Average plant growth rate.
    
    Each **colored line** ➔ represents **a plant type** showing its **growth trend over time**.
    
    ---
    
    ### 🚀 Key Observations:
    - Most plants maintain stable growth between **0.8 to 1.0**.
    - **Watercress**, **Basil**, and **Bell Pepper** are very stable.
    - **Cilantro** and **Strawberry** show periodic **growth dips**.
    - Occasional drops hint at possible environmental or nutrient issues.
    
    ---
    
    ### 📌 Conclusion:
    - Stable growers: **Watercress, Basil, Tomato** 🌱
    - Sensitive plants needing more care: **Cilantro, Strawberry** 🌿
    """)
    
    
  

# Parameter Correlations Tab
with tab2:
    st.header("Parameter Correlations")
    
    # 🧮 Heatmap showing correlation coefficients between numerical features.
    # Reveals which environmental variables are strongly linked to each other and to growth rate.
    st.subheader("Parameter Correlation Matrix")
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    st.pyplot(plt)
    
    # 🔍 Scatter plot of user-selected parameters.
    # Shows relationship strength and direction between any two variables.
    st.subheader("Parameter Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox(
            "X-axis parameter",
            numeric_cols,
            index=0
        )
    
    with col2:
        y_axis = st.selectbox(
            "Y-axis parameter",
            numeric_cols,
            index=1
        )
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_axis, y=y_axis, hue='plant_type', data=filtered_df)
    plt.title(f"{x_axis} vs {y_axis}")
    st.pyplot(plt)

    
    

# ... (previous imports remain the same)

# ML Models Tab
with tab3:
    st.header("Machine Learning Models")
    
    # Prepare data for ML
    ml_df = filtered_df.dropna(subset=['growth_rate'])
    X = ml_df[['temperature_c', 'humidity_percent', 'ph_level', 'ec_level', 
              'light_hours', 'co2_ppm', 'nitrogen_ppm', 'phosphorus_ppm', 
              'potassium_ppm']]
    y = ml_df['growth_rate']
    X = X.fillna(X.mean())  # Simple imputation
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model selection
    model_options = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }
    
    selected_model = st.selectbox(
        "Select a model to train and evaluate",
        list(model_options.keys())
    )
    
    if st.button("Train Model"):
        with st.spinner(f"Training {selected_model}..."):
            model = model_options[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Display metrics - UPDATED RMSE CALCULATION
            col1, col2 = st.columns(2)
            
            # Option 1: Calculate RMSE using numpy.sqrt
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            col1.metric("RMSE", f"{rmse:.4f}")
            
            # Option 2: Alternatively, you could use this more explicit version:
            # mse = mean_squared_error(y_test, y_pred)
            # rmse = np.sqrt(mse)
            # col1.metric("RMSE", f"{rmse:.4f}")
            
            col2.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importances = pd.Series(
                    model.feature_importances_, 
                    index=X.columns
                ).sort_values(ascending=False)
                
                plt.figure(figsize=(10, 6))
                importances.plot(kind='bar')
                plt.title('Feature Importance')
                plt.ylabel('Relative Importance')
                st.pyplot(plt)
            
            # Actual vs Predicted plot
            st.subheader("Actual vs Predicted Growth Rates")
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            plt.xlabel('Actual Growth Rate')
            plt.ylabel('Predicted Growth Rate')
            plt.title('Model Predictions')
            st.pyplot(plt)

# Footer
st.markdown("---")
st.markdown("Hydroponics Growth Dashboard - Powered by Streamlit") 