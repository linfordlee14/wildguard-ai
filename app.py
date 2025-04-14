import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 🌟 Set up the page configuration
st.set_page_config(page_title="🦏 AI Wildlife Conservation Model", layout="wide")

# 🏆 App Title & Description
st.title("🦏 AI Wildlife Conservation Model")
st.markdown("This app analyzes rhino conservation data and predicts poaching risk. 🐘🔍")

# 📥 Load Data Function
@st.cache_data
def load_data():
    df = pd.read_csv("rhino_data.csv")
    df.columns = df.columns.str.strip()  # Remove extra spaces from column names ✅
    return df

df = load_data()

# 📝 Sidebar: Data Options & Visualization Selection
st.sidebar.header("🔍 Data Options")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("📄 Raw Data")
    st.dataframe(df)

st.sidebar.markdown("### 📊 Select Visualization")
vis_option = st.sidebar.selectbox("Choose a visualization", [
    "Distribution of Poaching Incidents",
    "Ranger Patrol Frequency vs Poaching Incidents",
    "Correlation Heatmap",
    "Tourist Activity vs Poaching Incidents"
])

# 🎨 Visualizations Section
st.subheader("📈 Visualization")
if vis_option == "Distribution of Poaching Incidents":
    fig, ax = plt.subplots()
    sns.countplot(x="PoachingIncidents", hue="PoachingIncidents", data=df, palette="Set2", legend=False, ax=ax)
    ax.set_title("Distribution of Poaching Incidents 🚨")
    st.pyplot(fig)

elif vis_option == "Ranger Patrol Frequency vs Poaching Incidents":
    fig, ax = plt.subplots()
    sns.boxplot(x="PoachingIncidents", y="RangerPatrolFreq", data=df, palette="coolwarm", ax=ax)
    ax.set_title("Ranger Patrol Frequency vs Poaching Incidents 👮‍♂️")
    st.pyplot(fig)

elif vis_option == "Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

elif vis_option == "Tourist Activity vs Poaching Incidents":
    fig, ax = plt.subplots()
    sns.scatterplot(x="TouristActivity", y="PoachingIncidents", data=df, hue="Zone", palette="viridis", ax=ax)
    ax.set_title("Tourist Activity vs Poaching Incidents by Zone 🏖️")
    st.pyplot(fig)

# 🤖 Prediction Section
st.subheader("🧠 Predict Poaching Risk")

# For demonstration, we use these three features.
# Make sure your CSV file has these columns!
features = ["RangerPatrolFreq", "TouristActivity", "CameraSurveillance"]

if not set(features).issubset(df.columns):
    st.error("❌ Required columns for prediction are not found in the dataset.")
else:
    X = df[features]
    y = df["PoachingIncidents"]

    # Split the data for training the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    st.markdown("### 🔢 Enter New Zone Details:")

    # Interactive input widgets with emoji fun!
    patrol_freq = st.slider("Ranger Patrol Frequency (visits/week) 🚓", int(df["RangerPatrolFreq"].min()), int(df["RangerPatrolFreq"].max()), int(df["RangerPatrolFreq"].mean()))
    tourist_activity = st.slider("Tourist Activity (number of visitors) 🏖️", int(df["TouristActivity"].min()), int(df["TouristActivity"].max()), int(df["TouristActivity"].mean()))
    camera = st.selectbox("Camera Surveillance (0 = No, 1 = Yes) 📷", options=[0, 1])

    # Prepare user input for prediction
    user_input = pd.DataFrame([[patrol_freq, tourist_activity, camera]], columns=features)
    prediction = model.predict(user_input)[0]
    
    st.success(f"✅ Predicted Poaching Incident (0 = No, 1 = Yes): **{prediction}**")
    
# 🚀 Footer
st.markdown("---")
st.caption("Developed by Linford Musiyambodza (Hacker1401) | AI for Wildlife Conservation 🌍")
st.caption("Source Code: [GitHub](https://github.com/linfordlee14/wildguard-ai)")
# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>©️ 2025 <strong>Linfy Tech Solutions</strong> | <em>Serving Africa and Beyond</em> 🌍</p>",
    unsafe_allow_html=True
)
# Add a footer with a link to the GitHub repository