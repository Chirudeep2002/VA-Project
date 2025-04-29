# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Streamlit Config ---
st.set_page_config(page_title="Sleep Productivity Visual Analysis", layout="wide")

# --- Title + Tabs ---
st.title("💤 Sleep Patterns vs Productivity Across Professions")
tabs = st.tabs([
    "Introduction", 
    "Dataset Analysis", 
    "Filtered Data", 
    "Correlation Heatmap", 
    "Modeling", 
    "Predict Your Productivity", 
    "Conclusion"
])

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("sleep_productivity_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- Tab 1: Introduction ---
with tabs[0]:
    st.header("📘 Introduction")
    st.markdown("""
    Sleep plays a crucial role in maintaining cognitive functions, emotional well-being, and overall productivity.
    Especially in professional settings, variations in sleep duration can have significant impacts on an individual's efficiency,
    decision-making capabilities, and stress management.

    In this project, we aim to explore the relationship between sleep patterns and productivity levels across various professions.
    Using real-world sleep tracker datasets combined with productivity surveys, we uncover hidden trends and generate insights that
    could benefit individuals and organizations alike.

    ### 👥 Team Members
    - **Chirudeep Bandapalli**
    - **Sai Spandhana Billupati**
    - **Swethan Mandanapu**

    ### 🎯 Objective
    - Analyze how sleep duration correlates with productivity levels across different professions.
    - Compare sleep trends and identify professions with higher sleep debt risks.
    - Examine how extended work hours contribute to sleep deprivation and reduced efficiency.
    - Provide actionable insights that can guide wellness initiatives in workplaces.

    ### 📈 Expected Outcomes
    - Identify the optimal sleep duration for peak productivity.
    - Highlight professions that are most affected by sleep deprivation.
    - Demonstrate the impact of long work hours on sleep debt and performance.
    - Offer data-driven recommendations for individuals and organizations aiming to optimize productivity through better sleep habits.
    """)

# --- Tab 2: Dataset Analysis ---
with tabs[1]:
    st.header("📊 Dataset Overview")
    avg_sleep = df['Sleep Duration (hrs)'].mean()
    avg_productivity = df['Productivity Level'].mean()
    avg_workhours = df['Work Hours'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Sleep Duration (hrs)", f"{avg_sleep:.2f}")
    col2.metric("Average Productivity Level", f"{avg_productivity:.2f}")
    col3.metric("Average Work Hours", f"{avg_workhours:.2f}")

    st.dataframe(df.head())
    st.subheader("🔎 Summary Statistics")
    st.write(df.describe())

    st.subheader("📊 Choose Visualization")
    plot_type = st.selectbox("Select Plot Type:", ["Bar Plot", "Line Plot", "Scatter Plot", "Histogram", "Pie Chart"])

    if plot_type != "Pie Chart":
        x_axis = st.selectbox("Select X-axis:", df.columns)
        y_axis = st.selectbox("Select Y-axis:", df.columns)

    if st.button("Generate Plot"):
        if plot_type == "Bar Plot":
            fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis)
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Line Plot":
            fig = px.line(df, x=x_axis, y=y_axis, markers=True)
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color='Profession')
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Histogram":
            fig = px.histogram(df, x=x_axis)
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Pie Chart":
            pie_col = st.selectbox("Select Column for Pie Chart:", df.columns)
            fig = px.pie(df, names=pie_col)
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Filtered Data ---
with tabs[2]:
    st.header("🔍 View and Explore Filtered Data")
    professions = st.multiselect("Filter by Profession:", options=df['Profession'].unique(), default=df['Profession'].unique())
    sleep_range = st.slider("Sleep Duration Range (hrs):", 0.0, 12.0, (5.0, 9.0))
    filtered_df = df[(df['Profession'].isin(professions)) & (df['Sleep Duration (hrs)'].between(*sleep_range))]

    st.subheader("📄 Filtered Data Table")
    st.dataframe(filtered_df)

    st.subheader("📊 Summary Statistics of Filtered Data")
    st.write(filtered_df.describe())

    st.subheader("📈 Visualize Filtered Data")
    plot_type_filtered = st.selectbox("Select Plot Type (Filtered Data):", ["Bar Plot", "Line Plot", "Scatter Plot", "Histogram", "Pie Chart"])

    if plot_type_filtered != "Pie Chart":
        x_axis_filtered = st.selectbox("Select X-axis (Filtered Data):", filtered_df.columns)
        y_axis_filtered = st.selectbox("Select Y-axis (Filtered Data):", filtered_df.columns)

    if st.button("Generate Filtered Data Plot"):
        if plot_type_filtered == "Bar Plot":
            fig = px.bar(filtered_df, x=x_axis_filtered, y=y_axis_filtered, color=x_axis_filtered)
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type_filtered == "Line Plot":
            fig = px.line(filtered_df, x=x_axis_filtered, y=y_axis_filtered, markers=True)
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type_filtered == "Scatter Plot":
            fig = px.scatter(filtered_df, x=x_axis_filtered, y=y_axis_filtered, color='Profession')
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type_filtered == "Histogram":
            fig = px.histogram(filtered_df, x=x_axis_filtered)
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type_filtered == "Pie Chart":
            pie_col_filtered = st.selectbox("Select Column for Pie Chart:", filtered_df.columns)
            fig = px.pie(filtered_df, names=pie_col_filtered)
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Correlation Heatmap ---
with tabs[3]:
    st.header("📌 Correlation Heatmap Analysis")
    st.markdown("""
    The correlation heatmap below visualizes the strength and direction of relationships between key variables such as
    sleep duration, productivity level, work hours, and sleep debt.
    """)

    st.subheader("🎯 Set Correlation Threshold")
    threshold = st.slider("Select minimum correlation to display (absolute value):", 0.0, 1.0, 0.5, 0.1)

    corr_matrix = df[['Sleep Duration (hrs)', 'Productivity Level', 'Work Hours', 'Sleep Debt (hrs)']].corr()
    strong_corr = corr_matrix[(corr_matrix.abs() >= threshold)]

    st.subheader("📊 Correlation Heatmap (Filtered)")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(strong_corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📋 Correlation Values Table")
    st.dataframe(strong_corr)

# --- Tab 5: Modeling ---
with tabs[4]:
    st.header("🤖 Model Training")
    X = df[['Sleep Duration (hrs)', 'Work Hours']]
    y = df['Productivity Level']

    st.subheader("🎯 Set Train/Test Split Ratio")
    test_size = st.slider("Select Test Set Percentage:", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.subheader("⚙️ Choose a Machine Learning Model")
    model_choice = st.selectbox("Choose Model:", ["Random Forest", "Linear Regression"])
    if model_choice == "Random Forest":
        model = RandomForestRegressor()
    else:
        model = LinearRegression()

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        st.session_state.model = model  # Save model for prediction tab
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        st.success(f"✅ {model_choice} Model Trained Successfully!")

        st.subheader("📋 Model Summary")
        st.write(f"- **Training Set R² Score**: {model.score(X_train, y_train):.2f}")
        st.write(f"- **Testing Set R² Score**: {model.score(X_test, y_test):.2f}")

        st.subheader("📊 Model Performance Metrics")
        perf_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score'],
            'Value': [mse, rmse, mae, r2]
        })
        st.dataframe(perf_df)

# --- Tab 6: Prediction ---
with tabs[5]:
    st.header("🔮 Predict Your Productivity, Profession & Sleep Health")
    sleep_input = st.slider("Enter your Sleep Duration (hrs):", 0.0, 12.0, 7.0)
    work_input = st.slider("Enter your Work Hours per Day:", 0.0, 24.0, 8.0)
    input_df = pd.DataFrame({"Sleep Duration (hrs)": [sleep_input], "Work Hours": [work_input]})

    st.subheader("📈 Prediction Outputs")

    # Productivity Prediction
    if 'model' in st.session_state:
        pred_output = st.session_state.model.predict(input_df)
        st.success(f"Estimated Productivity Level: **{pred_output[0]:.2f}** (scale 1-10)")
    else:
        st.warning("⚠️ Please train a model first in the 'Modeling' tab to predict Productivity.")

    # Profession Prediction
    df_copy = df.copy()
    df_copy['Sleep_Diff'] = abs(df_copy['Sleep Duration (hrs)'] - sleep_input)
    df_copy['Work_Diff'] = abs(df_copy['Work Hours'] - work_input)
    df_copy['Total_Diff'] = df_copy['Sleep_Diff'] + df_copy['Work_Diff']
    predicted_profession = df_copy.sort_values('Total_Diff').iloc[0]['Profession']

    st.info(f"🧑‍💼 Based on your habits, you most closely match: **{predicted_profession}**")

    # Insomnia Prediction
    if sleep_input < 5:
        st.error("⚠️ Risk Alert: Based on your sleep duration, you might be suffering from **Insomnia** (Sleep < 5 hrs).")
    else:
        st.success("😴 Good Sleep Health: Your sleep duration looks healthy!")

# --- Tab 7: Conclusion ---
with tabs[6]:
    st.header("📌 Conclusion")
    st.markdown("""
    - **Sleep matters**: We found that 7-8 hours is the sweet spot for productivity.
    - **Workload affects sleep debt**: More hours → less sleep → lower productivity.
    - **Professions vary**: Doctors, tech workers face higher sleep debt.

    ### ✅ Takeaways
    - Balance sleep and work for optimal performance.
    - Companies can use this data for wellness strategies.
    - Individuals can plan routines around these insights.

    Thank you for viewing this project! 🙏
    """)

st.markdown("---")
st.caption("© 2025 Team: Chirudeep, Sai Spandhana, Swethan | Visual Analytics Project")
