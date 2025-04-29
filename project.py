# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Streamlit Config ---
st.set_page_config(page_title="Sleep Productivity Visual Analysis", layout="wide")

# --- Title + Tabs ---
st.title("💤 Sleep Patterns vs Productivity Across Professions")
tabs = st.tabs(["Introduction", "Dataset Analysis", "Filtered Data", "Correlation Heatmap", "Modeling", "Predict Your Productivity", "Conclusion"])

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

    Understanding these patterns can lead to the design of better work-life policies, promote healthier sleeping habits, and ultimately
    enhance overall performance in professional and personal spheres.

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
            fig = px.scatter(df, x=x_axis, y=y_axis, color=df['Profession'])
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
    st.header("🔍 View Filtered Data")
    professions = st.multiselect("Filter by Profession:", options=df['Profession'].unique(), default=df['Profession'].unique())
    sleep_range = st.slider("Sleep Duration Range (hrs):", 0.0, 12.0, (5.0, 9.0))
    filtered_df = df[(df['Profession'].isin(professions)) & (df['Sleep Duration (hrs)'].between(*sleep_range))]
    st.dataframe(filtered_df)

# --- Tab 4: Correlation Heatmap ---
with tabs[3]:
    st.header("📌 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(filtered_df[['Sleep Duration (hrs)', 'Productivity Level', 'Work Hours', 'Sleep Debt (hrs)']].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Scatter Plot: Work Hours vs Sleep Debt")
    scatter_fig = px.scatter(filtered_df, x='Work Hours', y='Sleep Debt (hrs)', color='Profession')
    st.plotly_chart(scatter_fig, use_container_width=True)

# --- Tab 5: Modeling ---
with tabs[4]:
    st.header("🤖 Model Training")
    X = filtered_df[['Sleep Duration (hrs)', 'Work Hours']]
    y = filtered_df['Productivity Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Choose Model:", ["Random Forest", "Linear Regression"])
    if model_choice == "Random Forest":
        model = RandomForestRegressor()
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    st.write(f"📉 Mean Squared Error: **{mse:.2f}**")
    st.write(f"📈 R2 Score: **{r2:.2f}**")

    if model_choice == "Random Forest":
        st.subheader("🔍 Feature Importance")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        st.bar_chart(imp_df.set_index("Feature"))

# --- Tab 6: Prediction ---
with tabs[5]:
    st.header("🔮 Predict Your Productivity")
    sleep_input = st.slider("Enter your Sleep Duration (hrs):", 0.0, 12.0, 7.0)
    work_input = st.slider("Enter your Work Hours per Day:", 0.0, 24.0, 8.0)
    input_df = pd.DataFrame({"Sleep Duration (hrs)": [sleep_input], "Work Hours": [work_input]})
    pred_output = model.predict(input_df)
    st.success(f"Estimated Productivity Level: **{pred_output[0]:.2f}** (scale 1-10)")

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
