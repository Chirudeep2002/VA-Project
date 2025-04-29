# Sleep Patterns vs Productivity Across Different Professions

## 📘 Project Overview
This Visual Analytics project explores the relationship between **sleep patterns** and **productivity levels** across various professions.  
Using public datasets from sleep trackers and productivity surveys, the project analyzes how sleep habits impact work efficiency, stress, and cognitive performance.

The web application is developed using **Streamlit**, allowing users to dynamically explore the data, generate custom visualizations, train simple models, and predict productivity scores based on sleep and work habits.

---

## 🧰 Technologies Used
- Python
- Pandas
- Seaborn
- Matplotlib
- Plotly
- Scikit-Learn
- Streamlit

---

## 📚 Dataset Information
The dataset combines sleep tracker data and productivity survey responses, including the following key columns:
- **Profession**  
- **Sleep Duration (hrs)**  
- **Productivity Level**  
- **Work Hours (hrs/day)**  
- **Sleep Debt (hrs)**

File: `sleep_productivity_dataset.csv`

---

## 🎯 Project Objectives
- Analyze how sleep duration correlates with productivity levels across different professions.
- Compare sleep patterns and identify professions with higher sleep debt risks.
- Examine the impact of extended work hours on sleep deprivation and performance.
- Provide actionable, data-driven recommendations for individuals and organizations.

---

## 📈 Expected Outcomes
- Identify the optimal sleep duration for peak productivity.
- Highlight high-risk professions affected by sleep deprivation.
- Demonstrate the negative impact of long work hours on productivity.
- Support wellness initiatives with concrete data.

---

## 🚀 Application Features
- **Interactive Dashboard** with multiple tabs:
  - Dataset Preview and Summary
  - Dynamic Visualizations (Bar, Line, Scatter, Histogram, Pie)
  - Filter Data by Profession and Sleep Range
  - Correlation Heatmap
  - Predict Productivity based on Sleep and Work Hours
  - Model Training (Random Forest or Linear Regression)
  - Conclusion and Takeaways

- **Dynamic Visual Options:**  
  Users can select their own x-axis and y-axis for custom visualizations.

- **Machine Learning Section:**  
  Train models to predict productivity, and analyze model performance (MSE, R²).

---

## 🛠️ How to Run the Project Locally
1. Clone or download the repository.
2. Install the required packages:
   ```bash
   pip install streamlit pandas matplotlib seaborn plotly scikit-learn
