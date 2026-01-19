import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

# ==============================
# LOAD DATASETS
# ==============================
user_df = pd.read_csv("user_profiles.csv")
workout_df = pd.read_csv("workout_plan.csv")
food_df = pd.read_csv("food_dataset.csv")

# ==============================
# CALORIE PREDICTION MODEL
# ==============================
X = user_df.drop("daily_calories", axis=1)
y = user_df["daily_calories"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

calorie_model = RandomForestRegressor(n_estimators=200, random_state=42)
calorie_model.fit(X_train, y_train)

# ==============================
# WORKOUT MODEL
# ==============================
X_w = workout_df[["goal", "time_required", "difficulty"]]
y_w = workout_df["workout_label"]

workout_model = DecisionTreeClassifier(random_state=42)
workout_model.fit(X_w, y_w)

# ==============================
# FUNCTIONS
# ==============================
def recommend_food(calories, diet_type, budget):
    return food_df[
        (food_df["diet_type"] <= diet_type) &
        (food_df["cost_rs"] <= budget)
    ].sort_values(by="protein", ascending=False).head(3)

def predict_all(user_input):
    calories = calorie_model.predict([user_input])[0]
    workout_label = workout_model.predict([[user_input[5], user_input[8], 3]])[0]
    workout_name = workout_df[workout_df["workout_label"] == workout_label]["workout_type"].values[0]
    foods = recommend_food(calories, user_input[6], user_input[7])
    return calories, workout_name, foods

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="AI Fitness Planner", layout="centered")
st.title("🏋️ AI-Based Personalized Workout & Diet Planner")

age = st.slider("Age", 18, 30, 22)
gender = st.selectbox("Gender", ["Female", "Male"])
height = st.slider("Height (cm)", 150, 190, 170)
weight = st.slider("Weight (kg)", 45, 95, 65)
activity = st.slider("Activity Level (1–5)", 1, 5, 3)
goal = st.selectbox("Goal", ["Fat Loss", "Muscle Gain", "Maintain"])
diet = st.selectbox("Diet Type", ["Vegetarian", "Eggitarian", "Non-Vegetarian"])
budget = st.slider("Daily Food Budget (₹)", 100, 400, 200)
time = st.slider("Workout Time (minutes)", 15, 90, 60)

gender = 1 if gender == "Male" else 0
goal_map = {"Fat Loss": 0, "Muscle Gain": 1, "Maintain": 2}
diet_map = {"Vegetarian": 0, "Eggitarian": 1, "Non-Vegetarian": 2}

user_input = [
    age, gender, height, weight,
    activity, goal_map[goal],
    diet_map[diet], budget, time
]

# ==============================
# OUTPUT + VISUALS
# ==============================
if st.button("Generate Personalized Plan"):
    calories, workout, foods = predict_all(user_input)

    st.success(f"🔥 Daily Calories Required: **{int(calories)} kcal/day**")
    st.info(f"🏋️ Recommended Workout: **{workout}**")

    st.subheader("🥗 Recommended Foods")
    st.dataframe(foods.reset_index(drop=True))

    # ==============================
    # PIE CHART 1: CALORIE SPLIT
    # ==============================
    st.subheader("📊 Calorie Distribution (Macros)")
    protein_cal = foods["protein"].sum() * 4
    fat_cal = calories * 0.25
    carb_cal = calories - (protein_cal + fat_cal)

    fig1, ax1 = plt.subplots()
    ax1.pie(
        [protein_cal, carb_cal, fat_cal],
        labels=["Protein", "Carbohydrates", "Fats"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax1.axis("equal")
    st.pyplot(fig1)

    # ==============================
    # BAR CHART: PROTEIN PER FOOD
    # ==============================
    st.subheader("📈 Protein Content per Food Item")
    fig2, ax2 = plt.subplots()
    ax2.bar(foods["food_name"], foods["protein"])
    ax2.set_ylabel("Protein (g)")
    ax2.set_xlabel("Food Item")
    st.pyplot(fig2)

    # ==============================
    # PIE CHART 2: BUDGET USAGE
    # ==============================
    st.subheader("💰 Budget Utilization")
    used = foods["cost_rs"].sum()
    remaining = budget - used

    fig3, ax3 = plt.subplots()
    ax3.pie(
        [used, remaining],
        labels=["Used", "Remaining"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax3.axis("equal")
    st.pyplot(fig3)

    # ==============================
    # INTERPRETATION
    # ==============================
    st.subheader("🧠 Interpretation")
    st.write(
        f"""
        • Your body requires approximately **{int(calories)} kcal/day** to meet your goal.  
        • The diet focuses on **high-protein foods** for muscle recovery and fat control.  
        • The workout **{workout}** matches your available time and fitness goal.  
        • Your daily food spending stays **within your budget**.
        """
    )
