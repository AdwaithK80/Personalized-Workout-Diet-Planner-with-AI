import pandas as pd
import numpy as np

# ==============================
# CONFIGURATION
# ==============================
np.random.seed(42)
SAMPLES = 3000

# ==============================
# DATASET 1: USER PROFILES
# ==============================
user_data = {
    "age": np.random.randint(18, 30, SAMPLES),
    "gender": np.random.randint(0, 2, SAMPLES),  # 0=Female, 1=Male
    "height_cm": np.random.randint(150, 190, SAMPLES),
    "weight_kg": np.random.randint(45, 95, SAMPLES),
    "activity_level": np.random.randint(1, 6, SAMPLES),  # 1=Low, 5=High
    "goal": np.random.randint(0, 3, SAMPLES),  # 0=Fat loss,1=Muscle,2=Maintain
    "diet_type": np.random.randint(0, 3, SAMPLES),  # 0=Veg,1=Egg,2=Nonveg
    "budget_rs": np.random.randint(100, 400, SAMPLES),
    "workout_time_min": np.random.randint(15, 90, SAMPLES)
}

user_df = pd.DataFrame(user_data)

# Calorie Calculation (BMR + Goal Adjustment)
user_df["daily_calories"] = (
    10 * user_df["weight_kg"]
    + 6.25 * user_df["height_cm"]
    - 5 * user_df["age"]
    + (5 * user_df["gender"])
    - (200 * user_df["goal"])
)

user_df.to_csv("user_profiles.csv", index=False)

# ==============================
# DATASET 2: WORKOUT PLANS
# ==============================
workout_data = {
    "workout_type": [
        "Cardio", "Strength Training", "HIIT",
        "Yoga", "Bodyweight", "Stretching"
    ],
    "goal": [0, 1, 0, 2, 1, 2],
    "time_required": [30, 45, 25, 40, 20, 15],
    "difficulty": [2, 4, 5, 2, 3, 1],
    "workout_label": [0, 1, 2, 3, 4, 5]
}

workout_df = pd.DataFrame(workout_data)
workout_df.to_csv("workout_plan.csv", index=False)

# ==============================
# DATASET 3: INDIAN FOOD DATASET
# ==============================
food_data = {
    "food_name": [
        "Rice", "Chapati", "Dal", "Curd",
        "Egg", "Paneer", "Chicken",
        "Fish", "Vegetable Curry", "Sprouts"
    ],
    "calories": [130, 120, 120, 98, 155, 265, 240, 200, 90, 150],
    "protein": [2.7, 3.5, 9, 11, 13, 18, 27, 22, 4, 12],
    "cost_rs": [10, 8, 15, 12, 8, 35, 40, 45, 10, 20],
    "diet_type": [
        0, 0, 0, 0,
        1, 0, 2,
        2, 0, 0
    ]  # 0=Veg,1=Egg,2=Nonveg
}

food_df = pd.DataFrame(food_data)
food_df.to_csv("food_dataset.csv", index=False)

# ==============================
# SUCCESS MESSAGE
# ==============================
print("✅ DATASETS GENERATED SUCCESSFULLY!")
print("Files created:")
print("• user_profiles.csv")
print("• workout_plan.csv")
print("• food_dataset.csv")
