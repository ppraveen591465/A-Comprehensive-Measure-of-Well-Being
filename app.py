import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Well-Being AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------------------------------------------------
# 1. Data Loading & Smart Column Normalizer
# ---------------------------------------------------------
@st.cache_data
def generate_student_dataset(n_samples=600):
    np.random.seed(42)
    return pd.DataFrame({
        "Sleep_Hours": np.round(np.clip(np.random.normal(6.8, 1.3, n_samples), 3.5, 10.0), 1),
        "Study_Hours": np.round(np.clip(np.random.normal(5.2, 2.1, n_samples), 1.0, 12.0), 1),
        "Physical_Activity_Mins": np.round(np.clip(np.random.normal(45, 25, n_samples), 0, 120), 0),
        "Social_Support_Score": np.random.randint(1, 11, n_samples),
        "Academic_Pressure": np.random.randint(1, 11, n_samples),
        "Financial_Stress": np.random.randint(1, 11, n_samples),
        "Leisure_Screen_Time": np.round(np.clip(np.random.normal(5.5, 2.0, n_samples), 1.0, 12.0), 1),
    })

def standardize_uploaded_df(df):
    data = df.copy()
    # Replace spaces/hyphens with underscores and match lowercase names
    clean_cols = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in data.columns}
    activity_is_hours = "physical_activity_hours_per_day" in clean_cols.values()
    data.rename(columns=clean_cols, inplace=True)

    # Map common Kaggle student dataset names to our standard names
    alias_map = {
        "sleep_duration": "Sleep_Hours",
        "sleep_hours": "Sleep_Hours",
        "sleep_hours_per_day": "Sleep_Hours",
        "work/study_hours": "Study_Hours",
        "study_hours": "Study_Hours",
        "study_hours_per_day": "Study_Hours",
        "physical_activity": "Physical_Activity_Mins",
        "physical_activity_mins": "Physical_Activity_Mins",
        "physical_activity_hours_per_day": "Physical_Activity_Mins",
        "social_support": "Social_Support_Score",
        "social_support_score": "Social_Support_Score",
        "academic_pressure": "Academic_Pressure",
        "financial_stress": "Financial_Stress",
        "screen_time": "Leisure_Screen_Time",
        "leisure_screen_time": "Leisure_Screen_Time",
    }
    data.rename(columns=alias_map, inplace=True)

    # Convert text values (like "5-6 hours" or "Yes/No") to numbers & fill missing columns safely
    fallback_df = generate_student_dataset(len(data))
    for col in fallback_df.columns:
        if col not in data.columns:
            data[col] = fallback_df[col]
        else:
            # Extract numeric digits if the CSV column has strings like "7-8 hours"
            data[col] = pd.to_numeric(
                data[col].astype(str).str.extract(r"(\d+\.?\d*)")[0],
                errors="coerce"
            ).fillna(fallback_df[col])

    if activity_is_hours:
        data["Physical_Activity_Mins"] *= 60

    return data[fallback_df.columns]

# ---------------------------------------------------------
# 2. ML Pipeline (PCA Index + Random Forest + K-Means)
# ---------------------------------------------------------
@st.cache_resource
def train_wellbeing_models(df):
    data = df.copy()

    # Invert negative stressors so higher value = healthier well-being
    data["Inv_Academic_Pressure"] = 11 - data["Academic_Pressure"]
    data["Inv_Financial_Stress"] = 11 - data["Financial_Stress"]
    data["Inv_Screen_Time"] = np.clip(13 - data["Leisure_Screen_Time"], 1, 12)

    # Optimal study balance (penalize extreme overwork > 8 hrs)
    data["Balanced_Study"] = 10 - np.abs(data["Study_Hours"] - 5.5)

    pca_features = [
        "Sleep_Hours",
        "Physical_Activity_Mins",
        "Social_Support_Score",
        "Balanced_Study",
        "Inv_Academic_Pressure",
        "Inv_Financial_Stress",
        "Inv_Screen_Time",
    ]

    # Step A: Standardize & Construct Composite Well-Being Score (0-100) via PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[pca_features])

    pca = PCA(n_components=1, random_state=42)
    raw_index = pca.fit_transform(X_scaled)

    # Ensure positive correlation with Sleep & Social Support
    if np.corrcoef(raw_index.ravel(), data["Sleep_Hours"])[0, 1] < 0:
        raw_index = -raw_index

    minmax = MinMaxScaler(feature_range=(0, 100))
    data["WellBeing_Score"] = np.round(minmax.fit_transform(raw_index), 1)
    explained_var = pca.explained_variance_ratio_[0] * 100

    # Step B: Train Random Forest Regressor on Raw Student Inputs
    raw_features = [
        "Sleep_Hours",
        "Study_Hours",
        "Physical_Activity_Mins",
        "Social_Support_Score",
        "Academic_Pressure",
        "Financial_Stress",
        "Leisure_Screen_Time",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        data[raw_features], data["WellBeing_Score"], test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "pca_var": explained_var,
    }

    # Step C: Unsupervised K-Means Clustering (3 Student Personas)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    data["Cluster_ID"] = clusters
    metrics["silhouette"] = silhouette_score(X_scaled, clusters)

    # Order clusters dynamically from lowest mean Well-Being to highest
    cluster_means = data.groupby("Cluster_ID")["WellBeing_Score"].mean().sort_values()
    rank_map = {
        cluster_means.index[0]: "Could use more balance",
        cluster_means.index[1]: "Finding a balance",
        cluster_means.index[2]: "A balanced routine",
    }
    data["Persona"] = data["Cluster_ID"].map(rank_map)

    return data, rf_model, kmeans, scaler, minmax, pca, rank_map, raw_features, metrics

# ---------------------------------------------------------
# Data source and student check-in
# ---------------------------------------------------------
st.sidebar.title("Student wellbeing")
st.sidebar.caption("Use your own student data for group comparisons, or explore the example data.")

uploaded_file = st.sidebar.file_uploader(
    "Upload student data (CSV)",
    type=["csv"],
    help="Your file can include sleep, study, activity, or wellbeing-related columns.",
)
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = generate_student_dataset(600)

raw_df = standardize_uploaded_df(raw_df)
df, rf_model, kmeans, scaler, minmax, pca, rank_map, raw_features, metrics = train_wellbeing_models(raw_df)

# ---------------------------------------------------------
# Main page and simple daily check-in
# ---------------------------------------------------------
st.title("Student wellbeing check-in", icon=":material/self_improvement:")
st.write(
    "Share a few details about a usual day to get a simple wellbeing estimate "
    "and practical ideas you can try."
)
if uploaded_file is None:
    st.caption("Using example student data for group comparisons. Upload a CSV in the sidebar to use your own data.")
else:
    st.caption(f"Comparisons use the {len(df):,} rows in your uploaded file.")

with st.form("daily_habits_form"):
    st.subheader("Your usual day")
    st.caption("Choose the answers that feel closest to your typical day.")

    left_col, right_col = st.columns(2)
    with left_col:
        in_sleep = st.slider(
            "Sleep each night (hours)",
            3.5, 10.0, 7.0, 0.5,
            help="Think about your average sleep on a usual night.",
        )
        in_study = st.slider(
            "Study each day (hours)",
            1.0, 12.0, 5.5, 0.5,
        )
        in_activity = st.slider(
            "Physical activity each day (minutes)",
            0, 120, 45, 5,
            help="Include walking, sports, exercise, or other movement.",
        )
        in_social = st.slider(
            "Support from friends or family (1–10)",
            1, 10, 7,
            help="1 means very little support; 10 means a lot of support.",
        )
    with right_col:
        in_academic = st.slider(
            "Study-related pressure (1–10)",
            1, 10, 5,
            help="1 means very little pressure; 10 means a lot of pressure.",
        )
        in_financial = st.slider(
            "Money-related stress (1–10)",
            1, 10, 4,
            help="1 means very little stress; 10 means a lot of stress.",
        )
        in_screen = st.slider(
            "Leisure screen time each day (hours)",
            1.0, 12.0, 4.5, 0.5,
            help="Count screen time for fun, not time spent studying or working.",
        )
    st.form_submit_button("Show my summary", type="primary", width="stretch")

# Prepare single student input for prediction
user_input = pd.DataFrame([{
    "Sleep_Hours": in_sleep,
    "Study_Hours": in_study,
    "Physical_Activity_Mins": in_activity,
    "Social_Support_Score": in_social,
    "Academic_Pressure": in_academic,
    "Financial_Stress": in_financial,
    "Leisure_Screen_Time": in_screen,
}])

# Predict Well-Being Score via Random Forest
pred_score = float(np.clip(rf_model.predict(user_input)[0], 0, 100))

# Predict Cluster via K-Means
user_transformed = pd.DataFrame([{
    "Sleep_Hours": in_sleep,
    "Physical_Activity_Mins": in_activity,
    "Social_Support_Score": in_social,
    "Balanced_Study": 10 - abs(in_study - 5.5),
    "Inv_Academic_Pressure": 11 - in_academic,
    "Inv_Financial_Stress": 11 - in_financial,
    "Inv_Screen_Time": np.clip(13 - in_screen, 1, 12),
}])
user_scaled = scaler.transform(user_transformed)
pred_cluster_id = int(kmeans.predict(user_scaled)[0])
pred_persona = rank_map[pred_cluster_id]

results_tab, groups_tab, about_tab = st.tabs([
    "Your summary",
    "Student group trends",
    "About this estimate",
])

with results_tab:
    st.subheader("Your wellbeing summary", icon=":material/insights:")
    with st.container(border=True):
        score_col, summary_col = st.columns([1, 2], vertical_alignment="center")
        with score_col:
            st.metric(
                "Wellbeing score",
                f"{pred_score:.0f} / 100",
                help="A relative estimate based on the student data currently loaded.",
            )
            st.progress(int(round(pred_score)), text="Estimated wellbeing score")
        with summary_col:
            st.markdown("**Your current pattern**")
            st.subheader(pred_persona)
            st.caption(
                "This is a general guide for reflection, not a medical assessment."
            )

    st.subheader("Your habits compared with students")
    st.caption(
        "Each habit is shown on a 0–100 scale to make them easier to compare. "
        "Higher means a more supportive habit, not a health grade."
    )
    habit_scores = pd.DataFrame(
        {
            "Habit": [
                "Sleep",
                "Study balance",
                "Physical activity",
                "Social support",
                "Study pressure",
                "Money stress",
                "Leisure screen time",
            ],
            "You": [
                in_sleep / 10 * 100,
                max(0, 10 - abs(in_study - 5.5)) * 10,
                in_activity / 120 * 100,
                in_social / 10 * 100,
                (11 - in_academic) / 10 * 100,
                (11 - in_financial) / 10 * 100,
                np.clip(13 - in_screen, 1, 12) / 12 * 100,
            ],
            "Student average": [
                df["Sleep_Hours"].mean() / 10 * 100,
                (10 - (df["Study_Hours"] - 5.5).abs()).clip(lower=0).mean() * 10,
                df["Physical_Activity_Mins"].mean() / 120 * 100,
                df["Social_Support_Score"].mean() / 10 * 100,
                (11 - df["Academic_Pressure"]).mean() / 10 * 100,
                (11 - df["Financial_Stress"]).mean() / 10 * 100,
                np.clip(13 - df["Leisure_Screen_Time"], 1, 12).mean() / 12 * 100,
            ],
        }
    ).set_index("Habit")
    selected_habits = st.multiselect(
        "Choose habits to compare",
        options=habit_scores.index.tolist(),
        default=habit_scores.index.tolist(),
        help="Select one or more habits to keep the chart focused.",
    )
    radar_labels = habit_scores.index.tolist()
    radar_chart = {
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0},
        "radar": {
            "indicator": [{"name": label, "max": 100} for label in radar_labels],
            "radius": "62%",
            "axisName": {"color": "#183431"},
            "axisLine": {"lineStyle": {"color": "#D8E5E2"}},
            "splitLine": {"lineStyle": {"color": "#D8E5E2"}},
            "splitArea": {
                "areaStyle": {
                    "color": [
                        "rgba(15, 118, 110, 0.03)",
                        "rgba(15, 118, 110, 0.07)",
                    ]
                }
            },
        },
        "series": [
            {
                "type": "radar",
                "data": [
                    {
                        "value": [float(value) for value in habit_scores["You"]],
                        "name": "Your habits",
                        "itemStyle": {"color": "#0F766E"},
                        "lineStyle": {"color": "#0F766E", "width": 3},
                        "areaStyle": {"color": "#0F766E", "opacity": 0.22},
                    },
                    {
                        "value": [
                            float(value)
                            for value in habit_scores["Student average"]
                        ],
                        "name": "Student average",
                        "itemStyle": {"color": "#D59B54"},
                        "lineStyle": {
                            "color": "#D59B54",
                            "width": 3,
                            "type": "dashed",
                        },
                        "areaStyle": {"color": "#D59B54", "opacity": 0.18},
                    },
                ],
            }
        ],
    }
    st.echarts_chart(
        radar_chart,
        height=420,
        alt="Radar chart comparing your habits with the student average; each axis ranges from 0 to 100",
    )
    st.caption(
        "A wider shape means a higher score on that habit. The scores are for comparison, not a health grade."
    )

    st.subheader("Compare habits one by one")
    st.bar_chart(
        habit_scores.loc[selected_habits],
        horizontal=True,
        x_label="Habit balance (higher = more supportive)",
        alt="Your selected habits compared with the student average on a 0 to 100 scale",
    )

    st.subheader("A few ideas to try", icon=":material/lightbulb:")
    tips = []
    if in_sleep < 6.5:
        tips.append("Sleep: try a consistent bedtime and wake-up time, and make your room comfortable for rest.")
        tips.append("If possible, reduce caffeine late in the day and take a break from screens before bed.")
    if in_academic >= 7:
        tips.append("Study pressure: break big assignments into small steps and take short breaks between study sessions.")
        tips.append("If deadlines feel overwhelming, ask a teacher, tutor, or advisor which task to prioritize first.")
    if in_study > 8:
        tips.append("Long study days: protect time for meals, rest, and activities you enjoy.")
    if in_activity < 30:
        tips.append("Movement: start with a short walk or stretch break, then build up gradually with an activity you enjoy.")
    if in_social <= 4:
        tips.append("Connection: consider checking in with a friend, family member, or someone you trust.")
    if in_screen >= 7:
        tips.append("Screen breaks: pause between long sessions and try keeping your phone away during bedtime.")
    if in_financial >= 7:
        tips.append("Money stress: if available, ask your school about student support, financial aid, or budgeting resources.")
    tips.append("Everyday care: drink water regularly, eat regular meals when you can, and make time for something enjoyable.")
    tips.append("Stress support: try a few slow breaths or a brief quiet break; talk with someone you trust if stress persists.")

    for tip in tips:
        st.markdown(f"- {tip}")
    st.caption(
        "These are general wellbeing ideas, not medical advice. If you feel unwell or need support, "
        "consider speaking with a qualified health professional or your school support service."
    )

with groups_tab:
    st.subheader("Explore student groups")
    st.caption(
        "Choose a measure to compare the average for each student group."
    )
    group_measures = {
        "Wellbeing score (out of 100)": ("WellBeing_Score", "Average score"),
        "Sleep (hours per night)": ("Sleep_Hours", "Hours"),
        "Study time (hours per day)": ("Study_Hours", "Hours"),
        "Physical activity (minutes per day)": ("Physical_Activity_Mins", "Minutes"),
        "Social support (out of 10)": ("Social_Support_Score", "Average rating"),
    }
    selected_measure = st.selectbox(
        "Compare groups by",
        options=list(group_measures),
    )
    measure_column, axis_label = group_measures[selected_measure]
    group_scores = (
        df.groupby("Persona", as_index=False)[measure_column]
        .mean()
        .rename(columns={measure_column: "Average"})
        .round(1)
    )
    st.bar_chart(
        group_scores,
        x="Persona",
        y="Average",
        horizontal=True,
        x_label=axis_label,
        alt=f"{selected_measure} for each student group",
    )
    st.metric("Students in this dataset", f"{len(df):,}", border=True)

    st.subheader("Explore individual students")
    st.caption(
        "Each dot is one student. Drag across the chart to highlight a range "
        "of sleep and wellbeing scores."
    )
    student_range = alt.selection_interval(name="student_range", encodings=["x", "y"])
    scatter = (
        alt.Chart(df)
        .mark_circle(size=65)
        .encode(
            x=alt.X("Sleep_Hours:Q", title="Sleep per night (hours)"),
            y=alt.Y("WellBeing_Score:Q", title="Estimated wellbeing score"),
            color=alt.Color("Persona:N", title="Student group"),
            tooltip=[
                alt.Tooltip("Sleep_Hours:Q", title="Sleep (hours)", format=".1f"),
                alt.Tooltip("WellBeing_Score:Q", title="Wellbeing score", format=".1f"),
                alt.Tooltip("Persona:N", title="Student group"),
            ],
            opacity=alt.condition(student_range, alt.value(0.9), alt.value(0.2)),
        )
        .add_params(student_range)
        .properties(height=320)
    )
    selection_event = st.altair_chart(
        scatter,
        on_select="rerun",
        key="student_wellbeing_scatter",
        alt="Student sleep hours compared with estimated wellbeing score",
    )
    selected_range = selection_event.selection.get("student_range")
    if selected_range and "x" in selected_range and "y" in selected_range:
        sleep_min, sleep_max = selected_range["x"]
        score_min, score_max = selected_range["y"]
        selected_students = df[
            df["Sleep_Hours"].between(sleep_min, sleep_max)
            & df["WellBeing_Score"].between(score_min, score_max)
        ]
        st.caption(
            f"{len(selected_students):,} students in the highlighted area. "
            "Clear the selection by clicking an empty area of the chart."
        )
    else:
        st.caption(f"Showing all {len(df):,} students. Hover over a dot for details.")

with about_tab:
    st.subheader("What does this score mean?")
    st.write(
        "The estimate brings together sleep, study time, physical activity, "
        "social support, stress, and leisure screen time."
    )
    st.markdown(
        "- Scores are compared with the data currently loaded, so another dataset may give different results.\n"
        "- The suggestions are general ideas, not medical advice or a diagnosis.\n"
        "- If you are worried about your wellbeing, consider talking with someone you trust or a qualified professional."
    )