import streamlit as st
import joblib
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from PIL import Image
from folium.plugins import MeasureControl, MousePosition
from math import radians, sin, cos, sqrt, atan2
import os
import plotly.express as px

# إعداد الصفحة
st.set_page_config(page_title="لوحة المعلومات العقارية", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <h1 style='text-align: center; font-size: 4rem; margin-top: 0;'> لوحة المعلومات العقارية 🏠</h1>
""", unsafe_allow_html=True)

st.markdown("""

<style>

html, body, [data-testid="stAppViewContainer"] {
    direction: rtl;
    text-align: right;
}

h2, h3, h4, h5, h6{
    text-align: right;
    font-size:2rem !important;
}

section[data-testid="stSidebar"] {
    direction: rtl;
    text-align: right;
}

.stNumberInput input {
    font-size: 1.6rem !important;
}



[data-testid="stForm"] label {
    font-size: 2rem !important;
    font-weight: bold !important;
    display: block;
    
    text-align: right;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-size: 2rem !important;
    font-weight: bold !important;
    text-align: right;
      
}
/* Selectbox height */
div[data-baseweb="select"] {
    min-height: 70px !important;
}

/* Selected value area (visible text when closed) */
div[data-baseweb="select"] > div {
    min-height: 70px !important;
    display: flex;
    align-items: center !important;
    font-size: 1.8rem !important;
}

/* Fix font size for selected item text */
div[data-baseweb="select"] div[role="combobox"] {
    font-size: 1.8rem !important;
}

/* Font size for dropdown menu options */
div[data-baseweb="menu"] div[role="option"] {
    font-size: 2rem !important;
}


</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div.stForm button {
    font-size: 2.4rem !important;
    font-weight: bold !important;
    background-color:#c0c0c0 !important;
    color: black !important;
    border-radius: 8px !important;
    padding: 0.4em 1.2em !important;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Apply font size to all elements inside the form */
div[data-testid="stForm"] * {
    font-size: 1.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# =======================
# MODEL LOADING
# =======================
@st.cache_resource
def load_model():
    return joblib.load("XGBM_DB_last.joblib")

@st.cache_resource
def load_model_columns():
    return joblib.load("xgb_model_columns_DB.pkl")

model = load_model()
model_columns = load_model_columns()

# =======================
# FIXED PREDICT FUNCTION
# =======================
def predict_price(new_record):
    df = pd.DataFrame([new_record])

    # Ensure all expected columns exist
    df = pd.get_dummies(df)

    # ✅ Reinsert coordinates as float
    for coord in ['location.lat', 'location.lng']:
        if coord not in df.columns:
            df[coord] = float(new_record.get(coord, 0))

    # ✅ Ensure all model columns are present
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Align and ensure numeric dtype
    df = df[model_columns].astype(float)

    # Predict
    log_price = model.predict(df)[0]
    return np.expm1(log_price)

# =======================
# STREAMLIT UI (Fix for Coordinates)
# =======================
if map_data.get('last_clicked'):
    st.session_state['location_lat'] = float(map_data['last_clicked']['lat'])
    st.session_state['location_lng'] = float(map_data['last_clicked']['lng'])
    st.session_state['location_manually_set'] = True

# When submitting
if submitted:
    with st.spinner('جاري الحساب...'):
        input_data = {
            'livings': livings,
            'area': area,
            'street_width': street_width,
            'age': age,
            'street_direction': street_direction,
            'location.lat': float(st.session_state['location_lat']),
            'location.lng': float(st.session_state['location_lng']),
            'district': district
        }

        # Debug check
        st.write("🔍 Coordinates used:", input_data['location.lat'], input_data['location.lng'])

        price = predict_price(input_data)
        st.success("✅ تمت عملية التوقع بنجاح!")
        st.metric("السعر التقريبي", f"ريال {price:,.2f}")


st.markdown("<h1 style='font-size:2.4rem;'>📊 الرؤى واتجاهات السوق العقاري</h1>", unsafe_allow_html=True)

# --- 📊 Feature Importance Section ---
FEATURE_IMPORTANCE_FILE = "feature importance.csv"  

@st.cache_data
def load_feature_importance_data():
    """Loads feature importance data from CSV."""
    if not os.path.exists(FEATURE_IMPORTANCE_FILE):
        st.error(f"⚠️ Missing file: {FEATURE_IMPORTANCE_FILE}")
        return None

    try:
        df = pd.read_csv(FEATURE_IMPORTANCE_FILE)

        # ✅ Check column names to avoid KeyError
        expected_columns = {"الخاصية", "تأثيرها على السعر"}
        if not expected_columns.issubset(df.columns):
            missing_cols = expected_columns - set(df.columns)
            st.error(f"⚠️ CSV file is missing required columns: {missing_cols}")
            return None

        return df

    except Exception as e:
        st.error(f"⚠️ Error reading {FEATURE_IMPORTANCE_FILE}: {e}")
        return None


df_features = load_feature_importance_data()
col3, col4, col5 = st.columns([1, 1, 1])


with col3:
    st.subheader("📊 تأثير الخصائص على السعر")
    if df_features is not None and all(col in df_features.columns for col in ["الخاصية", "تأثيرها على السعر"]):
  
        fig_features = px.bar(
            df_features,
            x="تأثيرها على السعر",
            y="الخاصية",
            orientation="h",
            color="تأثيرها على السعر",
            height=400  # تقليل الارتفاع
        )
        fig_features.update_layout(
            margin=dict(l=100, r=20, t=40, b=40),  # ضبط الهوامش
            yaxis=dict(
                tickfont=dict(size=14),
                title=dict(text="الخاصية", standoff=60, font=dict(size=20))
            ),
            xaxis=dict(
                title=dict(text="تأثيرها على السعر", font=dict(size=20))
            )
        )

        st.plotly_chart(fig_features, use_container_width=True)
    else:
        st.error("تحقق من أسماء الأعمدة: 'الخاصية' و 'تأثيرها على السعر' غير موجودة في df_features")


    
# File paths for CSV files
DEALS_FILES = {
    "2022": "selected2022_a.csv",
    "2023": "selected2023_a.csv",
    "2024": "selected2024_a.csv"
}
TOTAL_COST_FILE = "deals_total.csv"

# ✅ Load & Transform "Total Cost of Deals" CSV
@st.cache_data
def load_total_cost_data():
    if os.path.exists(TOTAL_COST_FILE):
        try:
            df = pd.read_csv(TOTAL_COST_FILE)
            first_col = df.columns[0]
            df = df.melt(id_vars=[first_col], var_name="Year", value_name="Total Cost")
            df.rename(columns={first_col: "District"}, inplace=True)
            df["Year"] = df["Year"].astype(int)
            return df
        except Exception as e:
            st.error(f"⚠️ Error reading {TOTAL_COST_FILE}: {e}")
            return None
    else:
        st.warning(f"⚠️ Missing file: {TOTAL_COST_FILE}")
        return None

# ✅ Load & Transform "Number of Deals" Data from Multiple CSV Files
@st.cache_data
def load_deals_data():
    dataframes = []
    for year, file in DEALS_FILES.items():
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                df["Year"] = int(year)
                dataframes.append(df)
            except Exception as e:
                st.error(f"⚠️ Error reading {file}: {e}")
        else:
            st.warning(f"⚠️ Missing file: {file}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else None

# ✅ Load Data
df_deals = load_deals_data()
df_cost = load_total_cost_data()

if df_deals is not None and df_cost is not None:
   

    # ✅ Sidebar Filters
    valid_years = [year for year in sorted(df_deals["Year"].unique()) if year in [2022, 2023, 2024]]
    selected_year = st.sidebar.selectbox("📅 Select Year", ["All"] + valid_years)
    sort_by = st.sidebar.radio("📊 Sort By", ["Deal Count", "Total Cost"])

    # ✅ Filter Data Based on Selected Year
    if selected_year != "All":
        df_deals_filtered = df_deals[df_deals["Year"] == int(selected_year)]
        df_cost_filtered = df_cost[df_cost["Year"] == int(selected_year)]
    else:
        df_deals_filtered = df_deals
        df_cost_filtered = df_cost

   

with col4:
    st.subheader("📊 عدد الصفقات حسب الحي")
    
    # تجميع عدد الصفقات حسب الحي
    deals_per_district = df_deals_filtered.groupby(["District"])["Deal Count"].sum().reset_index()
    deals_per_district = deals_per_district.sort_values(by="Deal Count", ascending=False)

    # رسم المخطط
    fig_deals = px.bar(
        df_deals_filtered,
        x="District",
        y="Deal Count",
        color="Year",
        category_orders={"District": deals_per_district["District"].tolist()},
        height=400  # تقليل الارتفاع لتناسق العرض
    )

    # تنسيق الرسم البياني
    fig_deals.update_layout(
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis=dict(
            title=dict(
                text="الحي",standoff=70,
                font=dict(size=20)
            ),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=dict(
                text="عدد الصفقات",  # ✅ عنوان المحور Y بالعربية
                standoff=60,
                font=dict(size=20)
            ),
            tickfont=dict(size=14)
        ),
        coloraxis_colorbar=dict(
            title="السنة",  # ✅ تعريب شريط الألوان
            tickvals=[2022, 2023, 2024],
            ticktext=["2022", "2023", "2024"]
        )
    )

    # عرض المخطط في Streamlit
    st.plotly_chart(fig_deals, use_container_width=True)
   

with col5:
    st.subheader("💰 التكلفة الكلية للصفقات")

    if df_cost_filtered is not None:
        # تجميع التكلفة حسب الحي
        cost_per_district = df_cost_filtered.groupby(["District"])["Total Cost"].sum().reset_index()
        cost_per_district = cost_per_district.sort_values(by="Total Cost", ascending=False)

        # رسم المخطط
        fig_cost = px.bar(
            df_cost_filtered,
            x="District",
            y="Total Cost",
            color="Year",
            category_orders={"District": cost_per_district["District"].tolist()},
            height=400  # تقليل الارتفاع لتناسق العرض
        )

        # تنسيق الرسم البياني
        fig_cost.update_layout(
            margin=dict(l=60, r=20, t=40, b=40),
            xaxis=dict(
                title=dict(
                    text="الحي", standoff=70,
                    font=dict(size=20)
                ),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(
                    text="التكلفة الكلية",
                    standoff=60,
                    font=dict(size=20)
                ),
                tickfont=dict(size=14)
            ),
            coloraxis_colorbar=dict(
                title="السنة",
                tickvals=[2022, 2023, 2024],
                ticktext=["2022", "2023", "2024"]
            )
        )

        # عرض المخطط في Streamlit
        st.plotly_chart(fig_cost, use_container_width=True)
    
    else:
        st.error("❌ البيانات غير متوفرة. الرجاء التأكد من توفر الملفات في المسارات المحددة.")


# Footer
st.markdown("---")
