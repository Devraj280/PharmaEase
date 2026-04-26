import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from collections import defaultdict

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaEase",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1117 0%, #1a1f2e 100%); border-right: 1px solid #2d3748; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { padding: 10px 14px; border-radius: 10px; margin: 2px 0; display: block; cursor: pointer; transition: background 0.2s; }
[data-testid="stSidebar"] .stRadio label:hover { background: #2d3748; }

[data-testid="metric-container"] { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 22px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
[data-testid="metric-container"] label { color: #64748b !important; font-size: 13px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #0f172a !important; font-size: 28px !important; font-weight: 600 !important; }

.stButton > button { background: #2563eb; color: white; border: none; border-radius: 10px; padding: 10px 24px; font-weight: 500; font-family: 'DM Sans', sans-serif; transition: all 0.2s; box-shadow: 0 2px 8px rgba(37,99,235,0.3); }
.stButton > button:hover { background: #1d4ed8; box-shadow: 0 4px 14px rgba(37,99,235,0.4); transform: translateY(-1px); }

[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; }

.stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div { border-radius: 10px !important; border: 1.5px solid #e2e8f0 !important; font-family: 'DM Sans', sans-serif !important; }
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important; }

.page-header { padding: 4px 0 16px 0; border-bottom: 1.5px solid #e2e8f0; margin-bottom: 20px; }
.page-header h1 { font-size: 24px; font-weight: 600; color: inherit; margin: 0 0 4px 0; line-height: 1.3; }
.page-header p  { color: #64748b; margin: 0; font-size: 14px; line-height: 1.5; }

.badge-mild     { background:#dcfce7; color:#166534; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-moderate { background:#fef3c7; color:#92400e; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-severe   { background:#fee2e2; color:#991b1b; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }

.disclaimer { background: #f1f5f9; border-left: 4px solid #94a3b8; padding: 12px 16px; border-radius: 0 10px 10px 0; color: #475569; font-size: 13px; margin-top: 20px; }
.remedy-pill   { display:inline-block; background:#dcfce7; color:#166534; border:1px solid #86efac; border-radius:20px; padding:4px 14px; font-size:13px; margin:3px; font-weight:500; }
.medicine-pill { display:inline-block; background:#dbeafe; color:#1e40af; border:1px solid #93c5fd; border-radius:20px; padding:4px 14px; font-size:13px; margin:3px; font-weight:500; }
.sym-pill      { display:inline-block; background:#f0fdf4; color:#166534; border:1px solid #86efac; border-radius:20px; padding:3px 12px; font-size:12px; margin:2px; }
.conf-bar-bg   { background:#e2e8f0; border-radius:6px; height:8px; margin:6px 0 12px 0; }
.conf-bar-fill { height:8px; border-radius:6px; background:linear-gradient(90deg,#2563eb,#10b981); }

.kpi-card { background:white; border-radius:16px; padding:22px 24px; box-shadow:0 2px 12px rgba(0,0,0,0.06); border:1px solid #e8ecf0; position:relative; overflow:hidden; }
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:4px; background:var(--accent,#2563eb); border-radius:16px 16px 0 0; }
.kpi-label { font-size:12px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px; }
.kpi-value { font-size:28px; font-weight:700; color:#0f172a; margin-bottom:4px; line-height:1; }
.kpi-sub   { font-size:12px; color:#64748b; }
.kpi-icon  { position:absolute; top:20px; right:20px; font-size:28px; opacity:0.15; }
.chart-card { background:white; border-radius:16px; padding:20px 20px 8px 20px; box-shadow:0 2px 12px rgba(0,0,0,0.06); border:1px solid #e8ecf0; margin-bottom:20px; }
.section-header { font-size:15px; font-weight:600; color:#0f172a; margin:0 0 2px 0; }
.section-sub    { font-size:12px; color:#94a3b8; margin:0 0 14px 0; }
.stTabs [data-baseweb="tab-list"] { background:#f1f5f9; border-radius:12px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; font-weight:500; font-size:13px; }
.stTabs [aria-selected="true"] { background:white !important; box-shadow:0 1px 4px rgba(0,0,0,0.1); }
</style>
""",
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

FILES = {
    "inventory": "inventory.csv",
    "prescriptions": "prescriptions.csv",
    "sales": "sales.csv",
    "employees": "employees.csv",
    "disease_info": "disease_info.csv",
}
PICKLE_FILES = {
    "model": "model.pkl",
    "label_encoder": "label_encoder.pkl",
    "symptom_list": "symptom_list.pkl",
    "disease_list": "disease_list.pkl",
}
COLORS = {
    "blue": "#2563eb",
    "teal": "#0d9488",
    "amber": "#d97706",
    "rose": "#e11d48",
    "purple": "#7c3aed",
    "green": "#16a34a",
    "slate": "#475569"
}
PALETTE = list(COLORS.values())

# ─────────────────────────────────────────────
#  SYMPTOM CATEGORIES
# ─────────────────────────────────────────────
SYMPTOM_CATEGORIES = {
    "🫁 Respiratory": [
        "shortness of breath", "cough", "wheezing", "chest tightness",
        "chest pain", "difficulty breathing", "rapid breathing",
        "nasal congestion", "runny nose", "sneezing", "sore throat",
        "hoarseness", "coughing up blood", "loss of voice", "stridor",
        "choking"
    ],
    "🧠 Neurological": [
        "headache", "dizziness", "fainting", "seizures", "tremors",
        "confusion", "memory loss", "numbness", "tingling", "weakness",
        "paralysis", "speech difficulty", "blurred vision", "double vision",
        "loss of consciousness", "abnormal involuntary movements",
        "unsteady gait", "poor balance", "insomnia"
    ],
    "❤️ Cardiovascular": [
        "palpitations", "rapid heart rate", "slow heart rate",
        "irregular heartbeat", "shortness of breath on exertion",
        "swelling of legs", "ankle swelling", "high blood pressure",
        "low blood pressure", "dizziness on standing", "cold extremities",
        "leg pain while walking"
    ],
    "🤢 Digestive": [
        "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain",
        "bloating", "indigestion", "heartburn", "loss of appetite",
        "difficulty swallowing", "blood in stool", "black tarry stool",
        "jaundice", "abdominal swelling", "excessive gas", "rectal bleeding",
        "weight loss"
    ],
    "🦴 Musculoskeletal": [
        "joint pain", "joint swelling", "muscle pain", "muscle weakness",
        "back pain", "neck pain", "stiffness", "bone pain",
        "limited range of motion", "swollen joints", "muscle cramps",
        "muscle spasm", "hip pain", "knee pain", "shoulder pain", "wrist pain",
        "ankle pain", "foot pain"
    ],
    "🌡️ General": [
        "fever", "chills", "fatigue", "weight loss", "weight gain",
        "night sweats", "loss of appetite", "general weakness", "malaise",
        "dehydration", "excessive thirst", "frequent urination",
        "excessive sweating", "swollen lymph nodes", "pallor"
    ],
    "🧴 Skin": [
        "rash", "itching", "hives", "skin redness", "dry skin", "peeling skin",
        "blisters", "bruising", "swelling", "hair loss", "nail changes",
        "acne", "skin discoloration", "jaundice", "cyanosis",
        "wound not healing"
    ],
    "👁️ Eyes / Ears": [
        "eye pain", "red eyes", "watery eyes", "eye discharge",
        "blurred vision", "ear pain", "ear discharge", "hearing loss",
        "ringing in ears", "tinnitus", "nasal discharge", "loss of smell",
        "nose bleeding", "facial pain", "sinus pressure"
    ],
    "🧬 Mental Health": [
        "anxiety and nervousness", "depression", "mood swings", "irritability",
        "difficulty concentrating", "hallucinations", "delusions",
        "panic attacks", "social withdrawal", "sleep problems",
        "excessive worry", "obsessive thoughts",
        "depressive or psychotic symptoms"
    ],
    "🚻 Urinary / Repro": [
        "burning urination", "blood in urine", "difficulty urinating",
        "pelvic pain", "vaginal discharge", "irregular periods",
        "painful periods", "erectile dysfunction", "testicular pain",
        "breast pain", "nipple discharge"
    ],
}


# ─────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────
def load_data(name):
    path = os.path.join(DATA_FOLDER, FILES[name])
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def save_data(df, name):
    df.to_csv(os.path.join(DATA_FOLDER, FILES[name]), index=False)


def display_df(df):
    d = df.reset_index(drop=True)
    d.index = d.index + 1
    d.index.name = "S.No"
    st.dataframe(d, use_container_width=True)
    return d


def delete_row(df, idx):
    return df.drop(df.index[idx]).reset_index(drop=True)


def page_header(title, subtitle=""):
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1>{"<p>"+subtitle+"</p>" if subtitle else ""}</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ML HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_ml_models():
    m = {}
    for key, fn in PICKLE_FILES.items():
        m[key] = pickle.load(open(fn, "rb")) if os.path.exists(fn) else None
    return m


@st.cache_resource
def load_cooccurrence():
    if os.path.exists("symptom_cooccurrence.pkl"):
        with open("symptom_cooccurrence.pkl", "rb") as f:
            return pickle.load(f)
    return None


def get_disease_info(name, df):
    if df.empty: return None
    row = df[df["disease"].str.lower() == name.lower()]
    return row.iloc[0].to_dict() if not row.empty else None


def sev_badge(s):
    s = str(s).lower().strip()
    cls = "badge-severe" if s == "severe" else (
        "badge-mild" if s == "mild" else "badge-moderate")
    return f'<span class="{cls}">● {s.capitalize()}</span>'


def run_prediction(selected, models):
    """Predict with rescaled relative confidence."""
    sl = models["symptom_list"]
    vec = [1 if s in selected else 0 for s in sl]
    proba = models["model"].predict_proba(pd.DataFrame([vec], columns=sl))[0]
    top5 = np.argsort(proba)[::-1][:5]
    tp = proba[top5]
    ts = tp.sum()
    rscl = (tp / ts * 100) if ts > 0 else tp * 100
    return [{
        "disease": models["label_encoder"].inverse_transform([top5[i]])[0],
        "confidence": round(float(rscl[i]), 1),
        "raw": round(float(tp[i]) * 100, 2)
    } for i in range(3)]


def get_suggestions(selected, cooc_data, n=8):
    """Top N co-occurring symptoms not already selected."""
    if not cooc_data or not selected: return []
    cooc = cooc_data.get("cooccurrence", {})
    freq = cooc_data.get("symptom_frequency", {})
    score = defaultdict(int)
    for s in selected:
        for co_s, cnt in cooc.get(s, []):
            if co_s not in selected: score[co_s] += cnt
    top = sorted(score.items(), key=lambda x: x[1], reverse=True)[:n]
    return [(s, freq.get(s, 0)) for s, _ in top]


def build_sorted_symptoms(symptom_list):
    """Return (sorted_list, category_map) with all 377 symptoms."""
    fl = {s.lower(): s for s in symptom_list}
    cmap, opts = {}, []
    for cat, kws in SYMPTOM_CATEGORIES.items():
        for s in sorted([fl[k.lower()] for k in kws if k.lower() in fl]):
            if s not in cmap:
                cmap[s] = cat
                opts.append(s)
    for s in sorted([s for s in symptom_list if s not in cmap]):
        cmap[s] = "🔬 Other"
        opts.append(s)
    return opts, cmap


# ─────────────────────────────────────────────
#  REPORTS HELPERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_dashboard_data():
    sales = load_data("sales")
    inv = load_data("inventory")
    emp = load_data("employees")
    rx = load_data("prescriptions")
    if not sales.empty and "Date" in sales.columns:
        sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
        sales["DayOfWeek"] = sales["Date"].dt.day_name()
        sales["Quarter"] = sales["Date"].dt.quarter.map({
            1: "Q1",
            2: "Q2",
            3: "Q3",
            4: "Q4"
        })
    if not rx.empty and "Date" in rx.columns:
        rx["Date"] = pd.to_datetime(rx["Date"], errors="coerce")
    if not inv.empty and "Expiry" in inv.columns:
        inv["Expiry"] = pd.to_datetime(inv["Expiry"], errors="coerce")
        inv["DaysToExpiry"] = (inv["Expiry"] - pd.Timestamp.today()).dt.days
        inv["ExpiryStatus"] = inv["DaysToExpiry"].apply(
            lambda x: "Expired" if x < 0 else ("Critical" if x < 30 else
                                               ("Soon" if x < 90 else "OK")))
        inv["StockValue"] = inv["Quantity"] * inv["Price"]
    if not emp.empty and "Joining Date" in emp.columns:
        emp["Joining Date"] = pd.to_datetime(emp["Joining Date"],
                                             errors="coerce")
        emp["YearsExp"] = (
            (pd.Timestamp.today() - emp["Joining Date"]).dt.days /
            365).round(1)
    return sales, inv, emp, rx


def cl(fig, h=300, leg=True):
    fig.update_layout(height=h,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans", size=12, color="#475569"),
                      showlegend=leg,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(showgrid=False, showline=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True,
                     gridcolor="#f1f5f9",
                     showline=False,
                     tickfont=dict(size=11))
    return fig


def kpi(col, label, val, sub, icon, color):
    col.markdown(
        f'<div class="kpi-card" style="--accent:{color}"><div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="padding:20px 10px 10px 10px;text-align:center;"><div style="font-size:36px;">💊</div><div style="font-size:20px;font-weight:600;color:#f1f5f9;margin:6px 0 2px 0;">PharmaEase</div><div style="font-size:11px;color:#94a3b8;letter-spacing:1px;">PHARMACY MANAGEMENT</div></div><hr style="border-color:#2d3748;margin:10px 0 20px 0;">',
    unsafe_allow_html=True)

menu = st.sidebar.radio("Nav", [
    "🏠  Dashboard",
    "📦  Inventory",
    "📄  Prescriptions",
    "💰  Sales & Billing",
    "👨‍💼  Employees",
    "🩺  Symptom Checker",
    "📊  Reports & Analytics",
],
                        label_visibility="collapsed")

st.sidebar.markdown(
    '<hr style="border-color:#2d3748;margin:20px 0 10px 0;"><div style="padding:0 10px;color:#64748b;font-size:11px;">PharmaEase v2.0 · Offline AI</div>',
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  🏠 DASHBOARD
# ═══════════════════════════════════════════════
if menu == "🏠  Dashboard":
    page_header("Dashboard", "Welcome back — here's your pharmacy at a glance")
    inv_df = load_data("inventory")
    sales_df = load_data("sales")
    rx_df = load_data("prescriptions")
    emp_df = load_data("employees")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Medicine SKUs", len(inv_df) if not inv_df.empty else 0)
    with c2:
        rev = sales_df["Amount"].sum(
        ) if not sales_df.empty and "Amount" in sales_df.columns else 0
        st.metric("Total Revenue", f"₹{rev:,.0f}")
    with c3:
        st.metric("Prescriptions", len(rx_df) if not rx_df.empty else 0)
    with c4:
        st.metric("Employees", len(emp_df) if not emp_df.empty else 0)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚠️ Low Stock Alert")
        if not inv_df.empty and "Quantity" in inv_df.columns:
            low = inv_df[inv_df["Quantity"] < 10]
            display_df(low[["Medicine Name", "Quantity"
                            ]].head(8)) if not low.empty else st.success(
                                "All medicines well stocked!")
        else:
            st.info("No inventory data.")
    with col2:
        st.markdown("#### 🕐 Recent Sales")
        display_df(sales_df.tail(
            8).iloc[::-1]) if not sales_df.empty else st.info("No sales yet.")

# ═══════════════════════════════════════════════
#  📦 INVENTORY
# ═══════════════════════════════════════════════
elif menu == "📦  Inventory":
    page_header("Inventory Management", "Track and manage your medicine stock")
    df = load_data("inventory")
    with st.expander("➕ Add New Medicine", expanded=False):
        with st.form("add_medicine"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                name = st.text_input("Medicine Name")
            with c2:
                category = st.text_input("Category")
            with c3:
                qty = st.number_input("Quantity", min_value=0, step=1)
            with c4:
                price = st.number_input("Price (₹)", min_value=0.0, step=0.5)
            expiry = st.date_input("Expiry Date")
            if st.form_submit_button("Add Medicine"):
                if name:
                    df = pd.concat([
                        df,
                        pd.DataFrame([{
                            "Medicine Name": name,
                            "Category": category,
                            "Quantity": qty,
                            "Price": price,
                            "Expiry": str(expiry)
                        }])
                    ],
                                   ignore_index=True)
                    save_data(df, "inventory")
                    st.success(f"✅ '{name}' added!")
                    st.rerun()
                else:
                    st.error("Medicine name is required.")
    st.markdown("#### 📋 Inventory List")
    if not df.empty:
        display_df(df)
        lc = len(df[df["Quantity"] < 10]) if "Quantity" in df.columns else 0
        if lc: st.warning(f"⚠️ {lc} medicine(s) below 10 units")
        del_no = st.selectbox("Select S.No to delete", range(1, len(df) + 1))
        if st.button("Delete Selected Medicine"):
            df = delete_row(df, del_no - 1)
            save_data(df, "inventory")
            st.success("Deleted.")
            st.rerun()
    else:
        st.info("No inventory data. Add medicines above.")

# ═══════════════════════════════════════════════
#  📄 PRESCRIPTIONS
# ═══════════════════════════════════════════════
elif menu == "📄  Prescriptions":
    page_header("Prescription Tracking",
                "Manage and retrieve patient prescriptions")
    df = load_data("prescriptions")
    with st.expander("➕ Add Prescription", expanded=False):
        with st.form("add_rx"):
            c1, c2 = st.columns(2)
            with c1:
                patient = st.text_input("Patient Name")
                doctor = st.text_input("Doctor Name")
            with c2:
                medicine = st.text_input("Medicine(s)")
                date = st.date_input("Date")
            notes = st.text_area("Notes / Dosage", height=80)
            if st.form_submit_button("Save Prescription"):
                if patient and medicine:
                    df = pd.concat([
                        df,
                        pd.DataFrame([{
                            "Patient": patient,
                            "Doctor": doctor,
                            "Medicine": medicine,
                            "Date": str(date),
                            "Notes": notes
                        }])
                    ],
                                   ignore_index=True)
                    save_data(df, "prescriptions")
                    st.success("✅ Saved!")
                    st.rerun()
                else:
                    st.error("Patient and medicine required.")
    st.markdown("#### 📋 Prescription Records")
    if not df.empty:
        search = st.text_input("🔍 Search by patient name", "")
        filtered = df[df["Patient"].str.contains(search, case=False,
                                                 na=False)] if search else df
        d = display_df(filtered)
        del_no = st.selectbox("Select S.No to delete", d.index)
        if st.button("Delete Selected Prescription"):
            df = delete_row(df, del_no - 1)
            save_data(df, "prescriptions")
            st.success("Deleted.")
            st.rerun()
    else:
        st.info("No prescriptions found.")

# ═══════════════════════════════════════════════
#  💰 SALES & BILLING
# ═══════════════════════════════════════════════
elif menu == "💰  Sales & Billing":
    page_header("Sales & Billing", "Generate bills and track transactions")
    sales_df = load_data("sales")
    inv_df = load_data("inventory")
    if not sales_df.empty and "Bill No" in sales_df.columns:
        bill_no = f"B{sales_df['Bill No'].astype(str).str.replace('B','',regex=False).astype(int).max()+1:03d}"
    else:
        bill_no = "B001"
    medicine_list = inv_df["Medicine Name"].tolist(
    ) if not inv_df.empty else []
    st.markdown("#### 🧾 Generate New Bill")
    with st.form("billing_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.text_input("Bill No", bill_no, disabled=True)
        with c2:
            customer = st.text_input("Customer Name")
        with c3:
            medicine = st.selectbox(
                "Medicine", medicine_list or ["No medicines in inventory"])
        with c4:
            quantity = st.number_input("Quantity", min_value=1, step=1)
        if st.form_submit_button("🧾 Generate Bill"):
            if not medicine_list: st.error("No medicines in inventory.")
            elif not customer: st.error("Customer name required.")
            else:
                stock = inv_df.loc[inv_df["Medicine Name"] == medicine,
                                   "Quantity"].values
                price = inv_df.loc[inv_df["Medicine Name"] == medicine,
                                   "Price"].values
                if len(stock) == 0: st.error("Medicine not found.")
                elif quantity > stock[0]:
                    st.error(f"❌ Only {stock[0]} units available.")
                else:
                    amount = quantity * price[0]
                    inv_df.loc[inv_df["Medicine Name"] == medicine,
                               "Quantity"] -= quantity
                    save_data(inv_df, "inventory")
                    sales_df = pd.concat([
                        sales_df,
                        pd.DataFrame([{
                            "Bill No": bill_no,
                            "Customer Name": customer,
                            "Medicine": medicine,
                            "Quantity": quantity,
                            "Amount": amount,
                            "Date": datetime.now().date()
                        }])
                    ],
                                         ignore_index=True)
                    save_data(sales_df, "sales")
                    st.success(f"✅ {bill_no} — ₹{amount:.2f}")
                    st.balloons()
                    st.rerun()
    if not sales_df.empty:
        st.metric(
            "Total Revenue", f"₹{sales_df['Amount'].sum():,.2f}"
            if "Amount" in sales_df.columns else "₹0")
        d = display_df(sales_df)
        del_no = st.selectbox("Select S.No to delete", d.index)
        if st.button("Delete Selected Bill"):
            sales_df = delete_row(sales_df, del_no - 1)
            save_data(sales_df, "sales")
            st.success("Deleted.")
            st.rerun()
    else:
        st.info("No sales yet.")

# ═══════════════════════════════════════════════
#  👨‍💼 EMPLOYEES
# ═══════════════════════════════════════════════
elif menu == "👨‍💼  Employees":
    page_header("Employee Management", "Manage pharmacy staff records")
    df = load_data("employees")
    with st.expander("➕ Add Employee", expanded=False):
        with st.form("add_emp"):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("Full Name")
                role = st.selectbox("Role", [
                    "Pharmacist", "Assistant", "Cashier", "Manager",
                    "Delivery", "Other"
                ])
            with c2:
                phone = st.text_input("Phone")
                email = st.text_input("Email")
            with c3:
                salary = st.number_input("Salary (₹)", min_value=0, step=500)
                join = st.date_input("Joining Date")
            if st.form_submit_button("Add Employee"):
                if name:
                    df = pd.concat([
                        df,
                        pd.DataFrame([{
                            "Name": name,
                            "Role": role,
                            "Phone": phone,
                            "Email": email,
                            "Salary": salary,
                            "Joining Date": str(join)
                        }])
                    ],
                                   ignore_index=True)
                    save_data(df, "employees")
                    st.success(f"✅ {name} added!")
                    st.rerun()
                else:
                    st.error("Name required.")
    if not df.empty:
        d = display_df(df)
        del_no = st.selectbox("Select S.No to delete", d.index)
        if st.button("Delete Selected Employee"):
            df = delete_row(df, del_no - 1)
            save_data(df, "employees")
            st.success("Removed.")
            st.rerun()
    else:
        st.info("No employees added yet.")

# ═══════════════════════════════════════════════
#  🩺 SYMPTOM CHECKER
# ═══════════════════════════════════════════════
elif menu == "🩺  Symptom Checker":
    page_header(
        "Symptom Checker",
        "Select symptoms to get disease prediction and medicine suggestions")

    models = load_ml_models()
    cooc_data = load_cooccurrence()
    disease_df = load_data("disease_info")
    inv_df = load_data("inventory")

    if models["model"] is None:
        st.error(
            "⚠️ Model files not found. Place model.pkl, label_encoder.pkl, "
            "symptom_list.pkl, disease_list.pkl in the project root.")
        st.stop()

    symptom_list = models["symptom_list"]
    sorted_opts, category_map = build_sorted_symptoms(symptom_list)

    # ── Initialize session state ──
    if "sc_selected" not in st.session_state:
        st.session_state["sc_selected"] = []
    if "sc_cat" not in st.session_state:
        st.session_state["sc_cat"] = None

    # ── Disclaimer ──
    st.warning("⚠️ **Medical Disclaimer:** For informational purposes only. "
               "Always consult a licensed doctor for diagnosis and treatment.")

    st.markdown("#### 🔍 Step 1 — Select Main Symptoms")
    st.caption("Filter by body system or type directly in the search box")

    # ── Category filter buttons ──
    cat_names = list(SYMPTOM_CATEGORIES.keys())
    btn_cols = st.columns(5)
    for i, cat in enumerate(cat_names):
        with btn_cols[i % 5]:
            active = st.session_state["sc_cat"] == cat
            if st.button(("✅ " if active else "") + cat,
                         key=f"cat_{i}",
                         use_container_width=True):
                st.session_state["sc_cat"] = None if active else cat
                st.rerun()

    sel_cat = st.session_state["sc_cat"]
    fl = {s.lower(): s for s in symptom_list}

    # Build display options
    if sel_cat:
        base_opts = sorted([
            fl[k.lower()] for k in SYMPTOM_CATEGORIES[sel_cat]
            if k.lower() in fl
        ])
        st.info(f"Showing **{len(base_opts)}** symptoms for {sel_cat}")
    else:
        base_opts = sorted_opts

    # Always include already-selected items so they don't vanish on category switch
    extra = [s for s in st.session_state["sc_selected"] if s not in base_opts]
    ms_opts = extra + base_opts

    # ── Multiselect — NO key=, NO on_change= ──
    # We read the return value directly; session state updated below
    selected = st.multiselect(
        "Symptoms",
        options=ms_opts,
        default=[s for s in st.session_state["sc_selected"] if s in ms_opts],
        placeholder="Type to search symptoms...",
        label_visibility="collapsed",
        format_func=lambda x: f"{category_map.get(x, '🔬')}  {x}",
    )

    # Sync back to session state from multiselect return value
    st.session_state["sc_selected"] = selected

    # ── Selected pills ──
    if selected:
        st.markdown(" ".join(f'<span class="sym-pill">{s}</span>'
                             for s in selected),
                    unsafe_allow_html=True)
        st.caption(f"✅ {len(selected)} symptom(s) selected")

    # ── Smart Co-occurrence Suggestions ──
    if selected:
        st.markdown("---")
        st.markdown("#### 💡 Step 2 — Add Related Symptoms")
        st.caption(
            "Symptoms most commonly found **together** with your selection in 246,945 patient records"
        )

        if cooc_data:
            cooc = cooc_data.get("cooccurrence", {})
            sym_freq = cooc_data.get("symptom_frequency", {})

            # IDF-weighted scoring — penalizes globally common symptoms
            raw_scores = defaultdict(float)
            for sel_sym in selected:
                if sel_sym in cooc:
                    for co_sym, count in cooc[sel_sym]:
                        if co_sym not in selected:
                            global_freq = sym_freq.get(co_sym, 1)
                            weight = count / (1 + global_freq / 1000)
                            raw_scores[co_sym] += weight

            top_suggestions = sorted(raw_scores.items(),
                                     key=lambda x: x[1],
                                     reverse=True)[:8]

            if top_suggestions:
                sug_cols = st.columns(4)
                for idx, (sym, _) in enumerate(top_suggestions):
                    freq = sym_freq.get(sym, 0)
                    short = sym[:24] + ("…" if len(sym) > 24 else "")
                    with sug_cols[idx % 4]:
                        if st.button(
                                f"+ {short}",
                                key=f"sug_{idx}",
                                use_container_width=True,
                                help=
                                f"'{sym}' co-occurs with your symptoms in {freq:,} patients"
                        ):
                            # ── KEY FIX: only update our own state key, never the widget key ──
                            if sym not in st.session_state["sc_selected"]:
                                st.session_state["sc_selected"].append(sym)
                            st.rerun()

                st.caption(
                    "💬 Hover any button to see patient frequency · Click to add instantly"
                )
            else:
                st.info(
                    "No specific related symptoms found — try adding more symptoms above."
                )

        else:
            # Fallback: suggest from same body system category
            st.caption(
                "💡 Showing category-based suggestions (run Colab co-occurrence cell for smarter results)"
            )
            from collections import Counter
            cats_selected = [category_map.get(s, "") for s in selected]
            dominant = Counter(cats_selected).most_common(1)
            if dominant:
                dom_cat = dominant[0][0]
                fallback = [
                    fl[k.lower()] for k in SYMPTOM_CATEGORIES.get(dom_cat, [])
                    if k.lower() in fl and fl[k.lower()] not in selected
                ][:8]
                if fallback:
                    fc = st.columns(4)
                    for idx, sym in enumerate(fallback):
                        short = sym[:24] + ("…" if len(sym) > 24 else "")
                        with fc[idx % 4]:
                            if st.button(f"+ {short}",
                                         key=f"fb_{idx}",
                                         use_container_width=True):
                                if sym not in st.session_state["sc_selected"]:
                                    st.session_state["sc_selected"].append(sym)
                                st.rerun()

    # ── Action buttons ──
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, _ = st.columns([2, 1, 4])
    with b1:
        analyze_btn = st.button("🔬 Analyze Symptoms",
                                use_container_width=True,
                                disabled=len(selected) == 0)
    with b2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["sc_selected"] = []
            st.session_state["sc_cat"] = None
            st.rerun()

    if 0 < len(selected) < 4:
        st.warning(
            "💡 **Tip:** Select at least 4–6 symptoms for better accuracy. "
            "Use the **Related Symptoms** buttons above!")

    # ── Prediction ──
    if analyze_btn and selected:
        with st.spinner("Analyzing symptoms..."):
            predictions = run_prediction(selected, models)

        if not predictions:
            st.error("Could not generate predictions. Check your model files.")
        else:
            st.markdown("---")
            st.markdown("### 🔮 Prediction Results")
            st.caption(
                "Confidence is relative to top predictions · More symptoms = higher accuracy"
            )

            sev_icon = {"mild": "🟢", "moderate": "🟡", "severe": "🔴"}

            for i, pred in enumerate(predictions):
                dname = pred["disease"]
                conf = pred["confidence"]
                raw = pred["raw"]
                info = get_disease_info(dname, disease_df)
                rank = ["🥇 Most Likely", "🥈 Possible", "🥉 Alternative"][i]

                st.markdown(f"**{rank} — {dname.title()}**")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
                    <div style="font-size:16px;font-weight:600;color:#0f172a;
                                min-width:260px;">{dname.title()}</div>
                    <div style="font-size:14px;color:#2563eb;
                                font-weight:600;">{conf:.1f}%</div>
                    <div style="font-size:11px;color:#94a3b8;">raw: {raw}%</div>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{max(conf,5)}%;"></div>
                </div>""",
                            unsafe_allow_html=True)

                if info:
                    sev = str(info.get("severity", "moderate")).lower().strip()
                    icon = sev_icon.get(sev, "🟡")
                    desc = info.get("description", "")
                    consult = info.get("consult_if", "")
                    rem = [
                        r.strip()
                        for r in str(info.get("home_remedies", "")).split("·")
                        if r.strip()
                    ]
                    meds = [
                        m.strip() for m in str(
                            info.get("medicine_category", "")).split("·")
                        if m.strip()
                    ]

                    with st.container(border=True):
                        h1, h2 = st.columns([4, 1])
                        with h1:
                            st.markdown(f"### {dname.title()}")
                        with h2:
                            st.markdown(f"**{icon} {sev.capitalize()}**")
                        st.markdown(desc)
                        st.divider()
                        rc, mc = st.columns(2)
                        with rc:
                            st.markdown("**🌿 Home Remedies**")
                            for r in rem:
                                st.markdown(f"- {r}")
                            if not rem: st.caption("No home remedies listed.")
                        with mc:
                            st.markdown("**💊 Suggested Medicine Categories**")
                            for m in meds:
                                st.markdown(f"- {m}")
                            if not meds:
                                st.caption("No medicine categories listed.")
                        if consult:
                            st.info(f"🩺 **Consult a doctor if:** {consult}")
                        if not inv_df.empty and "Medicine Name" in inv_df.columns and meds:
                            ml = [m.lower() for m in meds]
                            in_stock = [
                                f"**{r['Medicine Name']}** — "
                                f"{r.get('Quantity',0)} units @ ₹{r.get('Price',0)}"
                                for _, r in inv_df.iterrows()
                                if any(c in str(r["Medicine Name"]).lower()
                                       or str(r["Medicine Name"]).lower() in c
                                       for c in ml)
                            ]
                            if in_stock:
                                st.success(
                                    "✅ **Available in Your Inventory:**")
                                for item in in_stock:
                                    st.markdown(f"• {item}")
                else:
                    with st.container(border=True):
                        st.markdown(f"### {dname.title()}")
                        st.caption(
                            "Detailed info not available. Please consult a doctor."
                        )

                st.markdown("")

            st.info(
                "⚕️ **Important:** Suggestions are for *temporary relief only*. "
                "Visit a licensed physician for proper evaluation and treatment."
            )

# ═══════════════════════════════════════════════
#  📊 REPORTS & ANALYTICS
# ═══════════════════════════════════════════════
elif menu == "📊  Reports & Analytics":
    sales_df, inv_df, emp_df, rx_df = load_dashboard_data()
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);border-radius:20px;padding:24px 28px;margin-bottom:24px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <p style="font-size:20px;font-weight:700;color:white;margin:0;">📊 Reports & Analytics</p>
                <p style="font-size:13px;color:rgba(255,255,255,0.7);margin:4px 0 0 0;">Sales · Inventory · Employees · Prescriptions</p>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px;color:rgba(255,255,255,0.6);">Last updated</div>
                <div style="font-size:13px;color:white;font-weight:500;">{now}</div>
            </div>
        </div>
    </div>""",
                unsafe_allow_html=True)

    total_rev = sales_df["Amount"].sum(
    ) if not sales_df.empty and "Amount" in sales_df.columns else 0
    total_bills = len(sales_df) if not sales_df.empty else 0
    avg_bill = total_rev / total_bills if total_bills > 0 else 0
    low_stock = len(
        inv_df[inv_df["Quantity"] <
               10]) if not inv_df.empty and "Quantity" in inv_df.columns else 0
    expiring = len(
        inv_df[inv_df["DaysToExpiry"] < 90]
    ) if not inv_df.empty and "DaysToExpiry" in inv_df.columns else 0
    sal_bill = emp_df["Salary"].sum(
    ) if not emp_df.empty and "Salary" in emp_df.columns else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi(k1, "Total Revenue", f"₹{total_rev:,.0f}", f"{total_bills} bills", "💰",
        "#2563eb")
    kpi(k2, "Avg Bill", f"₹{avg_bill:,.0f}", "Per transaction", "🧾", "#0d9488")
    kpi(k3, "Medicine SKUs", f"{len(inv_df) if not inv_df.empty else 0}",
        f"{low_stock} low stock", "📦", "#d97706")
    kpi(k4, "Expiring Soon", f"{expiring}", "Items expiring <90d", "⏳",
        "#e11d48")
    kpi(k5, "Staff", f"{len(emp_df) if not emp_df.empty else 0}",
        f"₹{sal_bill:,.0f}/mo", "👥", "#7c3aed")
    kpi(k6, "Prescriptions", f"{len(rx_df) if not rx_df.empty else 0}",
        "Total records", "📄", "#16a34a")
    st.markdown("<br>", unsafe_allow_html=True)

    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly for full charts: `pip install plotly`")
        if not sales_df.empty and "Medicine" in sales_df.columns:
            st.markdown("#### Top Medicines")
            st.bar_chart(
                sales_df.groupby("Medicine")["Quantity"].sum().sort_values(
                    ascending=False).head(10))
        if not sales_df.empty and "Date" in sales_df.columns:
            st.markdown("#### Revenue Over Time")
            st.line_chart(sales_df.groupby("Date")["Amount"].sum())
        if not inv_df.empty and "Quantity" in inv_df.columns:
            st.markdown("#### Low Stock")
            display_df(inv_df[inv_df["Quantity"] < 10])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["💰  Sales", "📦  Inventory", "👥  Employees", "📄  Prescriptions"])

        with tab1:
            if sales_df.empty: st.info("No sales data.")
            else:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(
                        '<div class="chart-card"><p class="section-header">Revenue Over Time</p><p class="section-sub">Daily + 7-day moving average</p>',
                        unsafe_allow_html=True)
                    daily = sales_df.groupby("Date")["Amount"].sum(
                    ).reset_index().sort_values("Date")
                    daily["MA7"] = daily["Amount"].rolling(
                        7, min_periods=1).mean()
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=daily["Date"],
                            y=daily["Amount"],
                            fill="tozeroy",
                            fillcolor="rgba(37,99,235,0.08)",
                            line=dict(color=COLORS["blue"], width=2.5),
                            mode="lines",
                            name="Daily",
                            hovertemplate=
                            "<b>%{x|%d %b}</b><br>₹%{y:,.0f}<extra></extra>"))
                    fig.add_trace(
                        go.Scatter(x=daily["Date"],
                                   y=daily["MA7"],
                                   line=dict(color=COLORS["rose"],
                                             width=1.5,
                                             dash="dot"),
                                   mode="lines",
                                   name="7-day avg"))
                    cl(fig, 280)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        '<div class="chart-card"><p class="section-header">By Quarter</p>',
                        unsafe_allow_html=True)
                    qtr = sales_df.groupby(
                        "Quarter")["Amount"].sum().reset_index()
                    fig = px.pie(qtr,
                                 names="Quarter",
                                 values="Amount",
                                 color_discrete_sequence=PALETTE,
                                 hole=0.55)
                    fig.update_traces(textposition="outside",
                                      textinfo="label+percent",
                                      textfont_size=11)
                    cl(fig, 280, False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                c3, c4 = st.columns([2, 1])
                with c3:
                    st.markdown(
                        '<div class="chart-card"><p class="section-header">Top 10 Medicines by Revenue</p>',
                        unsafe_allow_html=True)
                    tm = sales_df.groupby("Medicine").agg(Revenue=(
                        "Amount",
                        "sum")).sort_values("Revenue").tail(10).reset_index()
                    fig = go.Figure(
                        go.Bar(x=tm["Revenue"],
                               y=tm["Medicine"],
                               orientation="h",
                               marker=dict(color=tm["Revenue"],
                                           colorscale=[[0, "#bfdbfe"],
                                                       [1, "#2563eb"]],
                                           showscale=False),
                               text=[f"₹{v:,.0f}" for v in tm["Revenue"]],
                               textposition="outside"))
                    cl(fig, 320, False)
                    fig.update_xaxes(visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(
                        '<div class="chart-card"><p class="section-header">Sales by Day</p>',
                        unsafe_allow_html=True)
                    day_order = [
                        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                        "Saturday", "Sunday"
                    ]
                    dow = sales_df.groupby("DayOfWeek")["Amount"].sum(
                    ).reindex(day_order).fillna(0).reset_index()
                    dow.columns = ["Day", "Revenue"]
                    fig = px.bar(dow,
                                 x="Day",
                                 y="Revenue",
                                 color="Revenue",
                                 color_continuous_scale=["#bfdbfe", "#2563eb"],
                                 text=[f"₹{v:,.0f}" for v in dow["Revenue"]])
                    fig.update_traces(textposition="outside", textfont_size=9)
                    fig.update_coloraxes(showscale=False)
                    cl(fig, 320, False)
                    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                c5, c6 = st.columns([1, 2])
                with c5:
                    st.markdown("**Top Customers**")
                    tc = sales_df.groupby("Customer Name")["Amount"].sum(
                    ).sort_values(ascending=False).head(8).reset_index()
                    tc.columns = ["Customer", "Spend"]
                    tc["Spend"] = tc["Spend"].map("₹{:,.0f}".format)
                    tc.index = range(1, len(tc) + 1)
                    st.dataframe(tc, use_container_width=True)
                with c6:
                    st.markdown("**Recent Transactions**")
                    rec = sales_df.sort_values("Date",
                                               ascending=False).head(10)[[
                                                   "Bill No", "Customer Name",
                                                   "Medicine", "Quantity",
                                                   "Amount", "Date"
                                               ]].copy()
                    rec["Amount"] = rec["Amount"].map("₹{:,.2f}".format)
                    rec["Date"] = rec["Date"].dt.strftime("%d %b %Y")
                    rec.index = range(1, len(rec) + 1)
                    st.dataframe(rec, use_container_width=True)

        with tab2:
            if inv_df.empty: st.info("No inventory data.")
            else:
                total_val = inv_df["StockValue"].sum(
                ) if "StockValue" in inv_df.columns else 0
                expired = len(inv_df[inv_df["DaysToExpiry"] < 0]
                              ) if "DaysToExpiry" in inv_df.columns else 0
                critical = len(inv_df[inv_df["DaysToExpiry"].between(
                    0, 29)]) if "DaysToExpiry" in inv_df.columns else 0
                ok = len(inv_df[inv_df["Quantity"] >=
                                10]) if "Quantity" in inv_df.columns else 0
                i1, i2, i3, i4 = st.columns(4)
                kpi(i1, "Inventory Value", f"₹{total_val:,.0f}", "Total value",
                    "💎", "#2563eb")
                kpi(i2, "Well Stocked", f"{ok}", "10+ units", "✅", "#16a34a")
                kpi(i3, "Low Stock", f"{low_stock}", "Below 10", "⚠️",
                    "#d97706")
                kpi(i4, "Expired", f"{expired}", f"{critical} expiring <30d",
                    "🚫", "#e11d48")
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    cv = inv_df.groupby("Category")["StockValue"].sum(
                    ).sort_values().reset_index()
                    fig = go.Figure(
                        go.Bar(x=cv["StockValue"],
                               y=cv["Category"],
                               orientation="h",
                               marker=dict(color=cv["StockValue"],
                                           colorscale=[[0, "#bbf7d0"],
                                                       [1, "#16a34a"]],
                                           showscale=False),
                               text=[f"₹{v:,.0f}" for v in cv["StockValue"]],
                               textposition="outside"))
                    cl(fig, 300, False)
                    fig.update_xaxes(visible=False)
                    st.markdown("**Stock Value by Category**")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    ec = inv_df["ExpiryStatus"].value_counts().reset_index()
                    ec.columns = ["Status", "Count"]
                    cmap = {
                        "OK": "#16a34a",
                        "Soon": "#d97706",
                        "Critical": "#e11d48",
                        "Expired": "#7c3aed"
                    }
                    fig = px.pie(ec,
                                 names="Status",
                                 values="Count",
                                 color="Status",
                                 color_discrete_map=cmap,
                                 hole=0.6)
                    fig.update_traces(textposition="outside",
                                      textinfo="label+value")
                    cl(fig, 300, False)
                    st.markdown("**Expiry Status**")
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("**⚠️ Alert — Low Stock & Expiry**")
                alert = inv_df[(inv_df["Quantity"] < 10) |
                               (inv_df["DaysToExpiry"] < 90)][[
                                   "Medicine Name", "Category", "Quantity",
                                   "ExpiryStatus", "DaysToExpiry"
                               ]].copy()
                if not alert.empty:
                    alert = alert.sort_values("DaysToExpiry")
                    alert["DaysToExpiry"] = alert["DaysToExpiry"].apply(
                        lambda x: "EXPIRED" if x < 0 else f"{int(x)} days")
                    alert.index = range(1, len(alert) + 1)
                    st.dataframe(alert, use_container_width=True)
                else:
                    st.success("✅ All inventory healthy!")

        with tab3:
            if emp_df.empty: st.info("No employee data.")
            else:
                avg_sal = emp_df["Salary"].mean(
                ) if "Salary" in emp_df.columns else 0
                avg_exp = emp_df["YearsExp"].mean(
                ) if "YearsExp" in emp_df.columns else 0
                e1, e2, e3, e4 = st.columns(4)
                kpi(e1, "Total Staff", f"{len(emp_df)}",
                    f"{emp_df['Role'].nunique()} roles", "👥", "#2563eb")
                kpi(e2, "Payroll", f"₹{sal_bill:,.0f}", "Monthly outflow", "💸",
                    "#e11d48")
                kpi(e3, "Avg Salary", f"₹{avg_sal:,.0f}",
                    f"Max ₹{emp_df['Salary'].max():,.0f}", "📈", "#0d9488")
                kpi(e4, "Avg Experience", f"{avg_exp:.1f} yrs", "Tenure", "🏅",
                    "#7c3aed")
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    rc = emp_df["Role"].value_counts().reset_index()
                    rc.columns = ["Role", "Count"]
                    fig = px.pie(rc,
                                 names="Role",
                                 values="Count",
                                 color_discrete_sequence=PALETTE,
                                 hole=0.5)
                    fig.update_traces(textposition="outside",
                                      textinfo="label+value")
                    cl(fig, 300, False)
                    st.markdown("**Staff by Role**")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    sr = emp_df.groupby(
                        "Role")["Salary"].mean().sort_values().reset_index()
                    fig = go.Figure(
                        go.Bar(y=sr["Role"],
                               x=sr["Salary"],
                               orientation="h",
                               marker_color=COLORS["purple"],
                               text=[f"₹{v:,.0f}" for v in sr["Salary"]],
                               textposition="outside"))
                    cl(fig, 300, False)
                    fig.update_xaxes(visible=False)
                    st.markdown("**Avg Salary by Role**")
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Full Staff Directory**")
                es = emp_df[[
                    "Name", "Role", "Phone", "Email", "Salary", "Joining Date",
                    "YearsExp"
                ]].copy()
                es["Salary"] = es["Salary"].map("₹{:,.0f}".format)
                es["Joining Date"] = es["Joining Date"].dt.strftime("%d %b %Y")
                es["YearsExp"] = es["YearsExp"].map("{:.1f} yrs".format)
                es.columns = [
                    "Name", "Role", "Phone", "Email", "Salary", "Joined",
                    "Experience"
                ]
                es.index = range(1, len(es) + 1)
                st.dataframe(es, use_container_width=True)

        with tab4:
            if rx_df.empty: st.info("No prescription data.")
            else:
                tp = rx_df["Doctor"].value_counts().index[0]
                tc = rx_df["Doctor"].value_counts().iloc[0]
                p1, p2, p3, p4 = st.columns(4)
                kpi(p1, "Total Rx", f"{len(rx_df)}", "All records", "📄",
                    "#2563eb")
                kpi(p2, "Unique Patients", f"{rx_df['Patient'].nunique()}",
                    "Distinct", "🧑‍⚕️", "#0d9488")
                kpi(p3, "Doctors", f"{rx_df['Doctor'].nunique()}", "Referring",
                    "👨‍⚕️", "#7c3aed")
                kpi(p4, "Top Doctor", tp.replace("Dr. ", ""),
                    f"{tc} prescriptions", "🏆", "#d97706")
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    dc = rx_df["Doctor"].value_counts().reset_index()
                    dc.columns = ["Doctor", "Count"]
                    fig = px.bar(dc,
                                 x="Doctor",
                                 y="Count",
                                 color="Count",
                                 color_continuous_scale=["#c7d2fe", "#7c3aed"],
                                 text="Count")
                    fig.update_traces(textposition="outside")
                    fig.update_coloraxes(showscale=False)
                    cl(fig, 300, False)
                    fig.update_xaxes(tickangle=-20)
                    st.markdown("**Prescriptions by Doctor**")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    rm = rx_df.groupby(
                        rx_df["Date"].dt.to_period("M")).size().reset_index()
                    rm.columns = ["Period", "Count"]
                    rm["PeriodStr"] = rm["Period"].dt.strftime("%b %Y")
                    fig = go.Figure(
                        go.Bar(x=rm["PeriodStr"],
                               y=rm["Count"],
                               marker_color=COLORS["teal"],
                               text=rm["Count"],
                               textposition="outside"))
                    cl(fig, 300, False)
                    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
                    st.markdown("**Prescriptions Over Time**")
                    st.plotly_chart(fig, use_container_width=True)
                c3, c4 = st.columns([1, 2])
                with c3:
                    tr = rx_df["Medicine"].value_counts().head(8).reset_index()
                    tr.columns = ["Medicine", "Count"]
                    fig = px.bar(tr.sort_values("Count"),
                                 x="Count",
                                 y="Medicine",
                                 orientation="h",
                                 color="Count",
                                 color_continuous_scale=["#bbf7d0", "#16a34a"],
                                 text="Count")
                    fig.update_traces(textposition="outside")
                    fig.update_coloraxes(showscale=False)
                    cl(fig, 300, False)
                    fig.update_xaxes(visible=False)
                    st.markdown("**Top Prescribed Medicines**")
                    st.plotly_chart(fig, use_container_width=True)
                with c4:
                    rs = rx_df[[
                        "Patient", "Doctor", "Medicine", "Date", "Notes"
                    ]].copy()
                    rs["Date"] = rs["Date"].dt.strftime("%d %b %Y")
                    rs.index = range(1, len(rs) + 1)
                    st.markdown("**Full Prescription Records**")
                    st.dataframe(rs, use_container_width=True, height=280)

    st.markdown(
        '<div style="text-align:center;padding:16px 0;color:#94a3b8;font-size:12px;">PharmaEase v2.0 · Streamlit + Plotly · 100% Offline</div>',
        unsafe_allow_html=True)
