import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Remove Streamlit default top padding ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #1a1f2e 100%);
    border-right: 1px solid #2d3748;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px;
    border-radius: 10px;
    margin: 2px 0;
    display: block;
    cursor: pointer;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #2d3748; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 22px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label { color: #64748b !important; font-size: 13px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3);
}
.stButton > button:hover {
    background: #1d4ed8;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4);
    transform: translateY(-1px);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}

/* ── Page header ── */
.page-header {
    padding: 4px 0 16px 0;
    border-bottom: 1.5px solid #e2e8f0;
    margin-bottom: 20px;
}
.page-header h1 {
    font-size: 24px;
    font-weight: 600;
    color: inherit;
    margin: 0 0 4px 0;
    line-height: 1.3;
}
.page-header p {
    color: #64748b;
    margin: 0;
    font-size: 14px;
    line-height: 1.5;
}

/* ── Severity badges ── */
.badge-mild     { background:#dcfce7; color:#166534; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-moderate { background:#fef3c7; color:#92400e; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-severe   { background:#fee2e2; color:#991b1b; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }

/* ── Disclaimer box ── */
.disclaimer {
    background: #f1f5f9;
    border-left: 4px solid #94a3b8;
    padding: 12px 16px;
    border-radius: 0 10px 10px 0;
    color: #475569;
    font-size: 13px;
    margin-top: 20px;
}
</style>
""",
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA PATHS
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


# ─────────────────────────────────────────────
#  HELPERS
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
    subtitle_html = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        {subtitle_html}
    </div>
    """,
                unsafe_allow_html=True)


@st.cache_resource
def load_ml_models():
    models = {}
    for key, filename in PICKLE_FILES.items():
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                models[key] = pickle.load(f)
        else:
            models[key] = None
    return models


def predict_disease(selected_symptoms, models, top_n=3):
    if models["model"] is None or models["symptom_list"] is None:
        return []
    symptom_list = models["symptom_list"]
    vector = [1 if s in selected_symptoms else 0 for s in symptom_list]
    input_df = pd.DataFrame([vector], columns=symptom_list)
    proba = models["model"].predict_proba(input_df)[0]
    top_idx = np.argsort(proba)[::-1][:top_n]
    results = []
    for idx in top_idx:
        disease = models["label_encoder"].inverse_transform([idx])[0]
        confidence = round(float(proba[idx]) * 100, 1)
        results.append({"disease": disease, "confidence": confidence})
    return results


def get_disease_info(disease_name, disease_df):
    if disease_df.empty:
        return None
    row = disease_df[disease_df["disease"].str.lower() == disease_name.lower()]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 20px 10px 10px 10px; text-align: center;">
    <div style="font-size: 36px;">💊</div>
    <div style="font-size: 20px; font-weight: 600; color: #f1f5f9; margin: 6px 0 2px 0;">PharmaEase</div>
    <div style="font-size: 11px; color: #94a3b8; letter-spacing: 1px;">PHARMACY MANAGEMENT</div>
</div>
<hr style="border-color: #2d3748; margin: 10px 0 20px 0;">
""",
                    unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠  Dashboard",
        "📦  Inventory",
        "📄  Prescriptions",
        "💰  Sales & Billing",
        "👨‍💼  Employees",
        "🩺  Symptom Checker",
        "📊  Reports",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("""
<hr style="border-color: #2d3748; margin: 20px 0 10px 0;">
<div style="padding: 0 10px; color: #64748b; font-size: 11px;">
    PharmaEase v2.0 · Offline AI
</div>
""",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────
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
        st.metric("Total Revenue (₹)", f"₹{rev:,.0f}")
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
            if not low.empty:
                display_df(low[["Medicine Name", "Quantity"]].head(8))
            else:
                st.success("All medicines are well stocked!")
        else:
            st.info("No inventory data.")

    with col2:
        st.markdown("#### 🕐 Recent Sales")
        if not sales_df.empty:
            display_df(sales_df.tail(8).iloc[::-1])
        else:
            st.info("No sales recorded yet.")

# ─────────────────────────────────────────────
#  INVENTORY
# ─────────────────────────────────────────────
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
                    new_row = {
                        "Medicine Name": name,
                        "Category": category,
                        "Quantity": qty,
                        "Price": price,
                        "Expiry": str(expiry),
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])],
                                   ignore_index=True)
                    save_data(df, "inventory")
                    st.success(f"✅ '{name}' added successfully!")
                    st.rerun()
                else:
                    st.error("Medicine name is required.")

    st.markdown("#### 📋 Inventory List")
    if not df.empty:
        df_display = display_df(df)
        low_count = len(
            df[df["Quantity"] < 10]) if "Quantity" in df.columns else 0
        if low_count:
            st.warning(
                f"⚠️ {low_count} medicine(s) have low stock (< 10 units)")

        st.markdown("#### 🗑️ Delete Medicine")
        del_no = st.selectbox("Select S.No to delete", df_display.index)
        if st.button("Delete Selected Medicine"):
            df = delete_row(df, del_no - 1)
            save_data(df, "inventory")
            st.success("Medicine deleted.")
            st.rerun()
    else:
        st.info("No inventory data. Add medicines above.")

# ─────────────────────────────────────────────
#  PRESCRIPTIONS
# ─────────────────────────────────────────────
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
                    new_row = {
                        "Patient": patient,
                        "Doctor": doctor,
                        "Medicine": medicine,
                        "Date": str(date),
                        "Notes": notes,
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])],
                                   ignore_index=True)
                    save_data(df, "prescriptions")
                    st.success("✅ Prescription saved!")
                    st.rerun()
                else:
                    st.error("Patient name and medicine are required.")

    st.markdown("#### 📋 Prescription Records")
    if not df.empty:
        search = st.text_input("🔍 Search by patient name", "")
        filtered = df[df["Patient"].str.contains(search, case=False,
                                                 na=False)] if search else df
        df_display = display_df(filtered)

        del_no = st.selectbox("Select S.No to delete", df_display.index)
        if st.button("Delete Selected Prescription"):
            df = delete_row(df, del_no - 1)
            save_data(df, "prescriptions")
            st.success("Prescription deleted.")
            st.rerun()
    else:
        st.info("No prescriptions found.")

# ─────────────────────────────────────────────
#  SALES & BILLING
# ─────────────────────────────────────────────
elif menu == "💰  Sales & Billing":
    page_header("Sales & Billing", "Generate bills and track transactions")

    sales_df = load_data("sales")
    inv_df = load_data("inventory")

    if not sales_df.empty and "Bill No" in sales_df.columns:
        nums = sales_df["Bill No"].astype(str).str.replace(
            "B", "", regex=False).astype(int)
        bill_no = f"B{nums.max() + 1:03d}"
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
                "Medicine", medicine_list
                if medicine_list else ["No medicines in inventory"])
        with c4:
            quantity = st.number_input("Quantity", min_value=1, step=1)

        if st.form_submit_button("🧾 Generate Bill"):
            if not medicine_list:
                st.error("No medicines in inventory.")
            elif customer:
                stock = inv_df.loc[inv_df["Medicine Name"] == medicine,
                                   "Quantity"].values
                price = inv_df.loc[inv_df["Medicine Name"] == medicine,
                                   "Price"].values
                if len(stock) == 0:
                    st.error("Medicine not found.")
                elif quantity > stock[0]:
                    st.error(
                        f"❌ Insufficient stock! Available: {stock[0]} units")
                else:
                    amount = quantity * price[0]
                    inv_df.loc[inv_df["Medicine Name"] == medicine,
                               "Quantity"] -= quantity
                    save_data(inv_df, "inventory")
                    new_row = {
                        "Bill No": bill_no,
                        "Customer Name": customer,
                        "Medicine": medicine,
                        "Quantity": quantity,
                        "Amount": amount,
                        "Date": datetime.now().date(),
                    }
                    sales_df = pd.concat(
                        [sales_df, pd.DataFrame([new_row])], ignore_index=True)
                    save_data(sales_df, "sales")
                    st.success(
                        f"✅ Bill {bill_no} generated! Amount: ₹{amount:.2f}")
                    st.balloons()
                    st.rerun()
            else:
                st.error("Customer name is required.")

    st.markdown("#### 📋 Sales History")
    if not sales_df.empty:
        total = sales_df["Amount"].sum() if "Amount" in sales_df.columns else 0
        st.metric("Total Revenue", f"₹{total:,.2f}")
        df_display = display_df(sales_df)
        del_no = st.selectbox("Select S.No to delete", df_display.index)
        if st.button("Delete Selected Bill"):
            sales_df = delete_row(sales_df, del_no - 1)
            save_data(sales_df, "sales")
            st.success("Bill deleted.")
            st.rerun()
    else:
        st.info("No sales yet.")

# ─────────────────────────────────────────────
#  EMPLOYEES
# ─────────────────────────────────────────────
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
                    new_row = {
                        "Name": name,
                        "Role": role,
                        "Phone": phone,
                        "Email": email,
                        "Salary": salary,
                        "Joining Date": str(join),
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])],
                                   ignore_index=True)
                    save_data(df, "employees")
                    st.success(f"✅ {name} added!")
                    st.rerun()
                else:
                    st.error("Name is required.")

    st.markdown("#### 👥 Staff List")
    if not df.empty:
        df_display = display_df(df)
        del_no = st.selectbox("Select S.No to delete", df_display.index)
        if st.button("Delete Selected Employee"):
            df = delete_row(df, del_no - 1)
            save_data(df, "employees")
            st.success("Employee removed.")
            st.rerun()
    else:
        st.info("No employees added yet.")

# ─────────────────────────────────────────────
#  SYMPTOM CHECKER  (100% OFFLINE — FIXED)
# ─────────────────────────────────────────────
elif menu == "🩺  Symptom Checker":
    page_header(
        "Symptom Checker",
        "Select symptoms to get disease prediction and medicine suggestions")

    models = load_ml_models()
    disease_df = load_data("disease_info")
    inv_df = load_data("inventory")

    if models["model"] is None:
        st.error(
            "⚠️ ML model not found. Place `model.pkl`, `label_encoder.pkl`, "
            "`symptom_list.pkl`, `disease_list.pkl` in the project root folder."
        )
        st.stop()

    symptom_list = models["symptom_list"]

    st.warning(
        "⚠️ **Medical Disclaimer:** This tool is for informational purposes only. "
        "It does not replace professional medical advice. Always consult a licensed "
        "doctor for diagnosis and treatment.")

    st.markdown("#### 🔍 Select Your Symptoms")
    st.caption("Search and select all symptoms the patient is experiencing")

    selected_symptoms = st.multiselect(
        "Symptoms",
        options=symptom_list,
        placeholder="Type to search symptoms...",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button(
            "🔬 Analyze Symptoms",
            use_container_width=True,
            disabled=len(selected_symptoms) == 0,
        )

    if selected_symptoms:
        st.caption(f"✅ {len(selected_symptoms)} symptom(s) selected")

    if analyze_btn and selected_symptoms:
        with st.spinner("Analyzing symptoms..."):
            predictions = predict_disease(selected_symptoms, models, top_n=3)

        if not predictions:
            st.error(
                "Could not generate predictions. Please check your model files."
            )
        else:
            st.markdown("---")
            st.markdown("### 🔮 Prediction Results")

            rank_labels = ["🥇 Most Likely", "🥈 Possible", "🥉 Alternative"]
            sev_icon = {"mild": "🟢", "moderate": "🟡", "severe": "🔴"}

            for i, pred in enumerate(predictions):
                disease_name = pred["disease"]
                confidence = pred["confidence"]
                info = get_disease_info(disease_name, disease_df)

                # Rank + native progress bar
                st.markdown(f"**{rank_labels[i]} — {disease_name.title()}**")
                st.progress(int(confidence), text=f"{confidence}% confidence")

                if info:
                    severity = str(info.get("severity",
                                            "moderate")).lower().strip()
                    icon = sev_icon.get(severity, "🟡")
                    desc = info.get("description", "No description available.")
                    consult = info.get("consult_if", "")

                    remedies = [
                        r.strip()
                        for r in str(info.get("home_remedies", "")).split("·")
                        if r.strip()
                    ]
                    med_cats = [
                        m.strip() for m in str(
                            info.get("medicine_category", "")).split("·")
                        if m.strip()
                    ]

                    with st.container(border=True):
                        h1, h2 = st.columns([4, 1])
                        with h1:
                            st.markdown(f"### {disease_name.title()}")
                        with h2:
                            st.markdown(f"**{icon} {severity.capitalize()}**")

                        st.markdown(desc)
                        st.divider()

                        rcol, mcol = st.columns(2)
                        with rcol:
                            st.markdown("**🌿 Home Remedies**")
                            if remedies:
                                for r in remedies:
                                    st.markdown(f"- {r}")
                            else:
                                st.caption("No home remedies listed.")

                        with mcol:
                            st.markdown("**💊 Suggested Medicine Categories**")
                            if med_cats:
                                for m in med_cats:
                                    st.markdown(f"- {m}")
                            else:
                                st.caption("No medicine categories listed.")

                        if consult:
                            st.info(f"🩺 **Consult a doctor if:** {consult}")

                        # Inventory cross-check
                        if not inv_df.empty and "Medicine Name" in inv_df.columns and med_cats:
                            med_cats_lower = [m.lower() for m in med_cats]
                            in_stock = []
                            for _, row in inv_df.iterrows():
                                med_lower = str(row["Medicine Name"]).lower()
                                if any(cat in med_lower or med_lower in cat
                                       for cat in med_cats_lower):
                                    qty = row.get("Quantity", 0)
                                    price = row.get("Price", 0)
                                    in_stock.append(
                                        f"**{row['Medicine Name']}** — {qty} units @ ₹{price}"
                                    )
                            if in_stock:
                                st.success(
                                    "✅ **Available in Your Inventory:**")
                                for item in in_stock:
                                    st.markdown(f"• {item}")

                else:
                    with st.container(border=True):
                        st.markdown(f"### {disease_name.title()}")
                        st.caption(
                            "Detailed info not available. Please consult a doctor."
                        )

                st.markdown("")

            st.info(
                "⚕️ **Important:** The above suggestions are for *temporary relief only*. "
                "This is not a substitute for professional medical diagnosis or treatment. "
                "Please visit a licensed physician for proper evaluation.")

# ─────────────────────────────────────────────
#  REPORTS & ANALYTICS
# ─────────────────────────────────────────────
elif menu == "📊  Reports":
    page_header("Reports & Analytics",
                "Insights to drive better pharmacy decisions")

    sales_df = load_data("sales")
    inv_df = load_data("inventory")
    rx_df = load_data("prescriptions")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rev = sales_df["Amount"].sum(
        ) if not sales_df.empty and "Amount" in sales_df.columns else 0
        st.metric("Total Revenue", f"₹{rev:,.2f}")
    with c2:
        bills = len(sales_df) if not sales_df.empty else 0
        st.metric("Total Bills", bills)
    with c3:
        avg = (rev / bills) if bills > 0 else 0
        st.metric("Avg Bill Value", f"₹{avg:,.2f}")
    with c4:
        low = (len(inv_df[inv_df["Quantity"] < 10])
               if not inv_df.empty and "Quantity" in inv_df.columns else 0)
        st.metric("Low Stock Items", low)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💊 Top Selling Medicines")
        if not sales_df.empty and "Medicine" in sales_df.columns:
            top = (sales_df.groupby("Medicine")["Quantity"].sum().sort_values(
                ascending=False).head(10))
            st.bar_chart(top)
        else:
            st.info("No sales data.")

    with col2:
        st.markdown("#### 📅 Revenue Over Time")
        if not sales_df.empty and "Date" in sales_df.columns and "Amount" in sales_df.columns:
            sales_df["Date"] = pd.to_datetime(sales_df["Date"],
                                              errors="coerce")
            daily = sales_df.groupby("Date")["Amount"].sum()
            st.line_chart(daily)
        else:
            st.info("No sales data.")

    st.markdown("#### ⚠️ Low Stock Medicines (< 10 units)")
    if not inv_df.empty and "Quantity" in inv_df.columns:
        low_stock = inv_df[inv_df["Quantity"] < 10]
        if not low_stock.empty:
            display_df(low_stock)
        else:
            st.success("✅ All medicines are adequately stocked!")
    else:
        st.info("No inventory data.")

    if not sales_df.empty:
        st.markdown("#### 📋 Full Sales Report")
        display_df(sales_df)
