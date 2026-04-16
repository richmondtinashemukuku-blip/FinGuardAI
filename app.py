import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans

# ===============================
# 🎨 UI + ICONS + TYPOGRAPHY
# ===============================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

<style>

/* Global font */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;    
}

/* Background */
body {
    background: linear-gradient(135deg, #f0fdf4, #ffffff);
}

/* Glass cards */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Titles */
h2, h3 {
    color: #064E3B;  /* dark emerald */
}
.section-title {
    font-size: 22px;
    color: #065F46;  /* slightly lighter emerald */
    font-weight: 600;
    margin-bottom: 15px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #10B981, #D4AF37);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.35);
}

/* Inputs */
.stNumberInput input {
    border-radius: 10px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 15px;
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# 🔐 LOGIN
# ===============================
users = {"admin": "admin", "manager": "1234"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.image("assets/images/dark_logo.png", width=120)
    
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.role = username
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ===============================
# HEADER
# ===============================
st.markdown(
    "<p style='text-align:center; color:#065F46; font-size:18px;'>AI Banking Decision Support System</p>",
    unsafe_allow_html=True
)

st.sidebar.markdown(f"👤 **{st.session_state.role}**")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.markdown("## 📂 Upload Data")

customer_file = st.sidebar.file_uploader("Customer CSV", type=["csv"])
transaction_file = st.sidebar.file_uploader("Transaction CSV", type=["csv"])

# ===============================
# DATA
# ===============================
customers = pd.read_csv(customer_file) if customer_file else pd.read_csv("data/customers.csv")
transactions = pd.read_csv(transaction_file) if transaction_file else pd.read_csv("data/transactions.csv")

# ===============================
# MODELS
# ===============================
X = customers[['age','income','account_balance','loan_amount','repayment_history','num_transactions']]
y = customers['default']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

risk_model = RandomForestClassifier()
risk_model.fit(X_train,y_train)

fraud_data = transactions[['amount','time']]
fraud_model = IsolationForest(contamination=0.3)
fraud_model.fit(fraud_data)

transactions['fraud_prediction'] = fraud_model.predict(fraud_data)
transactions['fraud_prediction'] = transactions['fraud_prediction'].map({1:'Normal',-1:'Fraud'})

seg_data = customers[['income','account_balance','num_transactions']]
kmeans = KMeans(n_clusters=3, random_state=42)
customers['segment'] = kmeans.fit_predict(seg_data)
customers['segment'] = customers['segment'].map({0:'Low Value',1:'Medium Value',2:'High Value'})

# ===============================
# KPIs
# ===============================
col1,col2,col3 = st.columns(3)
col1.metric("Customers", len(customers))
col2.metric("Fraud Alerts", (transactions['fraud_prediction']=="Fraud").sum())
col3.metric("High Risk", customers['default'].sum())

# ===============================
# TABS
# ===============================
tabs = st.tabs(["Data","Risk","Fraud","Dashboard","Segments"])

# ===============================
# DATA TAB
# ===============================
with tabs[0]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><i class="fas fa-users"></i> Customer Data</div>', unsafe_allow_html=True)
    st.write(customers)

    st.markdown('<div class="section-title"><i class="fas fa-credit-card"></i> Transaction Data</div>', unsafe_allow_html=True)
    st.write(transactions)

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# RISK TAB
# ===============================
with tabs[1]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><i class="fas fa-brain"></i> Credit Risk Prediction</div>', unsafe_allow_html=True)

    age = st.number_input("Age",18,100)
    income = st.number_input("Income")
    balance = st.number_input("Balance")
    loan = st.number_input("Loan")
    repay = st.slider("Repayment",0.0,1.0)
    trans = st.number_input("Transactions")

    if st.button("Predict Risk"):
        data = [[age,income,balance,loan,repay,trans]]
        pred = risk_model.predict(data)
        prob = risk_model.predict_proba(data)

        score = prob[0][1]*100

        if pred[0]==1:
            st.error("High Risk Customer")
        else:
            st.success("Low Risk Customer")

        st.write(f"Risk Score: {score:.2f}%")
        st.progress(int(score))

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FRAUD TAB
# ===============================
with tabs[2]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><i class="fas fa-shield-halved"></i> Fraud Detection</div>', unsafe_allow_html=True)
    st.write(transactions)

    st.markdown('<div class="section-title"><i class="fas fa-magnifying-glass"></i> Check Transaction</div>', unsafe_allow_html=True)

    amt = st.number_input("Amount")
    time = st.number_input("Time")

    if st.button("Check Fraud"):
        res = fraud_model.predict([[amt,time]])
        if res[0]==-1:
            st.error("Fraud Detected")
        else:
            st.success("Normal Transaction")

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# DASHBOARD TAB
# ===============================
with tabs[3]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><i class="fas fa-chart-pie"></i> Banking Dashboard</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers",len(customers))
    c2.metric("Risk",customers['default'].sum())
    c3.metric("Fraud",(transactions['fraud_prediction']=="Fraud").sum())
    c4.metric("Avg Balance",f"{customers['account_balance'].mean():.2f}")

    colA,colB = st.columns(2)

    with colA:
        fig,ax=plt.subplots()
        ax.bar(['Safe','Risky'],customers['default'].value_counts())
        st.pyplot(fig)

    with colB:
        fig,ax=plt.subplots()
        ax.bar(transactions['fraud_prediction'].value_counts().index,
               transactions['fraud_prediction'].value_counts().values)
        st.pyplot(fig)

fig, ax = plt.subplots()

ax.scatter(
    transactions['time'],
    transactions['amount'],
    alpha=0.7,
    s=50,
    marker='o',
    c='gold',
    edgecolors='white'
)

ax.grid(True, linestyle='--', alpha=0.5)

ax.set_xlabel("Transaction Time")
ax.set_ylabel("Transaction Amount")
ax.set_title("Transaction Pattern (Dotted Scatter Plot)")

st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# SEGMENTATION TAB
# ===============================
with tabs[4]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><i class="fas fa-layer-group"></i> Customer Segmentation</div>', unsafe_allow_html=True)

    st.write(customers)

    fig,ax=plt.subplots()
    for seg in customers['segment'].unique():
        d = customers[customers['segment']==seg]
        ax.scatter(d['income'],d['account_balance'],label=seg)

    ax.legend()
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)