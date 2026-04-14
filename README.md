# ZWMBGUIDE-AI  
### AI-Powered Banking Decision Support System

---

## Overview

ZWMBGUIDE-AI is a standalone data science-driven banking decision support system designed to assist retail banking staff in making informed decisions. The system integrates machine learning techniques to provide insights into credit risk prediction, fraud detection, customer segmentation, and data visualization through an interactive dashboard.

The application is built using Python, Streamlit, and Scikit-learn, and is designed to simulate real-world banking analytics tools without requiring integration into a core banking system.

---

## Features

### Credit Risk Prediction
- Predicts whether a customer is high risk or low risk  
- Provides a risk probability score (%)  
- Uses Random Forest Classifier  

---

### Fraud Detection
- Detects suspicious transactions using anomaly detection  
- Identifies unusual patterns in:
  - Transaction amount  
  - Transaction time  
- Uses Isolation Forest Algorithm  

---

### Customer Segmentation
- Groups customers into:
  - High Value  
  - Medium Value  
  - Low Value  
- Uses K-Means Clustering  

---

### Interactive Dashboard
- Displays:
  - Total Customers  
  - Fraud Alerts  
  - High Risk Customers  
- Includes:
  - Scatter plots  
  - Distribution charts  
  - Transaction visualizations  

---

### Authentication System
- Simple login system for controlled access  
- Role-based simulation (Admin & Manager)  

---
## Installation & Setup

### 1. Clone the Repository
```
git clone https://github.com/richmondtinashemukuku-blip/FinGuardAI.git
cd FinGuardAI
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the Application
```
python -m streamlit run app.py
```
---

## Test Credentials

Use the following credentials to access the system:

### Admin Credentials
Username:
```
admin
```
Password:
```
admin
```

### Manager Credentials
Username: 
```
manager
```
Password: 
```
1234
```

| Role    | Username | Password |
|--------|----------|----------|
| Admin   | admin    | admin     |
| Manager | manager  | 1234    |

---

## Project Structure

```
ZWMBGUIDE-AI/
│
├── app.py
├── assets/
│   └── images/
│       └── dark_logo.png
├── data/
│   ├── customers.csv
│   └── transactions.csv
├── requirements.txt
└── README.md
```


---


## How to Use the System
1. Login using provided credentials
2. Upload your own datasets (optional)
3. Navigate using tabs:
- Data
- Risk Prediction
- Fraud Detection
- Dashboard
- Segmentation

4.Interact with models:
- Predict credit risk
- Check transaction fraud
- Analyse customer segments

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

## Use Cases
- Loan approval decision support
- Fraud monitoring
- Customer behaviour analysis
- Banking data visualization

## Limitations
- Uses synthetic data (not real banking data)
- Not integrated with live banking systems
- Basic authentication (for simulation purposes only)

## Future Improvements
- Real-time data integration
- Advanced model tuning
- Role-based access control
- Export reports (PDF/Excel)
- Interactive dashboards (Plotly)

## Author

Richmond Mukuku - R2310251J
Data Science Student | ICT Specialist

## License

This project is for academic and demonstration purposes.