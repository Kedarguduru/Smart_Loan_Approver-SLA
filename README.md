<div align="center">

<h2>🧠 Smart Loan Approver</h2>  
Machine Learning + Node.js Backend + Interactive Frontend UI

🚀 Predict whether a loan should be **Approved or Rejected** based on applicant data  
📊 Trained from CSV using a **Decision Tree Classifier**  
⚡ Fast API · 🎨 Clean UI · 🔁 Retrain Anytime

</div>

---

<h4>📌 Features</h4>

- ✅ Use CSV dataset for ML training  
- ✅ Train model from browser (no Python required)  
- ✅ Predict loan approval from user input  
- ✅ Saves trained model as JSON & reloads instantly  
- ✅ Clear success/error messages in UI  
- ✅ Works offline after setup  
- ✅ Beginner-friendly & customizable

---

📂 Project Structure

```txt
🧠 smart-loan-approver/
│
├── 🖥️ backend/                  # ML API & Server
│   ├── ⚙️ index.js              # Handles /train & /predict routes
│   ├── 📁 data/
│   │   └── 📄 M2 T2.csv         # Loan dataset (training source)
│   ├── 🧪 ml_model/             # Auto-generated after training
│   │      ├── tree.json         # Saved Decision Tree model
│   │      └── meta.json         # Feature + target metadata
│
├── 🌐 frontend/                 # User Interface
│   └── 🪟 index.html            # UI (Train + Predict buttons)
│
├── 📦 package.json              # Dependencies + npm scripts
└── 📘 README.md                 # Documentation (this file)
```
---
<h4>🧠 How the ML Prediction Works</h4>

This project uses a Decision Tree Classifier trained on real loan application data. When the user enters Credit Score, Income, and Loan Amount, the model compares the values against learned patterns and follows a series of logical decision branches to determine whether the loan should be ✅ Approved or ❌ Rejected. The prediction also returns a confidence score (probability %) along with a brief reasoning 
summary — making the result transparent and easy to understand, not a “black-box” model.
