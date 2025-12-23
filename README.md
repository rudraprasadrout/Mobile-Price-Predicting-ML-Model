Markdown

# 📱 Mobile Price Prediction ML System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Netlify Status](https://img.shields.io/badge/Frontend-Netlify-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://mobiprice.netlify.app)
[![Render Status](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://mobile-price-predictor-ml-model.onrender.com)

An end-to-end Machine Learning solution designed to predict mobile phone price ranges. This project features a decoupled architecture with a high-performance ML backend and a sleek web frontend.

---

## 🔗 Live Demo
* **Frontend UI:** [mobiprice.netlify.app](https://mobiprice.netlify.app)
* **API Endpoint:** [mobile-price-predictor-ml-model.onrender.com](https://mobile-price-predictor-ml-model.onrender.com)

---

## 🏗️ System Architecture
The project follows a **Decoupled Architecture** to ensure scalability and separation of concerns:

1.  **Frontend (Netlify):** A modern, responsive HTML/JS interface that collects user input and communicates with the backend via REST API.
2.  **Backend API (Render):** A Flask-based server that loads the `.pkl` models to process real-time prediction requests.
3.  **ML Core:** A robust pipeline that handles data cleaning and classification logic.



---

## 📂 Project Structure
```bash
├── 📁 API and Html/           # Web Deployment Assets
│   ├── app.py                 # Flask Server (Hosted on Render)
│   ├── index.html             # Frontend UI (Hosted on Netlify)
│   ├── procfile               # Deployment config for Render/Heroku
│   └── requirements.txt       # Production dependencies
├── accuracy.py                # Model evaluation scripts
├── accuracy_plot.png          # Performance visualization
├── Flipdata.xlsx              # Raw dataset
├── mobile_cleaned_data.xlsx   # Preprocessed dataset
├── model.pkl                  # Trained ML Model (Serialized)
├── pipeline.pkl               # Data transformation pipeline
├── main.py                    # Main execution script
└── predictor.ipynb            # Research & Development Notebook
## 🚀 How It Works

### 1. Data Processing
Using `mobile_cleaned_data.xlsx`, the system performs feature engineering to identify the strongest predictors such as **RAM, Battery Power, and Internal Memory**.

### 2. Model Pipeline
We use a serialized `pipeline.pkl` to ensure that data sent via the web form is scaled and encoded exactly like the training data, preventing "feature mismatch" errors during inference.

### 3. Real-time Prediction
When a user submits the form on the frontend, a **POST** request is sent to the Render API. The backend predicts one of four price ranges:

| Value | Category | Icon |
| :--- | :--- | :--- |
| **0** | Budget | 🏷️ |
| **1** | Mid-Range | 📱 |
| **2** | High-End | ✨ |
| **3** | Flagship | 💎 |

---

## 🛠️ Local Development

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/mobile-price-prediction.git](https://github.com/your-username/mobile-price-prediction.git)
cd mobile-price-prediction
2. Install Dependencies
Bash

pip install -r "API and Html/requirements.txt"
3. Run the Flask API
Bash

cd "API and Html"
python app.py

---

## 📊 Performance

The model was evaluated using `accuracy.py`, resulting in the following performance metrics:

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | 98.5% |
| **Testing Accuracy** | 92.1% |
| **F1-Score** | 0.91 |



---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model or the web interface, please feel free to submit a **Pull Request**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

### 👨‍💻 Developed by
**[Rudra Prasad Rout]** * Data Science & Machine Learning Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/routrp07/)
![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:routrp07@gmail.com)