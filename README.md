# 🛠️ Machine Decaying Predictor

A robust Flask-based application designed for advanced machine degradation prediction using various analytical models. The system supports detailed preprocessing, dual analysis methods, visual insights, and Excel reporting—geared towards predictive maintenance scenarios.

---

## 🚀 Features

### 🔍 Core Functionality

- **Dual Analysis Modes**: Incremental and Fallback days analysis  
- **Advanced Preprocessing**: Filtering, gap handling, interpolation, and oversampling  
- **Multiple Prediction Models**: Linear, Polynomial (1–4 degrees), and Exponential  
- **High-Quality Visuals**: Forecast lines, thresholds, confidence zones  
- **Excel Report Generator**: Auto-generated reports with fitted values & summaries  
- **Real-Time Updates**: Progress tracking and batch-wise results  

---

## ⚙️ Project Structure

```
Machine Decaying Predictor/
│
├── app.py                        # Main Flask app entry point
├── requirements.txt              # Python dependencies
│
├── data_preprocessor.py          # Preprocessing logic: filtering, gaps, oversampling
├── data_processor.py             # Data transformation and formatting
├── fallback_analyzer.py          # Fallback days method logic
├── predictor.py                  # Incremental method logic
├── visualizer.py                 # Chart generation and visualization
│
├── models/                       # Machine decay models
│   ├── linear_model.py
│   ├── polynomial_model.py
│   ├── exponential_model.py
│   └── model_manager.py          # Manages model loading and usage
│
├── reports/                      # Folder for Excel report generation logic
│
├── routes/                       # Flask API routes
│   ├── main_routes.py
│   ├── analysis_routes.py
│   └── report_routes.py
│
├── scripts/
│   └── cleanup.py                # File cleanup script
│
├── static/
│   └── graphs/                   # Saved graphs/images
│
├── templates/                    # HTML templates (rendered by Flask)
│   ├── index.html
│   ├── results.html
│   └── fallback_results.html
│
├── uploads/                      # User-uploaded files
│
├── utils/
│   ├── file_utils.py             # Helpers for file validation/cleanup
│   └── report_generator.py       # Excel report writer
│
└── venv/                         # Python virtual environment
```

---

## 🛠️ Installation & Setup

### ✅ Prerequisites
- Python 3.8 or higher  
- Pip (Python package manager)  

---

### ⚙️ Backend Setup (Flask)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Machine\ Decaying\ Predictor
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**
   ```bash
   python app.py
   ```
   Your server runs at: http://127.0.0.1:5000

---

## 📊 Usage Workflow

- **Upload Data**: Lower and Upper CSV files with required columns  
- **Configure Parameters**: Select models, method, and thresholds  
- **Run Analysis**: Choose Incremental or Fallback analysis  
- **View Results**: Visuals + metrics (R², RMSE, critical threshold)  
- **Download Reports**: Excel output with fitted values, stats, and forecasts  

---

## 📁 Data Format Requirements

- **CSV Columns**: Datetime, RMS, Energy, Predicted RUL  
- **Two Datasets**: Lower-bound and Upper-bound sensor data  
- **Order**: Chronologically sorted time series  

---

## 🔬 Analysis Methods

### 📘 Incremental Method
- Progressive batch analysis: 100, 200, 300... rows  
- Supports forward/backward/bidirectional  
- Visualizes each batch, compares all models  

### 📕 Fallback Days Method
- Uses last `n` days for predictions  
- Calculates baseline & critical thresholds  
- Supports bootstrap confidence intervals  

---

## 📉 Prediction Models

- Linear Regression  
- Polynomial Regression (Degrees 1 to 4)  
- Exponential Decay  
- R² and RMSE provided for model comparison  
- Critical failure date estimation  

---

## 📈 Visualizations

- Scatter points of raw RUL data  
- Fitted model lines (solid)  
- Forecasts (dashed)  
- Critical threshold (horizontal red)  
- “Today” marker (green vertical)  
- Grid of all model charts (2x3 layout)  
- High-resolution PNG export  

---

## 🙏 Acknowledgments

- **Python Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib  
- **Frameworks**: Flask for API, Bootstrap for front-end templates  
- **You**: For exploring and improving predictive maintenance! 💪  

---

**Machine Decaying Predictor** – Bringing predictive insights to machine life expectancy through clean data science and reliable models.
