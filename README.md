---
title: Ankur The Agribot
emoji: 🌱
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
---


# 🌱 Ankur The AgriBot – AI-powered Agricultural Assistant

AgriBot is an intelligent chatbot designed to assist farmers by providing hyper-local agricultural insights, weather forecasts, market prices, and crop advisory.  
It integrates multiple APIs with an LLM backend to deliver **actionable, real-time guidance**.

---

## 🚀 Features
-  Crop advisory using AI
-  Real-time weather updates
-  Market price information
-  Satellite-based monitoring via Agromonitoring API
-  Integrated with multiple government and third-party data sources

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ankur_the_agribot.git
cd ankur_the_agribot
```

---

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 3. Configure API Keys

AgriBot requires multiple APIs for full functionality. You can set up keys in two ways:

#### 🔹 Local Development (`.env` file)
Create a `.env` file in the project root and add:
```env
GROQ_API_KEY=your_groq_key_here
OPENWEATHER_API_KEY=your_openweather_key_here
TAVILY_API_KEY=your_tavily_key_here
AGROMONITORING_API_KEY=your_agromonitoring_key_here
DATA_GOV_IN_API_KEY=your_datagovin_key_here
```

#### 🔹 Hugging Face Spaces (Recommended for Deployment)
1. Go to your Space → **Settings → Repository secrets**.  
2. Add the following secrets:
   - `GROQ_API_KEY`
   - `OPENWEATHER_API_KEY`
   - `TAVILY_API_KEY`
   - `AGROMONITORING_API_KEY`
   - `DATA_GOV_IN_API_KEY`

3. In the code, they will be automatically available as environment variables:
```python
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AGROMONITORING_API_KEY = os.getenv("AGROMONITORING_API_KEY")
DATA_GOV_IN_API_KEY = os.getenv("DATA_GOV_IN_API_KEY")
```


---

### 4. Run the Application
If your main script is `app.py` (Gradio-based UI):
```bash
python app.py
```

AgriBot will launch locally. Open the provided URL in your browser.

---

## 🛠️ Tech Stack
- **Python** 🐍
- **Gradio** – UI for chat
- **Hugging Face / GROQ** – LLM inference
- **OpenWeather** – Weather data
- **Tavily API** – Knowledge retrieval
- **AgroMonitoring API** – Satellite insights
- **Data.gov.in API** – Government agricultural datasets

---

## 📂 Project Structure
```
agribot/
│── app.py                # Main entry point
│── Content/index         # Chunks Storage
│── requirements.txt       # Dependencies
│── .env.example           # Example env file
│── README.md              # Project docs
```

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License
This project is licensed under the MIT License.

