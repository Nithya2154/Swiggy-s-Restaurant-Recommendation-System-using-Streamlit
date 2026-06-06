# 🍽️ Swiggy Restaurant Recommendation System

A content-based restaurant recommendation system built using Swiggy data, powered by cosine similarity and deployed via Streamlit.

---

## 📌 Project Overview

This project recommends restaurants to users based on their preferences — city, cuisine type, cost, and rating — using **cosine similarity** on engineered features extracted from Swiggy restaurant data.

---

## 🗂️ Repository Structure

```
swiggy-recommend/
│
├── data/
│   ├── cleaned_data.csv          # Cleaned restaurant data
│   └── encoded_data.csv          # Feature-engineered & encoded data
│
├── models/
│   ├── mlb.pkl                   # MultiLabelBinarizer (cuisine encoder)
│   └── scaler.pkl                # StandardScaler (cost & rating normalizer)
│
├── notebooks/
│   ├── swiggy_recommend_data_cleaning.ipynb        # Step 1: Data Cleaning
│   ├── swiggy_recommend_feature_engineering.ipynb  # Step 2: Feature Engineering
│   └── swiggy_recommend_cosine_similarity.ipynb    # Step 3: Cosine Similarity Model
│
├── swiggy_streamlit.py           # Streamlit web app
└── README.md
```

---

## ⚙️ How It Works

### Step 1 — Data Cleaning
- Removed nulls, duplicates, and inconsistent values
- Standardized columns: `name`, `maincity`, `city`, `rating`, `rating_count`, `cost`, `cuisine`

### Step 2 — Feature Engineering
- Split multi-label `cuisine` column into lists
- Applied **MultiLabelBinarizer** to encode cuisines as binary vectors
- Applied **One-Hot Encoding** on `city`
- Applied **StandardScaler** on `cost` and `rating`
- Combined all features into a final encoded matrix (`encoded_data.csv`)

### Step 3 — Cosine Similarity
- Built a user preference vector from UI inputs (city, cuisine, cost, rating)
- Computed cosine similarity between the user vector and all restaurant vectors
- Returned the top-N most similar restaurants

---

## 🚀 Running the App

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/swiggy-recommend.git
cd swiggy-recommend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App

```bash
streamlit run swiggy_streamlit.py
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
pickle
```

Create a `requirements.txt` with:

```bash
pip freeze > requirements.txt
```

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| 🏙️ City Filter | Select your city from a dropdown |
| 🍜 Cuisine Input | Type preferred cuisines (e.g., Biryani, Chinese) |
| 💰 Cost Slider | Set your budget preference (₹0 – ₹1000) |
| ⭐ Rating Slider | Set minimum desired rating (0.0 – 5.0) |
| 📋 Recommendations | Top 5 restaurants displayed in a table |

---

## 📊 Dataset

The dataset contains Swiggy restaurant listings across multiple Indian cities with the following fields:

| Column | Description |
|---|---|
| `name` | Restaurant name |
| `maincity` | Primary city |
| `city` | Locality/area |
| `rating` | Average rating (0–5) |
| `rating_count` | Number of ratings |
| `cost` | Average cost for two (₹) |
| `cuisine` | Comma-separated cuisine types |

---

## 🧠 Model Details

- **Algorithm**: Cosine Similarity (content-based filtering)
- **Encoders**: `MultiLabelBinarizer` for cuisines, `get_dummies` for city
- **Scaler**: `StandardScaler` for `cost` and `rating`
- **Saved Artifacts**: `mlb.pkl`, `scaler.pkl`

---

## 📸 App Preview

> Launch the app and enter your preferences to get personalized restaurant recommendations instantly!

---

## 🙌 Acknowledgements

- Data sourced from **Swiggy** restaurant listings
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), and [pandas](https://pandas.pydata.org/)

---

## 📄 License

This project is for educational purposes only.
