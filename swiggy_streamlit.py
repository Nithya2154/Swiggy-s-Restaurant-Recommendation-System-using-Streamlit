import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load data
cleaned_df = pd.read_csv("cleaned_data.csv")
encoded_df = pd.read_csv("encoded_data.csv")

# ✅ Load models
mlb = pickle.load(open("mlb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ✅ Create user vector


def create_user_vector(user_input):

    default_input = {
        'city': None,
        'cuisine': None,
        'cost': 0,
        'rating': 0
    }

    default_input.update(user_input)

    user_df = pd.DataFrame([default_input])

    # City
    if default_input['city']:
        city_encoded = pd.get_dummies(user_df['city'], prefix='city')
    else:
        city_encoded = pd.DataFrame()

    # Cuisine
    if default_input['cuisine']:
        cuisine_list = [c.strip() for c in default_input['cuisine'].split(',')]
        cuisine_encoded = mlb.transform([cuisine_list])
        cuisine_df = pd.DataFrame(cuisine_encoded, columns=mlb.classes_)
    else:
        cuisine_df = pd.DataFrame(columns=mlb.classes_)

    # Combine
    final_user = pd.concat([
        user_df[['cost', 'rating']],
        city_encoded,
        cuisine_df
    ], axis=1)

    # Align columns
    final_user = final_user.reindex(columns=encoded_df.columns, fill_value=0)

    # Scale
    final_user[['cost', 'rating']] = scaler.transform(
        final_user[['cost', 'rating']]
    )

    return final_user


# ✅ Recommendation function
def recommend_for_user(user_vector, user_input, top_n=5):

    df_filtered = cleaned_df.copy()
    encoded_filtered = encoded_df.copy()

    # City filter
    if user_input.get('city'):
        mask = df_filtered['city'].str.lower() == user_input['city'].lower()
        df_filtered = df_filtered[mask]
        encoded_filtered = encoded_filtered[mask]

    # Cuisine filter
    if user_input.get('cuisine'):
        cuisine_list = [c.strip().lower()
                        for c in user_input['cuisine'].split(',')]

        mask = df_filtered['cuisine'].str.lower().apply(
            lambda x: any(c in x for c in cuisine_list)
        )

        df_filtered = df_filtered[mask]
        encoded_filtered = encoded_filtered[mask]

    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)
    encoded_filtered = encoded_filtered.reset_index(drop=True)

    # Fallback
    if len(df_filtered) == 0:
        df_filtered = cleaned_df.reset_index(drop=True)
        encoded_filtered = encoded_df.reset_index(drop=True)

    # Similarity
    similarity = cosine_similarity(user_vector, encoded_filtered)

    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in scores[:top_n]]

    return df_filtered.iloc[top_indices][
        ['name', 'maincity', 'city', 'cuisine', 'rating', 'cost']
    ]


# ✅ UI STARTS HERE
st.title("🍽️ Restaurant Recommendation System")

st.write("Enter your preferences:")

# ✅ Inputs
city = st.selectbox(
    "Select City", [""] + sorted(cleaned_df['city'].dropna().unique()))
cuisine = st.text_input("Enter Cuisine (e.g., Biryani, Chinese)")
cost = st.slider("Cost Preference", 0, 1000, 200)
rating = st.slider("Rating Preference", 0.0, 5.0, 3.0)

# ✅ Button
if st.button("Recommend"):

    user_input = {}

    if city != "":
        user_input['city'] = city
    if cuisine:
        user_input['cuisine'] = cuisine

    user_input['cost'] = cost
    user_input['rating'] = rating

    # Create vector
    user_vec = create_user_vector(user_input)

    # Get recommendations
    results = recommend_for_user(user_vec, user_input)

    # Display
    st.subheader("Top Recommendations 🍴")
    st.dataframe(results)
