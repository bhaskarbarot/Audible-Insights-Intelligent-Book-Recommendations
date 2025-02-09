import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity


def load_model(model_path=r"D:\\bhachu\\Book_Recommendations\\book_recommender_models_dbscan_lsa.pkl"):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def main():
    st.title('📚 Audible Book Recommendation System')

    # Load the model and data
    try:
        model_data = load_model()
        df = pd.read_csv(r"D:\\bhachu\\Book_Recommendations\\processed_audible_data(1).csv")

        # Ensure necessary columns are numeric
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df.dropna(subset=['Author', 'Book Name'], inplace=True)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Sidebar for user preferences
    st.sidebar.header('Your Preferences')

    # Rating filter
    min_rating = st.sidebar.slider('Minimum Rating (out of 5):', 1.0, 5.0, 4.0, 0.1)

    # Price filter
    max_price = st.sidebar.slider('Maximum Price (in $):', min_value=0.0, max_value=500.0, value=50.0)

    if st.button('Recommend Books'):
        # Filter books based on user input
        filtered_books = df[(df['Rating'] >= min_rating) &
                            (df['Price'] <= max_price)]

        if not filtered_books.empty:
            st.subheader("Recommended Books")
            for _, book in filtered_books.iterrows():
                with st.expander(f"{book['Book Name']} - {book['Author']}"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**Rating:** ⭐ {float(book['Rating']):.1f}")
                        st.write(f"**Price:** ${book['Price']:.2f}")
                    with col2:
                        st.write(f"**Description:** {book['Description'] if 'Description' in book and book['Description'] else 'No description available'}")
        else:
            st.warning("No books found matching your criteria. Please adjust your filters.")

    st.markdown('---')

    # Explore Books Tab
    st.header("Explore Books")

    search_method = st.radio("How would you like to find books?", ["Search by Book Name", "Browse All Books"])

    if search_method == "Search by Book Name":
        available_books = df['Book Name'].dropna().unique().tolist()
        book_name = st.selectbox('Search for a book:', available_books)
        
        if st.button('Find Similar Books'):
            try:
                # Get similar books using model
                book_idx = df[df['Book Name'] == book_name].index[0]
                features = model_data['features']
                
                if isinstance(features, np.ndarray):
                    similarities = cosine_similarity(
                        features[book_idx:book_idx+1],
                        features
                    )
                else:
                    similarities = cosine_similarity(
                        features[book_idx:book_idx+1].toarray(),
                        features.toarray()
                    )
                
                similar_indices = similarities.argsort()[0][-6:-1][::-1]
                similar_books = df.iloc[similar_indices]

                for _, book in similar_books.iterrows():
                    with st.expander(f"{book['Book Name']} by {book['Author']}"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(f"**Rating:** ⭐ {float(book['Rating']):.1f}")
                            st.write(f"**Price:** ${book['Price']:.2f}")
                        with col2:
                            st.write(f"**Description:** {book['Description'] if 'Description' in book and book['Description'] else 'No description available'}")

            except Exception as e:
                st.error(f"Error finding similar books: {str(e)}")

    else:  # Browse All Books
        filtered_books = df[df['Rating'] >= min_rating]
        
        filtered_books = filtered_books.sort_values('Rating', ascending=False)
        
        st.subheader('Top Rated Books in Your Preferred Genres')
        for _, book in filtered_books.head(10).iterrows():
            with st.expander(f"{book['Book Name']} by {book['Author']}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**Rating:** ⭐ {float(book['Rating']):.1f}")
                    st.write(f"**Price:** ${book['Price']:.2f}")
                with col2:
                    st.write(f"**Description:** {book['Description'] if 'Description' in book and book['Description'] else 'No description available'}")

if __name__ == "__main__":
    main()
