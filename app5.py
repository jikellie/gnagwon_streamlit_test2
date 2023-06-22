# this is with the new search format using TF-IDF vectorizer

import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import ast


# Load the data
df = pd.read_csv('travel_prepr+tags.csv')

# Convert the strings in the 'tokens' column back to lists
df['tokens'] = df['tokens'].apply(ast.literal_eval)

# Define your custom list of stopwords
custom_stopwords = ['located', 'also', 'various', 'offers', 'enjoy', 'provides', 'place']

# Add your custom stopwords to the list of English stopwords
stopwords = stopwords.words('english') + custom_stopwords

# Modify the preprocess_text function to use your custom stopwords
def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Define a function to return the top n most similar documents to a query
def search(query, n=10):
    # Preprocess the query
    query = preprocess_text(query)
    # Transform the query using the vectorizer
    query_tfidf = vectorizer.transform([query])

    # Compute the cosine similarity between the query and all documents
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)

    # Define keywords for each contenttypeid
    keywords = {
        82: ['restaurant'],
        85: ['festival'],
        80: ['hotel', 'pension', 'resort'],
        79: ['mall', 'shopping', 'mart'],
        77: ['station'],
        # Add more here if needed
    }

    # Count the number of matches for each contenttypeid
    matches = {contenttypeid: sum(keyword in query for keyword in words) for contenttypeid, words in keywords.items()}

    # Get the contenttypeid with the most matches
    contenttypeid = max(matches, key=matches.get) if matches else None

    # Start with the indices of all documents
    indices = df.index

    # If a contenttypeid was found and the query contains more than one word, restrict the search to that contenttypeid
    if contenttypeid is not None and matches[contenttypeid] > 0 and len(query.split()) > 1:
        indices = df[df['contenttypeid'] == contenttypeid].index
        similarities = similarities[:, indices]

    # Get the top n most similar documents
    top_n = similarities[0].argsort()[-n:][::-1]

    # Map the indices of the top_n results back to the original DataFrame
    top_n_indices = [indices[i] for i in top_n]

    # Get the similarity scores of the top_n results
    top_n_scores = [similarities[0, i] for i in top_n]

    # Return the titles, overviews, and similarity scores of the top n documents
    results = df.loc[top_n_indices][['title', 'overview', 'firstimage', 'tokens']]
    results['similarity'] = top_n_scores
    return results


def get_top_n_words(row, n=5):
    return row.sort_values(ascending=False).head(n).index.tolist()

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the 'overview' column
tfidf_matrix = vectorizer.fit_transform(df['tags'])

# Create a search bar at the top of the page
st.title("Gangwon-do Travel Search")
query = st.text_input("Enter your query:")

# When the user enters a query, perform the search and display the results
if query:
    results = search(query, n=10)  # Get the top 10 results
    for i, row in results.iterrows():
        # Create two columns
        col1, col2 = st.columns(2)

        # If there's an image, display it in the left column
        if pd.notnull(row['firstimage']):
            col1.image(row['firstimage'], width=350)  # Set the width of the image

        # Display the title and overview in the right column
        col2.markdown(f"## {row['title']}")  # Use markdown to control the size of the title

        # Check if the button has been clicked
        if st.session_state.get(row['title'] + '_state', False):
            # If the button has been clicked, display the full text and a "Show less" button
            col2.markdown(row['overview'])
            if col2.button('Show less', key=row['title'] + '_button'):
                st.session_state[row['title'] + '_state'] = False
        else:
            # If the button has not been clicked, display the first 50 words and a "Show more" button
            overview_words = row['overview'].split()[:50]
            col2.markdown(' '.join(overview_words))
            if col2.button('Show more', key=row['title'] + '_button'):
                st.session_state[row['title'] + '_state'] = True

        # Display the tags as hashtags
        tags = ' '.join(f'#{tag}' for tag in row['tokens'])
        st.markdown(tags)

        # Display the similarity score below the columns
        st.text(f"Similarity: {row['similarity']}")
