import pandas as pd
import nltk
import string
import ast
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

# Load the data
df = pd.read_csv('travel_prepr+tags.csv')

# Convert the strings in the 'tokens' column back to lists
df['tokens'] = df['tokens'].apply(ast.literal_eval)

# Define your custom list of stopwords
custom_stopwords = ['located', 'also', 'various', 'offers', 'enjoy', 'provides', 'place']

# Add your custom stopwords to the list of English stopwords
stopwords = stopwords.words('english') + custom_stopwords


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(tokens)


def search(query, n=10):
    query = preprocess_text(query)
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)

    keywords = {
        82: ['restaurant'],
        85: ['festival'],
        80: ['hotel', 'pension', 'resort'],
        79: ['mall', 'shopping', 'mart'],
        77: ['station'],
    }

    matches = {contenttypeid: sum(keyword in query for keyword in words) for contenttypeid, words in keywords.items()}
    contenttypeid = max(matches, key=matches.get) if matches else None
    indices = df.index

    if contenttypeid is not None and matches[contenttypeid] > 0 and len(query.split()) > 1:
        indices = df[df['contenttypeid'] == contenttypeid].index
        similarities = similarities[:, indices]

    top_n = similarities[0].argsort()[-n:][::-1]
    top_n_indices = [indices[i] for i in top_n]
    top_n_scores = [similarities[0, i] for i in top_n]

    results = df.loc[top_n_indices][['title', 'overview', 'firstimage', 'tokens']]
    results['similarity'] = top_n_scores
    return results


def get_most_common_tags(df, n=20):
    all_tags = [tag for tags_list in df['tokens'] for tag in tags_list]
    return Counter(all_tags).most_common(n)


def display_results(results):
    for i, row in results.iterrows():
        col1, col2 = st.columns(2)

        if pd.notnull(row['firstimage']):
            col1.image(row['firstimage'], width=350)

        col2.markdown(f"## {row['title']}")

        if st.session_state.get(row['title'] + '_state', False):
            col2.markdown(row['overview'])
            if col2.button('Show less', key=row['title'] + '_button'):
                st.session_state[row['title'] + '_state'] = False
        else:
            overview_words = row['overview'].split()[:50]
            col2.markdown(' '.join(overview_words))
            if col2.button('Show more', key=row['title'] + '_button'):
                st.session_state[row['title'] + '_state'] = True

        tags = ' '.join(f'#{tag}' for tag in row['tokens'])
        st.markdown(tags)
        st.text(f"Similarity: {row['similarity']}")


# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['tags'])

# Create a search bar at the top of the page
st.title("Gangwon-do Travel Search")
query = st.text_input("Enter your query:")

# Display the most common tags below the search bar as buttons
common_tags = get_most_common_tags(df, n=10)
st.markdown("### Popular Tags:")

# Create a row of columns for the buttons
columns = st.columns(len(common_tags))

# Place each button within its own column
for col, (tag, count) in zip(columns, common_tags):
    if col.button(tag, key=tag):
        query = tag
        results = search(query, n=10)
        display_results(results)

# Perform the search and display the results when the user enters a query
if query and not any(st.session_state.get(tag, False) for tag, _ in common_tags):
    results = search(query, n=10)
    display_results(results)
