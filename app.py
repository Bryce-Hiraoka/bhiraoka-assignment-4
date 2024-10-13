from flask import Flask, render_template, request
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the 20 newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(newsgroups_data.data)

# Perform SVD
n_components = 100
svd = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = svd.fit_transform(tfidf_matrix)

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0  # Return 0 similarity if either vector is zero
    return np.dot(vec1, vec2) / (norm1 * norm2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    
    # Transform query to TF-IDF vector
    query_tfidf = vectorizer.transform([query])
    
    # Check if the query vector is all zeros
    if query_tfidf.nnz == 0:
        return render_template('results.html', error="Query contains only stop words or unknown terms. Please try a different query.")
    
    # Project query into LSA space
    query_lsa = svd.transform(query_tfidf)
    
    # Compute cosine similarities
    similarities = [cosine_similarity(query_lsa[0], doc_vec) for doc_vec in lsa_matrix]
    
    # Get top 5 documents
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_documents = [(newsgroups_data.data[i][:200] + "...", similarities[i]) for i in top_indices]
    
    # Create bar chart
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(top_documents)), [doc[1] for doc in top_documents], alpha=0.7, color='blue')
    plt.yticks(range(len(top_documents)), [f'Document {i+1}' for i in range(len(top_documents))])
    plt.xlabel('Cosine Similarity')
    plt.title('Top 5 Documents Cosine Similarity to Query')
    
    # Save plot to a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    plt.close()  # Close the plot to free up memory
    
    return render_template('results.html', documents=top_documents, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
