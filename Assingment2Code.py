import os
import math
import re
from collections import defaultdict

# Path to the corpus (documents folder)
CORPUS_DIRECTORY = "C:\\Users\\Umang\\Desktop\\Corpus"

# Function to preprocess text: lowercasing and tokenizing
def clean_and_tokenize(text):
    lowercase_text = text.lower()  # Convert text to lowercase
    tokens = re.findall(r'\b\w+\b', lowercase_text)  # Extract tokens (words)
    return tokens

# Function to build the inverted index and document lengths
def create_inverted_index(directory_path):
    inverted_index = defaultdict(list)  # Dictionary to store term -> [(doc_name, term_freq)]
    document_lengths = defaultdict(float)  # To store document lengths for normalization
    total_docs = 0  # Counter for the total number of documents
    
    # Loop through each document in the corpus directory
    for document_name in os.listdir(directory_path):
        total_docs += 1
        document_path = os.path.join(directory_path, document_name)
        
        # Read the contents of the document
        with open(document_path, 'r', encoding='utf-8') as document_file:
            document_content = document_file.read()
            word_list = clean_and_tokenize(document_content)  # Tokenize the document content
            
            # Dictionary to store term frequencies for the current document
            term_frequencies = defaultdict(int)
            
            # Count the frequency of each term in the document
            for word in word_list:
                term_frequencies[word] += 1
            
            # Update the inverted index with term frequencies for the current document
            for word, frequency in term_frequencies.items():
                inverted_index[word].append((document_name, frequency))
            
            # Calculate the document length (using squared term weights)
            squared_weights_sum = sum((1 + math.log10(frequency))**2 for frequency in term_frequencies.values())
            document_lengths[document_name] = math.sqrt(squared_weights_sum)  # Square root of sum of squared weights
    
    return inverted_index, document_lengths, total_docs

# Function to compute tf-idf for query terms
def calculate_query_tf_idf(query_words, index, total_docs):
    query_tf_idf = defaultdict(float)  # To store tf-idf for query terms
    query_term_frequencies = defaultdict(int)  # To store term frequencies in the query
    
    # Calculate term frequencies for the query terms
    for word in query_words:
        query_term_frequencies[word] += 1
    
    # Compute tf-idf for each query term
    for word, frequency in query_term_frequencies.items():
        if word in index:
            document_frequency = len(index[word])  # Number of documents containing the term
            inverse_doc_frequency = math.log10(total_docs / document_frequency)  # IDF calculation
            query_term_weight = 1 + math.log10(frequency)  # Query term frequency (TF)
            query_tf_idf[word] = query_term_weight * inverse_doc_frequency  # Compute tf-idf
    
    return query_tf_idf

# Function to calculate cosine similarity between the query and documents
def compute_cosine_similarity(query_weights, index, doc_lengths):
    document_scores = defaultdict(float)  # To store similarity scores for each document
    
    # Compute dot product for each document containing query terms
    for word, query_weight in query_weights.items():
        if word in index:
            postings = index[word]
            for doc_name, doc_term_freq in postings:
                document_weight = 1 + math.log10(doc_term_freq)  # Document term frequency (TF)
                document_scores[doc_name] += query_weight * document_weight  # Dot product for cosine similarity
    
    # Normalize document scores using their precomputed lengths
    for doc_name in document_scores:
        if doc_lengths[doc_name] != 0:  # Avoid division by zero
            document_scores[doc_name] /= doc_lengths[doc_name]  # Cosine normalization
    
    return document_scores

# Main function to handle user queries and output results
def search_documents(user_query, directory_path):
    # Step 1: Build the inverted index and compute document lengths
    index, doc_lengths, doc_count = create_inverted_index(directory_path)
    
    # Step 2: Preprocess the query (tokenize and clean)
    query_tokens = clean_and_tokenize(user_query)
    
    # Step 3: Compute tf-idf for query terms
    query_tf_idf = calculate_query_tf_idf(query_tokens, index, doc_count)
    
    # Step 4: Calculate cosine similarity for each document
    similarity_scores = compute_cosine_similarity(query_tf_idf, index, doc_lengths)
    
    # Step 5: Sort documents by similarity score and document name for ties
    ranked_documents = sorted(similarity_scores.items(), key=lambda item: (-item[1], item[0]))
    
    # Return the top 10 ranked documents
    return ranked_documents[:10]

# User input for the search query
query = input("Enter your search query: ")

# Get the ranked results from the search function
results = search_documents(query, CORPUS_DIRECTORY)

# Display the results in the required format (document name and similarity score)
for document, score in results:
    print(f"('{document}', {score})")
