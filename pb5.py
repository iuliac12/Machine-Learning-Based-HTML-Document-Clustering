""" 
/*** 
os: Standard Python library for interacting with the operating system
Provides functions for file path manipulation and directory operations
Used here to build file paths and extract base file names
***/
"""
import os

""" 
/*** 
glob: Standard library module used for Unix-style pathname pattern expansion
Used here to search and list all HTML files in a given directory
***/
"""
import glob

""" 
/*** 
numpy (np): Fundamental package for scientific computing in Python
Provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions
Used here for numerical operations on similarity and distance matrices
***
"""
import numpy as np

""" 
/*** 
pandas (pd): Library providing high-performance, easy-to-use data structures and data analysis tools
Although imported here, it is not actively used in the current code
Could be useful for further data manipulation or exporting results
***/
"""
import pandas as pd

""" 
/*** 
BeautifulSoup: Library for parsing HTML and XML documents
Facilitates easy extraction of data from HTML files by navigating, searching, and modifying the parse tree
Used here to extract the main text content from HTML files
***/
"""
from bs4 import BeautifulSoup

""" 
/*** 
TfidfVectorizer: Converts a collection of raw documents to a matrix of TF-IDF features
TF-IDF stands for Term Frequency-Inverse Document Frequency, which highlights words that are more informative
Used here to transform the text extracted from HTML files into numerical feature vectors
***/
"""
from sklearn.feature_extraction.text import TfidfVectorizer

""" 
/***
cosine_similarity: Computes the cosine similarity between vectors
Measures the cosine of the angle between two vectors, which is used here to quantify how similar two documents are based on their TF-IDF vectors
***/
"""
from sklearn.metrics.pairwise import cosine_similarity

""" 
/*** 
DBSCAN (Density-Based Spatial Clustering of Applications with Noise): 
A clustering algorithm that groups together points (in this case, document vectors) that are closely packed
It does not require specifying the number of clusters beforehand and can identify outliers (points not belonging to any cluster)
Used here to cluster similar HTML documents based on their computed distance matrix
***/
"""
from sklearn.cluster import DBSCAN


def extract_text(html_path):

    """
        Extracts main text content from an HTML file
        
        Opens the HTML file using UTF-8 encoding (ignoring errors) and parses it with BeautifulSoup
        Extracts and returns the visible text from the HTML document by stripping away tags and extra whitespace
        
        Parameters:
        - html_path: The file path to the HTML document
        
        Returns:
        - A string containing the extracted text
    """
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text(separator=' ', strip=True)


def load_html_documents(directory):
    """
    Loads all HTML files from a given directory and extracts their text content
    
    Uses the glob module to find all files in the specified directory that end with '.html'
    Iterates through each file, extracting its text content using the extract_text function
    
    Parameters:
    - directory: The directory path containing HTML files

    Returns:
    - A dictionary where keys are file paths and values are the extracted text content
    """
    html_files = glob.glob(os.path.join(directory, "*.html"))
    documents = {file: extract_text(file) for file in html_files}
    return documents


def compute_similarity_matrix(documents):
    """
    Computes a TF-IDF matrix and the cosine similarity between documents
    
    Converts the extracted text content from the documents into TF-IDF feature vectors using TfidfVectorizer
    Then computes the cosine similarity between each pair of TF-IDF vectors to create a similarity matrix
    
    Parameters:
    - documents: A dictionary with file paths as keys and text content as values
    
    Returns:
    - A similarity matrix (2D NumPy array) where each entry [i][j] represents the cosine similarity between document i and document j
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents.values())
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


def cluster_documents(similarity_matrix, documents, eps=0.5, min_samples=2):
    """
    Clusters documents using the DBSCAN algorithm based on the computed similarity matrix
    
    Process:
    - Converts the cosine similarity matrix into a distance matrix using the formula: distance = 1 - similarity
    - Ensures the diagonal (self-distance) is zero
    - Uses np.maximum to replace any negative values with zero, ensuring a non-negative distance matrix
    - Applies DBSCAN clustering using the precomputed distance matrix
    - Groups document file paths into clusters, where a cluster label of -1 indicates outliers
    
    Parameters:
    - similarity_matrix: A matrix containing cosine similarity scores between documents
    - documents: A dictionary mapping file paths to their text content
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood (DBSCAN parameter)
    - min_samples: The minimum number of samples required in a neighborhood to form a cluster (DBSCAN parameter)
    
    Returns:
    - A dictionary where the keys are cluster labels and the values are lists of file paths corresponding to documents in that cluster
    """
    # Convert similarity to distance: distance = 1 - similarity
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)  # Ensure self-distance is 0
    
    # Ensure all values are non-negative by replacing negative values with 0
    distance_matrix = np.maximum(distance_matrix, 0)
    
    # Perform DBSCAN clustering using the precomputed distance matrix
    clustering = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples).fit(distance_matrix)
    
    # Group documents based on their cluster labels
    clusters = {}
    for doc_idx, cluster_id in enumerate(clustering.labels_):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(list(documents.keys())[doc_idx])
    return clusters



def main(directory):
    """
    Main function that orchestrates the loading, processing, clustering, and output display of HTML documents
    
    Steps:
    1. Loads HTML documents from the specified directory and extracts their text content
    2. Computes the TF-IDF feature vectors and the cosine similarity matrix
    3. Converts the similarity matrix into a distance matrix and applies DBSCAN clustering
    4. Prints the grouped documents in a readable format, displaying only the file names
    
    Parameters:
    - directory: The directory path containing the HTML files to process
    """
    documents = load_html_documents(directory)
    similarity_matrix = compute_similarity_matrix(documents)
    clusters = cluster_documents(similarity_matrix, documents)
    
    for cluster_id, docs in clusters.items():
        if cluster_id != -1:
            print(f"\nCluster {cluster_id}:")
            for doc in docs:
                print(f"  - {os.path.basename(doc)}")
        else:
            print("\nOutliers:")
            for doc in docs:
                print(f"  - {os.path.basename(doc)}")

    
if __name__ == "__main__":
    # Execute the main function with the path to the directory containing HTML documents
    main(r"C:\Users\Iulia\Downloads\clones 2\clones\tier1")
