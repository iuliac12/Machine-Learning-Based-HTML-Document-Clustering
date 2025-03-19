# Machine-Learning-Based-HTML-Document-Clustering

## Description

This project implements a machine-learning-based approach to cluster HTML documents based on their textual similarity. It extracts the main content from HTML files, converts them into numerical feature representations using TF-IDF (Term Frequency-Inverse Document Frequency), and applies DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to group similar documents together. The goal is to identify clusters of webpages that appear similar from a user's perspective.

## Technologies Used

 - Python: Programming language used for implementation.

 - BeautifulSoup: Extracts text from HTML documents.

 - scikit-learn:

     - TfidfVectorizer: Converts text data into numerical TF-IDF features.

     - cosine_similarity: Computes pairwise similarity between documents.

 - DBSCAN: Clustering algorithm to group similar documents.

 - NumPy: Handles similarity matrices and numerical computations.

 - os & glob: Used for file handling and retrieving HTML documents from directories.
