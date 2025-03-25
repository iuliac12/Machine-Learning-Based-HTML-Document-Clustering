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

 ## How It Works

 - Extract Text from HTML: The script parses HTML files and extracts their visible text content.

 - Compute TF-IDF Features: It converts the extracted text into a numerical feature matrix using TF-IDF.

 - Calculate Similarity: Computes the cosine similarity between document vectors.

 - Clustering with DBSCAN: Converts similarity into a distance matrix and applies DBSCAN to cluster similar documents. Output Clusters: Displays the clustered HTML documents, showing which files belong to the same group.

 ## Running the Project
 
To execute the script, run:

- python script.py <path_to_html_directory>

Replace <path_to_html_directory> with the folder containing the HTML files to be clustered.

<img src="https://github.com/user-attachments/assets/c6a16362-ef6a-4f87-8fb0-97968ab4d8ef" width="300" height="600">
