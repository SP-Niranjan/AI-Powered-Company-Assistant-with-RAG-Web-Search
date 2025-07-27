from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load dataset
with open("Epsilon_Technologies_Company_Profile_Expanded.txt", "r") as f:
    dataset = [line.strip() for line in f if line.strip()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed dataset
embeddings = model.encode(dataset, convert_to_numpy=True)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = inner product = cosine if normalized
index.add(embeddings)

# Save index and dataset mapping
faiss.write_index(index, "company_faiss.index")
with open("company_mapping.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("FAISS index built and saved.")
"""
how to uplodead this entiar folder to GitHub:
1. Create a new repository on GitHub.
2. Clone the repository to your local machine.
3. Copy the entire folder containing this script and the FAISS index files into the cloned repository
4. Use the following commands to commit and push the changes:
   bash
   git add .
   git commit -m "Add FAISS index and dataset"
   git push origin main
   Description:
This script builds a FAISS index from a dataset of company profiles. It uses the SentenceTransformer
model to embed the text data into vectors, normalizes them for cosine similarity, and then creates a FAISS index for efficient similarity search. The index and the dataset mapping are saved for later use.

README.md
# FAISS Index Builder for Company Profiles
This script builds a FAISS index from a dataset of company profiles using the SentenceTransformer model for embeddings. The index allows for efficient similarity search on the embedded vectors.
## Requirements 
- Python 3.x
- sentence-transformers
- faiss-cpu or faiss-gpu
- numpy
- pickle
## Usage
1. Ensure you have the required libraries installed:
```bash
pip install sentence-transformers faiss-cpu numpy   pickle
```         
2. Place your dataset in a text file named `Epsilon_Technologies_Company_Profile_Expanded.txt`.
3. Run the script:      
```bash
python build_faiss_index.py 

    """
