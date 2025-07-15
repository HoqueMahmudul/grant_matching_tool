## Grant Matching Script Notes

This script matches a project description to multiple grant opportunities using semantic similarity.  
It employs a pretrained language model (**all-MiniLM-L6-v2**) to evaluate how closely a project's content aligns with various grant descriptions.  
The comparison is based on vector embeddings and cosine similarity.

---

## Function Overview

### 1. `upload_project_description()`

- Prompts the user to upload **one `.txt` file** containing the project description.
- Reads the file content and stores it in a dictionary.
- Accepts only **one file per execution**.

---

### 2. `upload_grant_texts()`

- Allows uploading of **multiple `.txt` files** (one for each grant).
- Uses filenames as grant identifiers.
- Returns a dictionary like:  
  `{ "grant1.txt": "Grant 1 description text", ... }`

---

### 3. `chunk_text(text, chunk_size=150, overlap=30)`

- **all-MiniLM-L6-v2** works best on shorter texts (up to 256 tokens).  
- Longer descriptions are split into **overlapping chunks** for better context retention.
- Default: **150 words per chunk** with **30 words overlap**.
- Ensures the model handles long texts without truncation.

---

### 4. `encode_with_chunking(model, texts, chunk_size=150)`

- Encodes text inputs into semantic embeddings.
- Short texts are encoded directly.
- Long texts are:
  - Split into chunks.
  - Each chunk encoded individually.
  - Chunk embeddings averaged into a single vector.
- Output shape: **(number_of_texts, 384)**.

---

### 5. `calculate_similarities(projects, grants, model_name='all-MiniLM-L6-v2')`

- Computes cosine similarity between each project description and each grant.
- Loads the pretrained model.
- Uses `encode_with_chunking()` for embedding.
- Returns a **DataFrame** with similarity scores.

---

### 6. `display_results(similarity_df, top_n=5)`

- Displays the **top N matching grants** for each project.
- Categorizes similarity scores:
  - **High Similarity**: ≥ 0.80
  - **Moderate Similarity**: 0.60 – 0.79
  - **Low Similarity**: < 0.60

---

### 7. `display_similarity_matrix(similarity_df)`

- Prints the full similarity matrix.

---

### 8. `main()`

- Runs the complete workflow.

---

