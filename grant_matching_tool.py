# Grant-Project Matching Tool 
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tkinter import Tk, filedialog

def select_txt_file(single=True):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if single:
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        return [file_path] if file_path else []
    else:
        return filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])

def upload_project_description():
    print("\nSTEP 1: Select Project Description File (.txt)")
    paths = select_txt_file(single=True)
    if not paths:
        raise ValueError("No project file selected.")
    
    with open(paths[0], 'r', encoding='utf-8') as f:
        text = f.read().strip()
    print(f"Loaded project file: {os.path.basename(paths[0])}")
    return {os.path.basename(paths[0]): text}

def upload_grant_texts():
    print("\nSTEP 2: Select Multiple Grant Files (.txt)")
    paths = select_txt_file(single=False)
    if not paths:
        raise ValueError("No grant files selected.")

    grants = {}
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                grants[os.path.basename(path)] = text

    print(f"Successfully processed {len(grants)} grant descriptions.")
    return grants

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def encode_with_chunking(model, texts, chunk_size=150):
    all_embeddings = []
    for idx, text in enumerate(texts):
        word_count = len(text.split())
        print(f"\nEncoding Text {idx + 1}: {word_count} words")

        if word_count <= chunk_size:
            print("  Short text: encoding directly")
            embedding = model.encode([text])
            all_embeddings.append(embedding[0])
        else:
            chunks = chunk_text(text, chunk_size)
            print(f"  Long text: split into {len(chunks)} chunks")
            chunk_embeddings = model.encode(chunks)
            avg_embedding = np.mean(chunk_embeddings, axis=0)
            all_embeddings.append(avg_embedding)

    return np.array(all_embeddings)

def calculate_similarities(projects, grants, model_name='all-MiniLM-L6-v2'):
    print("\nCalculating semantic similarities...")
    model = SentenceTransformer(model_name)

    project_names = list(projects.keys())
    grant_names = list(grants.keys())
    project_texts = [projects[name] for name in project_names]
    grant_texts = [grants[name] for name in grant_names]

    project_embeddings = encode_with_chunking(model, project_texts)
    grant_embeddings = encode_with_chunking(model, grant_texts)
    similarity_matrix = cosine_similarity(project_embeddings, grant_embeddings)

    return pd.DataFrame(similarity_matrix, index=project_names, columns=grant_names)

def display_results(similarity_df, top_n=5):
    print(f"\nTop {top_n} matching grants per project\n" + "=" * 60)
    for project in similarity_df.index:
        print(f"\nProject: {project}\n" + "-" * 50)
        sorted_scores = similarity_df.loc[project].sort_values(ascending=False)

        for i, (grant, score) in enumerate(sorted_scores.head(top_n).items()):
            level = "High" if score >= 0.8 else "Moderate" if score >= 0.6 else "Low"
            print(f"{level} Similarity - Rank {i+1}: {grant}")
            print(f"   Similarity Score: {score:.4f}")

def display_similarity_matrix(similarity_df):
    print("\nFull Similarity Matrix (Projects vs. Grants)\n" + "-" * 40)
    print(similarity_df.round(3))
    
def main():
    print("Grant-Project Matching Tool\n" + "=" * 60)
    projects = upload_project_description()
    grants = upload_grant_texts()

    print("\nFiles loaded successfully.")
    print(f"  Number of projects: {len(projects)}")
    print(f"  Number of grants:   {len(grants)}")

    similarity_df = calculate_similarities(projects, grants)
    display_results(similarity_df)

    show = input("\nWould you like to view the full similarity matrix? (y/n): ").strip().lower()
    if show == 'y':
        display_similarity_matrix(similarity_df)

# if __name__ == "__main__":
#     main()
main()