## Grant Matching Script Notes

This script enables the matching of a project description to multiple grant opportunities using semantic similarity. 
It uses a pretrained language model (all-MiniLM-L6-v2) to evaluate how closely a project's content aligns with the textual descriptions of various grants. 
The comparison is achieved by transforming each document into a vector representation (embedding) and computing similarity scores using cosine similarity.

## Function overview:

1. upload_project_description()

Prompts the user to upload a single .txt file containing the project description.
Accepts only one .txt file per execution
Reads the file's content and stores it in a dictionary format

2. upload_grant_texts()

Allows the user to upload multiple .txt files, with each file representing a different grant opportunity.
Allows multiple text files to be uploaded simultaneously.
Extracts the content of each txt file and uses the filename as the grant identifier.

Returns a dictionary like e.g. { "grant1.txt": "Grant 1 description text",...}

3. chunk_text(text, chunk_size=150, overlap=30)

all-MiniLM-L6-v2 converts texts into a embedding vector of shape (,384) and accepts truncates long texts into 256 tokens. 
Cosequently the sentence transformer model accepts short sentences and works very well. I explored jinaai/jina-embeddings-v2-base-en model which could work for upto 5000 words
per .txt file but these models require more time to load and did not perform well on shorter text. The model all-MiniLM-L6-v2 has superior performnce, therefore for longer text description input,
I decided to convert them into overlapping chunks (chunk size 150, 30 overlapping words). It does the following:


Splits the text into segments of chunk_size words (default: 150).
Applies an overlap of 30 words between chunks to retain contextual continuity.
Useful for long documents that exceed the model’s maximum input length.

4. encode_with_chunking(model, texts, chunk_size=150)

Generates semantic embeddings for each text input, handling both short and long documents.

For shorter text it encodes directly using the sentence transformer model.

For long text:

	Split into chunks using chunk_text().
	Each chunk is encoded separately.
	All chunk embeddings are averaged into one single vector.

Output: (number_of_texts, 384), where 384 is the dimensionality of each embedding vector.

5. calculate_similarities(projects, grants, model_name='all-MiniLM-L6-v2')

Calculates similarity scores between each project description and each grant using cosine similarity.
Loads the pretrained model.
Uses encode_with_chunking() to convert all project and grant texts into embeddings.
Applies cosine_similarity() to generate a similarity matrix.
Returns dataframe with rows as project description, coloumn as grants, and values as similarity score.

6. display_results(similarity_df, top_n=5)

Displays the top N matching grants for each uploaded project description.
Sorts similarity scores in descending order.
Categorizes each result:
	High Similarity: Score ≥ 0.80
	Moderate Similarity: Score ≥ 0.60 and < 0.80
	Low Similarity: Score < 0.60
	
7. display_similarity_matrix(similarity_df)

Prints the full similarity matrix showing similarity scores between all project and grant combinations.

8. main()

Driver function for the full workflow execution
