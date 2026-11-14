1: 
DOCUMENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUERY_MODEL = "intfloat/e5-small-v2"
...
document_encoder = SentenceTransformer(DOCUMENT_MODEL)
query_encoder = SentenceTransformer(QUERY_MODEL)

When running the vectoarization for the encoder, if we use 2 different models this breakes, so best solution is to switch to one and the same
What we did
EMBEDDING_MODEL = "intfloat/e5-small-v2"
encoder = SentenceTransformer(EMBEDDING_MODEL)

2: The e5 model was missing its prefixes
What we did
Added prefixies for better presition

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
...
[f"{PASSAGE_PREFIX}{document.title}\n{document.text}" for document in documents]
[f"{QUERY_PREFIX}{question['question']}" for question in questions]

3.Documents were cut into 180-character pieces
CHUNK_SIZE = 180
for start in range(0, len(searchable_text), CHUNK_SIZE):
    chunks.append(Chunk(document.document_id, searchable_text[start : start + CHUNK_SIZE]))

Remove the whole Chunkolization because we had document cutoffs

4: Archived notices look almost identical to current ones

For every exhibit and city, the corpus has two near-twins:
The title names the exhibit and city, e.g. “CURRENT ACCESS NOTICE: Solar Spectroscope, Arusha”.
The collection desk and time are in the third paragraph, several hundred characters later.

This error was solved by this function

def is_current_notice(document: Document) -> bool:
    return document.title.upper().startswith(CURRENT_NOTICE_PREFIX)
...
current_mask = np.array([is_current_notice(document) for document in documents])
similarities[:, ~current_mask] = -np.inf   # archived/general docs can never win
