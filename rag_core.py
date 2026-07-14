import ollama

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
CHAT_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

PHILOSOPHER_SYSTEM_PROMPT = """You are a thoughtful philosophy chatbot. You discuss
philosophical ideas, arguments, and thought experiments in a clear, engaging way.

Use the following context passages as your primary source of ideas when they are
relevant. If the context doesn't cover the question, you may reason more broadly
using general philosophical knowledge, but say so explicitly rather than presenting
it as coming from the provided sources.

Context:
{context}
"""


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)


def build_vector_db(data_file):
    """Read chunks from data_file and embed each one. Returns a list of (chunk, embedding)."""
    with open(data_file, 'r') as file:
        chunks = file.readlines()

    vector_db = []
    for chunk in chunks:
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        vector_db.append((chunk, embedding))
    return vector_db


def retrieve(vector_db, query, top_n=3):
    """Return the top_n (chunk, similarity) pairs most relevant to query."""
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in vector_db:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def build_system_prompt(retrieved_knowledge):
    """Format the system prompt with the retrieved context passages."""
    context = '\n'.join(f' - {chunk.strip()}' for chunk, _ in retrieved_knowledge)
    return PHILOSOPHER_SYSTEM_PROMPT.format(context=context)


def chat_stream(input_query, system_prompt):
    """Yields streaming response chunks from the chat model."""
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )
    for chunk in stream:
        yield chunk['message']['content']