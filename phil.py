import ollama

DATA_FILE = 'philosophy-notes.txt'

phil_data = []
with open(DATA_FILE, 'r') as file:
    phil_data = file.readlines()
print(f'Loaded {len(phil_data)} entries')

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

VECTOR_DB = []

def add_chunk_to_db(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(phil_data):
    add_chunk_to_db(chunk)
    print(f'Added chunk {i+1}/{len(phil_data)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]
    
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

def ask(input_query, top_n=3):
    retrieved_knowledge = retrieve(input_query, top_n=top_n)
 
    print('Retrieved knowledge: ')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk.strip()}')
 
    context = '\n'.join(f' - {chunk.strip()}' for chunk, _ in retrieved_knowledge)
    system_prompt = PHILOSOPHER_SYSTEM_PROMPT.format(context=context)
 
    print('\nChatbot response: ')
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print()
 
 
if __name__ == '__main__':
    input_query = input('Ask me a philosophical question: ')
    ask(input_query)
