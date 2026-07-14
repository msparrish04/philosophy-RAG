from rag_core import build_vector_db, retrieve, build_system_prompt, chat_stream

DATA_FILE = 'philosophy-notes.txt'

def ask(vector_db, input_query, top_n=3):
    retrieved_knowledge = retrieve(vector_db, input_query, top_n=top_n)
 
    print('Retrieved knowledge: ')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk.strip()}')
 
    system_prompt = build_system_prompt(retrieved_knowledge)

    print('\nChatbot response: ')
    for text in chat_stream(input_query, system_prompt):
        print(text, end='', flush=True)
    print()
 
 
if __name__ == '__main__':
    print('Loading knowledge base...')
    vector_db = build_vector_db(DATA_FILE)
    print(f'Loaded and embedded {len(vector_db)} chunks')
 
    input_query = input('Ask me a philosophical question: ')
    ask(vector_db, input_query)