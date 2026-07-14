import streamlit as st
from rag_core import build_vector_db, retrieve, build_system_prompt, chat_stream

DATA_FILE = 'philosophy-notes.txt'


@st.cache_resource(show_spinner='Embedding knowledge base (this only happens once)...')
def get_vector_db():
    return build_vector_db(DATA_FILE)


st.set_page_config(page_title='Philosophy Chatbot', page_icon='🧠')
st.title('🧠 Philosophy Chatbot')
st.caption('Ask a question and I\'ll answer using retrieved quotes and key ideas from real philosophers.')

vector_db = get_vector_db()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Replay past messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Ask me a philosophical question...')

if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    retrieved_knowledge = retrieve(vector_db, user_input)

    with st.expander('Retrieved knowledge'):
        for chunk, similarity in retrieved_knowledge:
            st.markdown(f'- (similarity: {similarity:.2f}) {chunk.strip()}')

    system_prompt = build_system_prompt(retrieved_knowledge)

    with st.chat_message('assistant'):
        full_response = st.write_stream(chat_stream(user_input, system_prompt))

    st.session_state.messages.append({'role': 'assistant', 'content': full_response})