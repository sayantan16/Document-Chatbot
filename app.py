import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdfReader = PdfReader(pdf)
    for page in pdfReader.pages:
      text += page.extract_text()
  return text


def get_text_chunks(raw_text):
  text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
  )
  chunks = text_splitter.split_text(raw_text)
  return chunks


def get_vectorstore(text_chunks):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore


def get_conversation_chain(vectoreStore):
  llm = ChatOpenAI()
  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectoreStore.as_retriever(),
    memory = memory
  )
  return conversation_chain


def handle_user_input(user_question):
  response = st.session_state.conversation({'question': user_question})
  st.session_state.chat_history = response["chat_history"]

  for i, message in enumerate(st.session_state.chat_history):
    if i%2 == 0:
      st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
      st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
  load_dotenv()
  st.set_page_config(page_title='Documentation ChatBot', page_icon=':books:')
  st.write(css, unsafe_allow_html=True)
  
  if "conversation" not in st.session_state:
    st.session_state.conversation = None

  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

  st.header("Documentation ChatBot :books:")
  user_question = st.text_input("Ask a question about your documents...")

  if user_question:
    handle_user_input(user_question)

  # st.write(user_template.replace("{{MSG}}", "Hi!!"), unsafe_allow_html=True)
  # st.write(bot_template.replace("{{MSG}}", "Hello, How can I help you today?"), unsafe_allow_html=True)
  
  with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader("Upload you pdfs here and click Process...", accept_multiple_files=True)
    
    if st.button("Process"):
      with st.spinner("processing..."):
        
        # get pdf text
        raw_text = get_pdf_text(pdf_docs)

        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create the vector storage with embeddings
        vectoreStore = get_vectorstore(text_chunks)

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vectoreStore)



if __name__ == '__main__':
  main()