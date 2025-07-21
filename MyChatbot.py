import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

OpenAI="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"

st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file=st.file_uploader("Upload notes PDF and start making questions",type="pdf")
if file is not None:
    my_pdf=PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text+=page.extract_text()
        #st.write(text)

    #break it into chunks
    splitter=RecursiveCharacterTextSplitter(separators=["\n"],chunk_size=250,chunk_overlap=50)
    chunks=splitter.split_text(text)
    #st.write(chunks)

    embeddings=OpenAIEmbeddings(api_key=OpenAI)

    vector_store=FAISS.from_texts(chunks,embeddings)

user_query=st.text_input("Type Your Query here")

if user_query:
    matching_chunks=vector_store.similarity_search(user_query)
    llm=ChatOpenAI(
        api_key=OpenAI,
        max_tokens=300,
        temperature=0,
        model="gpt-3.5-turbo"
    )
    chain=load_qa_chain(llm,chain_type="stuff")
    output=chain.run(question=user_query,input_documents=matching_chunks)
    st.write(output)


