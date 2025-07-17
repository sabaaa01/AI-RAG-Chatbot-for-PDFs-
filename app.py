import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings



with st.sidebar:
    st.title("Chat with your pdf")
    st.markdown('''

  '''  )
    st.write("\n" * 5)  # Adds 5 blank lines
    st.write('Made with by Saba')

def main():
    st.header("Chat with pdf")
    load_dotenv()
    #upload a file
    pdf=st.file_uploader("Upload Your PDF" , type= 'pdf')
    # st.write(pdf.file)
    if pdf is not None:
         pdf_reader=PdfReader(pdf)
         #st.write(pdf_reader)

         text = ""
         for page in pdf_reader.pages:
              text += page.extract_text()
         #st.write(text) 

         #split into chunks
         text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=1000,
              chunk_overlap=200,
              length_function=len
              )
         chunks = text_splitter.split_text(text=text)
         #st.write(chunks)
  
        #embeddings- 
         store_name = pdf.name[:-4]


         if os.path.exists(f"{store_name}.pkl"):
               with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
               st.write('Embeddings loaded from the disk')
         else:
              embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
              VectorStore = FAISS.from_texts(chunks, embedding= embeddings)
              with open(f"{store_name}.pkl","wb") as f:
                     pickle.dump(VectorStore, f)
              st.write('Embeddings completed')

              #accept user questions
         query = st.text_input("Ask a question about the pdf")
   

         if query:
                docs= VectorStore.similarity_search(query=query, k=3)
                llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",  # Alternative model
                model_kwargs={"temperature": 0.5, "max_length": 512}
                )
                chain= load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question= query)
                #st.write(docs)
                st.write(response)


if __name__ == "__main__":
        main()
