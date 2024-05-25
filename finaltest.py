import os
import subprocess
import uuid
import shutil
import time
import json
import nbformat
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, PythonLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def clone_repo(url: str, path: str = '.'):
    repo_name = url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(path, repo_name)
    if os.path.exists(repo_path):
        st.success(f"Repository {url} already exists. Skipping clone step.")
        return repo_path
    try:
        subprocess.check_call(['git', 'clone', url], cwd=path)
        st.success(f"Repository {url} cloned successfully!")
        return repo_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred while trying to clone {url}. Error: {str(e)}")
        return None

def load_and_index_files(repo_path):
    extensions = ['txt', 'md', 'rst', 'py', 'js', 'html', 'yaml', 'yml', 'ini', 'cfg', 'conf', 'sh', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'cpp', 'java', 'c']
    documents_dict = {}
    readme_doc = None
    readme_files = [os.path.join(root, file) for root, _, files in os.walk(repo_path) for file in files if file.lower().startswith('readme')]
    for file_path in readme_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                relative_path = os.path.relpath(file_path, repo_path)
                file_id = str(uuid.uuid4())
                doc = Document(content, {
                    "source": relative_path,
                    "file_id": file_id,
                    "file_type": "readme"
                })
                documents_dict[file_id] = doc
                readme_doc = doc
               
        except Exception as e:
            st.error(f"Error loading README file '{file_path}': {e}")
 
    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            if ext == 'py':
                loader = DirectoryLoader(repo_path, glob=glob_pattern, recursive=True, loader_cls=PythonLoader)
            
          
               
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern, recursive=True)

            if loader:
                loaded_documents = loader.load() if callable(loader.load) else []
                for doc in loaded_documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata['source'] = relative_path
                    doc.metadata['file_id'] = file_id
                    documents_dict[file_id] = Document(doc.page_content, doc.metadata)
                    
        except Exception as e:
            st.error(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue
 
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.ipynb'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        notebook_content = nbformat.read(f, as_version=4)
                        cells = [cell['source'] for cell in notebook_content.cells if cell.cell_type in ['code', 'markdown']]
                        content = "\n".join(cells)
                        relative_path = os.path.relpath(file_path, repo_path)
                        file_id = str(uuid.uuid4())
                        doc = Document(content, {
                            "source": relative_path,
                            "file_id": file_id,
                            "file_type": "ipynb"
                        })
                        documents_dict[file_id] = doc
                        
                except Exception as e:
                    st.error(f"Error loading notebook '{file_path}': {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    split_documents = []

    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']
            split_doc.metadata['source'] = original_doc.metadata['source']
            split_doc.metadata['file_type'] = original_doc.metadata.get('file_type', 'unknown')
            split_doc.metadata['doc_title'] = original_doc.metadata.get('doc_title', '')
        split_documents.extend(split_docs)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
 
    chroma_db_dir = f"./chroma_db/{os.path.basename(repo_path)}"
    if os.path.exists(chroma_db_dir):
      
        retries = 5
        for i in range(retries):
            try:
                shutil.rmtree(chroma_db_dir)  
                break
            except PermissionError:
                if i < retries - 1:
                    time.sleep(1)   
                else:
                    st.error(f"Could not delete Chroma DB directory after {retries} attempts.")
                    return None, None

    chroma_db = Chroma.from_documents(split_documents, embeddings, persist_directory=chroma_db_dir)
    chroma_db.persist()

    return chroma_db, embedding_model, readme_doc, documents_dict

def search_documents(query, chroma_db, embedding_model, n_results=5):
    query_embedding = embedding_model.encode(query).tolist()  # Ensure the embedding is a list of floats
    results = chroma_db.similarity_search_by_vector(query_embedding, k=n_results)
    
    return results

def ask_question(question, context):
    chroma_db, model_local, repo_name, repo_url, conversation_history, embedding_model, readme_doc, documents_dict = context

    relevant_docs = search_documents(question, chroma_db, embedding_model, n_results=5)
 
    specific_file_name = None
    specific_file_ext = None
    question_terms = question.lower().split()
 

    if specific_file_name is None:
        for term in question_terms:
            for ext in ['py', 'ipynb', 'pdf''txt', 'md', 'rst', 'py', 'js', 'html', 'yaml', 'yml', 'ini', 'cfg', 'conf', 'sh', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'cpp', 'java', 'c']:
                if term.endswith(f'.{ext}'):
                    print(term)
                    specific_file_name = term
                    specific_file_ext = ext
                    break
            if specific_file_name:
                break
    print(specific_file_name)
    specific_file_doc = None
    if specific_file_name:
         
        if specific_file_ext:
            specific_file_name_with_ext = f"{specific_file_name}"
            print(specific_file_name_with_ext)
            for doc in documents_dict.values():
                if specific_file_name_with_ext in doc.metadata['source'].lower():
                    specific_file_doc = doc

                    break
    else:
         
        for doc in documents_dict.values():
            for term in question_terms:
                first_split =doc.metadata['source'].lower() .split('.', 1)[0]
                if term in first_split.split('\\')[-1]:
                    print(term)
                    specific_file_doc = doc
                    
   
    if specific_file_doc and specific_file_doc not in relevant_docs:
        relevant_docs.insert(0, specific_file_doc)   
    if readme_doc and readme_doc not in relevant_docs:
        relevant_docs.insert(0, readme_doc)  
 
    all_docs = relevant_docs
    numbered_documents = "\n".join([f"{i+1}. {doc.metadata['source']}" for i, doc in enumerate(all_docs)])
    question_context = f"This question is about the GitHub repository '{repo_name}' available at {repo_url}. The most relevant documents are:\n\n{numbered_documents}"

    detailed_context = "\n\n".join([f"Document: {doc.metadata['source']}\n\n{doc.page_content}" for doc in all_docs])

    conversation_history.append(f"User: {question}")
    history_text = "\n".join(conversation_history)


    after_rag_template = """
    The conversation so far is:
    {history}

    Answer the question based only on the following context:
    {context}
    Detailed context:
    {detailed_context}

    If you do not have context about something, return 'I do not have context about that.'

    Let's work this out in a step-by-step way to be sure we have the right answer.

    Question: {question}
    """

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    retriever = lambda inputs: {
        "history": inputs["history"],
        "context": inputs["context"],
        "detailed_context": inputs["detailed_context"],
        "question": inputs["question"]
    }

    after_rag_chain = (
        {"history": RunnablePassthrough(), "context": lambda _: question_context, "detailed_context": lambda _: detailed_context, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    answer = after_rag_chain.invoke({"history": history_text, "context": question_context, "detailed_context": detailed_context, "question": question})
    conversation_history.append(f"Bot: {answer}")

    return answer

def main():
    st.title("GitHub Repo Search and Code Chat")
    st.write("Clone a GitHub repo, extract content, and query using RAG pipeline with conversation history.")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'context' not in st.session_state:
        st.session_state.context = None
 
    with st.form(key='clone_form'):
        repo_url = st.text_input('Enter the GitHub repo URL:')
        clone_button = st.form_submit_button('Clone Repo')

    if clone_button:
        if repo_url:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = clone_repo(repo_url)
            if repo_path:
                with st.spinner('Processing...'):
                    chroma_db, embedding_model, readme_doc, documents_dict = load_and_index_files(repo_path)
                    if chroma_db is None:
                        return
                    if not chroma_db._collection.count():
                        st.error("No documents were found to index. Please check the repository content and try again.")

                    model_local = Ollama(model="llama2")
                    st.session_state.context = (chroma_db, model_local, repo_name, repo_url, st.session_state.conversation_history, embedding_model, readme_doc, documents_dict)
                    st.success("Repository cloned and indexed successfully! Now you can ask questions.")
 
    if st.session_state.context:
        question = st.text_input("Enter your question")
        if st.button('Ask'):
            if question:
                answer = ask_question(question, st.session_state.context)
                st.text_area("Answer", value=answer, height=300, disabled=True)
            else:
                st.error("Please enter a question.")

if __name__ == "__main__":
    main()
