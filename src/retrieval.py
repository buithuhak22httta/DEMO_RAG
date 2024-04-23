import yaml
import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.singlestoredb import SingleStoreDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from src.prompts import qa_template
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SINGLESTOREDB_URL = os.getenv("SINGLESTOREDB_URL")
BASE_URL = os.getenv("BASE_URL")

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
RETURN_SOURCE_DOCUMENTS = config.get('RETURN_SOURCE_DOCUMENTS')
VECTOR_COUNT = config.get('VECTOR_COUNT')
DATA_PATH = config.get('DATA_PATH')
CHUNK_SIZE = config.get('CHUNK_SIZE')
CHUNK_OVERLAP = config.get('CHUNK_OVERLAP')

def build_retrieval_qa(llm, prompt, vectordb):
    # dbqa = RetrievalQA.from_chain_type(llm=llm,
    #                                    chain_type='stuff',
    #                                    retriever=vectordb.as_retriever(search_kwargs={'k': VECTOR_COUNT}),
    #                                    return_source_documents=RETURN_SOURCE_DOCUMENTS,
    #                                    chain_type_kwargs={'prompt': prompt}
    #                                    )
    memory = ConversationBufferWindowMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer',
        k=5
        )
    dbqa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': VECTOR_COUNT}),
        memory=memory,
        return_source_documents=RETURN_SOURCE_DOCUMENTS,
        condense_question_prompt=prompt
    )
    return dbqa


def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    vectorstore = SingleStoreDB(embeddings, distance_strategy="DOT_PRODUCT", table_name="demo0")
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key = OPENAI_API_KEY, base_url=BASE_URL)
    qa_prompt = set_qa_prompt()
    print(qa_prompt)
    dbqa = build_retrieval_qa(llm, qa_prompt, vectorstore)

    return dbqa