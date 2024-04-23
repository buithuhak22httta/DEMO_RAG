import os
import requests
import json
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from langchain.docstore.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.singlestoredb import SingleStoreDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
import tiktoken
import yaml
import os
from src.retrieval import setup_dbqa

load_dotenv(find_dotenv())

SINGLESTOREDB_URL = os.getenv("SINGLESTOREDB_URL")

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

CHUNK_SIZE = config.get('CHUNK_SIZE')
CHUNK_OVERLAP = config.get('CHUNK_OVERLAP')
DATA_PATH = config.get('DATA_PATH')


tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def process_document():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                            model_kwargs={'device': 'cpu'})
        
        loader = DirectoryLoader(DATA_PATH,
                                glob='*.pdf',
                                loader_cls=PyPDFLoader,
                                show_progress=True)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                        chunk_overlap=CHUNK_OVERLAP,
                                                        length_function=tiktoken_len,
                                                        separators=["\n\n", "\n", " ", ""])
        texts = text_splitter.split_documents(documents)

        # os.environ["SINGLESTOREDB_URL"] = "root:root@10.1.70.212:3306/svtech"
        vectorstore = SingleStoreDB.from_documents(texts, embeddings, distance_strategy="DOT_PRODUCT", table_name="demo0")
        
    except FileNotFoundError as e:
        return f"File not found: {e}"
    except Exception as e:
        return f"Error processing documents: {e}"
    return "Chunk and embed successfully!"

# def update_prompts_file(messages_to_send):
#     prompts_file_path = os.path.join('src', 'prompts.py')
#     with open(prompts_file_path, 'r') as f:
#         file_content = f.read()

#     updated_content = file_content.replace("Context: {context}", f"Conversation: {messages_to_send}\nContext: {{context}}")

#     print(updated_content)
#     with open(prompts_file_path, 'w', encoding='utf-8') as f:
#         f.write(updated_content)

# def format_messages_to_string(messages_to_send) -> str:
#     formatted_string = ''
#     for message in messages_to_send:
#         role = message['role']
#         content = message['content']
#         if content is not None:
#             formatted_string += f"{role}: '{content}', "
#     formatted_string = formatted_string.rstrip(', ')
#     return formatted_string

def chat_completion(messages: list[dict]) -> list:
    try:
        # messages_to_send = messages[-6:] if len(messages) >= 6 else messages
        # formated_messages = format_messages_to_string(messages_to_send)
        # update_prompts_file(formated_messages)
        dbqa = setup_dbqa()
        # response = dbqa({'query': messages[-1]['content']})
        response = dbqa({'question': messages[-1]['content']})
        # return response["result"]
        result = response["answer"] + '\n**Tài liệu tham khảo:**\n' + str(response["source_documents"]) 
        # print(response["source_documents"])
        # return response["answer"]
        return result
        
    except Exception as e:
        return f'We are facing an issue: {e}'

    
def format_messages(chat_history: list[list]) -> list[dict]:
    formated_messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    for ch in chat_history:
        formated_messages.append(
            {
                "role": "user",
                "content": ch[0]
            }
        )
        formated_messages.append(
            {
                "role": "assistant",
                "content": ch[1]
            }
        )
    formated_messages.append(
            {
                "role": "user",
                "content": chat_history[-1][0]
            }
        )
    return formated_messages

def generate_response(text: str, chatbot: list[list]) -> tuple:
    chatbot.append([text, None])
    formated_messages = format_messages(chatbot)
    # print(formated_messages)
    # print('------------')
    response = chat_completion(formated_messages)
    # print(response)
    # print('------------')
    chatbot[-1][1] = response
    # print(text)
    print(chatbot)
    return '', chatbot
