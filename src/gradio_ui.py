import gradio as gr
from pathlib import Path
from src.utils import generate_response, process_document
import shutil
import os
from pathlib import Path
from threading import Thread

def upload_file(files):
    UPLOAD_FOLDER = "./data"    
    if not os.path.exists(UPLOAD_FOLDER):    
        os.mkdir(UPLOAD_FOLDER)
    for file in files:    
        shutil.copy(file, UPLOAD_FOLDER)    
    gr.Info("File Uploaded!!!")
    if files is not None:
        file_paths = [file.name for file in files]
        return file_paths
    else:
        return "No file uploaded" 

def run_process():
    result = process_document()
    gr.Info(result)

with gr.Blocks() as demo:
    gr.Markdown('# Q&A Bot with SVTECHGPT')
    with gr.Tab("Knowledge Bot"):
        chatbot = gr.components.Chatbot(label='SVTECH Assistant')
        msg = gr.components.Textbox(label='Input query')
        # clear = gr.components.Button(value='Clear', variant='stop')
        clear = gr.ClearButton([msg, chatbot])
        msg.submit(
            fn=generate_response,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot])
        
    with gr.Tab("Input Text Document"):
        file_output = gr.File()
        upload_button=gr.UploadButton("Upload file", file_types=[".pdf",".csv",".docx"], file_count="multiple")
        files_running = upload_button.upload(upload_file, upload_button, file_output)
        process_button = gr.Button("Chunk & Embed")
        # def process_files():
        #     progress_box.value = process_document()
        process_button.click(run_process)


    

# if __name__ == '__main__':
#     demo.launch()