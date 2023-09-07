import logging

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
from fastapi import FastAPI, UploadFile
from typing import List, Optional
import time
from chromadb.config import Settings

app = FastAPI()

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
print(os.environ.get('MODEL_N_BATCH', 8))
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
source_directory = os.environ.get('SOURCE_DIRECTORY')

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


async def test_embedding():
    print('INFO - Testing the embedding mechanism')
    # Create the folder if it doesn't exist
    os.makedirs(source_directory, exist_ok=True)
    # Create a sample.txt file in the source_documents directory
    file_path = os.path.join(source_directory, "test.txt")
    with open(file_path, "w") as file:
        file.write("This is a test.")
    # Run the ingest.py command
    os.system('python ingest.py --collection test')
    # Delete the sample.txt file
    os.remove(file_path)
    logging.info('embedding working')
    print('INFO - embedding working')


# Init all the components necessary to make queries on the model
def init_model():
    print('INFO - Starting to initialize the llm')
    print('INFO - Embedding initialization')
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    print('INFO - Chroma initialization')
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    print('INFO - Retriever initialization')
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    print('INFO - Model initialization')
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    print('INFO - Retrieval initialization')
    global qa
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    logging.info('model initialize successfuly')
    print('INFO - The model has been initialize successfully')


# Starting the app with embedding and llm download
@app.on_event("startup")
async def startup_event():
    print('INFO - Startup started')
    await test_embedding()
    init_model()
    print('INFO - Startup finished')


# Example route
@app.get("/")
async def root():
    return {"message": "Hello, the APIs are now ready for your embeds and queries!"}


@app.post("/embed")
async def embed(files: List[UploadFile], collection_name: Optional[str] = None):
    saved_files = []
    # Save the files to the specified folder
    for file in files:
        file_path = os.path.join(source_directory, file.filename)
        saved_files.append(file_path)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if collection_name is None:
            # Handle the case when the collection_name is not defined
            collection_name = file.filename

    os.system(f'python ingest.py --collection {collection_name}')

    # Delete the contents of the folder
    [os.remove(os.path.join(source_directory, file.filename)) or os.path.join(source_directory, file.filename) for file
     in files]

    return {"message": "Files embedded successfully", "saved_files": saved_files}


@app.post("/retrieve")
async def query(query: str, hide_source: bool):
    # Get the answer from the chain
    start = time.time()
    res = qa(query)
    print("\n> Question:"+query)
    answer, docs = res['result'], [] if hide_source else res['source_documents']
    print("\n> Response:" + answer)
    end = time.time()
    print(f"\n> Answer (took {round(end - start, 2)} s.):")

    return {"results": answer, "docs": docs}
