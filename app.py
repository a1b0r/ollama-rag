import gradio as gr  # Gradio library for creating web UIs for Python applications
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader  # document loaders for web and PDF sources
from langchain_community.vectorstores import Chroma  # the Chroma vector store for document embeddings
from langchain_community import embeddings  # embeddings module for generating embeddings
from langchain_community.chat_models import ChatOllama  # ChatOllama, a chat model for generating responses
from langchain_core.runnables import RunnablePassthrough  # RunnablePassthrough for passing arguments as-is in pipelines
from langchain_core.output_parsers import StrOutputParser  # StrOutputParser for parsing outputs to strings
from langchain_core.prompts import ChatPromptTemplate  # ChatPromptTemplate for creating chat prompts
from langchain.output_parsers import PydanticOutputParser  # PydanticOutputParser for parsing outputs using Pydantic models
from langchain.text_splitter import CharacterTextSplitter  # CharacterTextSplitter for splitting text based on character count

def process_input(urls, question):
    llm = ChatOllama(model="llama2", temperature=0.3)  # Initialize ChatOllama model

    # Convert string of URLs to list by splitting on new lines
    urls_list = urls.split("\n")

    # Load documents from URLs using WebBaseLoader and compile them into a list
    docs = [WebBaseLoader(url).load() for url in urls_list]

    # Flatten the list of lists into a single list
    docs_list = [item for sublist in docs for item in sublist]
    
    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=8000, chunk_overlap=100)
    
    # Split the loaded documents into chunks
    doc_splits = text_splitter.split_documents(docs_list)

    # Create a Chroma vector store from the document splits, using Ollama embeddings
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-ollama",
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    # Convert the Chroma vector store into a retriever
    retriever = vectorstore.as_retriever()

    # Template for generating responses, emphasizing brevity
    rag_template = """Answer the question based exclusively on the following context:
    {context}.
    Question: {question}
    """

    # Create a chat prompt template from the template string
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    # Chain the components to create a processing pipeline
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # Pass context and question to the pipeline
        | rag_prompt  # Apply the chat prompt template
        | llm  # Pass the result to the LLM for processing
        | StrOutputParser()  # Parse the model's output to a string
    )
    # Invoke the processing chain with the input question and return the result
    return after_rag_chain.invoke(question)

# Define a Gradio interface with specified inputs and output
iface = gr.Interface(fn=process_input,
                     inputs=[gr.Textbox(label="Enter URLs separated by new lines"), 
                             gr.Textbox(label="Question")],
                     outputs="text",
                     title="Document Query with Ollama",
                     description="Enter URLs and a question to query the documents.")
iface.launch()  # Launch the Gradio interface
