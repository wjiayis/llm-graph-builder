from llm_graph_transformer.llm_graph_transformer import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
import os

url = os.environ.get('NEO4J_URI')
username =os.environ.get('NEO4J_USERNAME')
password = os.environ.get('NEO4J_PASSWORD')
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

loader = PyPDFLoader("/workspaces/llm-graph-builder/data/Football_news.pdf")

start_time = datetime.now()

pages = loader.load_and_split()

# Define chunking strategy
text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

# Only take the first 4 pages of the document
documents = text_splitter.split_documents(pages)

openai_api_key = os.environ.get('OPENAI_API_KEY')
llm_transformer= LLMGraphTransformer(llm_model='gpt-3.5-turbo-16k')
openai_transformer=llm_transformer.get_llm_object(api_key=openai_api_key)

graph_document=openai_transformer.extract_and_store_graph(documents=documents,graph=graph,model_version='gpt-3.5-turbo-16k')
print(graph_document)
