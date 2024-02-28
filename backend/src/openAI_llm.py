from src.llm_graph_transformer.OpenAI_graph_transformer import *
from tqdm import tqdm
from src.make_relationships import create_source_chunk_entity_relationship
import logging,os
from langchain_community.graphs import Neo4jGraph
from typing import List
from langchain.schema import Document
def extract_graph_from_OpenAI(model_version,
                            graph: Neo4jGraph,
                            chunks: List[Document],
                            file_name : str,
                            uri : str,
                            userName : str,
                            password : str):
    """
        Extract graph from OpenAI and store it in database. 
        This is a wrapper for extract_and_store_graph
                                
        Args:
            model_version : identify the model of LLM
            graph: Neo4jGraph to be extracted.
            chunks: List of chunk documents created from input file
            file_name (str) : file name of input source
            uri: URI of the graph to extract
            userName: Username to use for graph creation ( if None will use username from config file )
            password: Password to use for graph creation ( if None will use password from config file )    
        Returns: 
            List of langchain GraphDocument - used to generate graph
    """
    graph_document_list = []
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    openai_transformer= OpenaiGraphTransformer(openai_api_key=openai_api_key)
    logging.info(f"create relationship between source,chunk and entity nodes created from {model_version}")
    for i, chunk_document in tqdm(enumerate(chunks), total=len(chunks)):
        graph_document=openai_transformer.extract_and_store_graph(model_version=model_version,graph=graph,documents=[chunk_document])
        create_source_chunk_entity_relationship(file_name,graph,graph_document,chunk_document,uri,userName,password)
        graph_document_list.append(graph_document[0])     
    return graph_document_list