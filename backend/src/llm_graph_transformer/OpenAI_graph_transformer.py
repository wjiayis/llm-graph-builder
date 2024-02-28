from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv 
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from typing import List, Optional
from langchain.pydantic_v1 import Field, BaseModel
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging
import re
from .prompt import *

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(message)s',level='INFO')

class Property(BaseModel):
  """A single property consisting of key and value"""
  key: str = Field(..., description="key")
  value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )
class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )
  
    
def format_property_key(s: str) -> str: 
    """
     Formats a property key to make it easier to read. 
     This is used to ensure that the keys are consistent across the server
     
     Args:
     	 s: The string to format.
     
     Returns: 
     	 The formatted string or the original string if there was no
    """
    
    words = s.split()
    logging.debug("Returns the word if words is not empty.")
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def props_to_dict(props) -> dict:
    """
     Convert properties to a dictionary. 
     This is used to convert a list of : class : ` Property ` objects to a dictionary
     
     Args:
     	 props: List of : class : ` Property ` objects
     
     Returns: 
     	 Dictionary of property keys and values or an empty dictionary
    """
    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties


def map_to_base_node(node: Node) -> BaseNode:
    """
     Map KnowledgeGraph Node to the Base Node. 
     This is used to generate Cypher statements that are derived from the knowledge graph
     
     Args:
     	 node: The node to be mapped
     
     Returns: 
     	 A mapping of the KnowledgeGraph Node to the BaseNode
    """
    properties = props_to_dict(node.properties) if node.properties else {}
    properties["name"] = node.id.title().replace(' ','_')
    #replace all non alphanumeric characters and spaces with underscore
    node_type = re.sub(r'[^\w]+', '_', node.type.capitalize())
    return BaseNode(
        id=node.id.title().replace(' ','_'), type=node_type, properties=properties
    )


def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """
     Map KnowledgeGraph relationships to the base Relationship. 
    
     Args:
     	 rel: The relationship to be mapped
     
     Returns: 
     	 The mapped : class : ` BaseRelationship ` 
    """
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )  

def get_extraction_chain(
    model_version,
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None,
    openai_api_key=None
    ):
    
    """
    Get a chain of nodes and relationships to extract from GPT. 
    This is an interactive function that prompts the user to select which nodes and relationships 
    to extract from the Knowledge Graph and returns them as a list of Node objects.
    
    Args:
     	 Optional allowed_nodes: A list of node IDs to be considered for extraction. 
         Optional allowed_rels: A list of relationships to be considered for extraction. 
         
         If None ( default ) all nodes are considered. If a list of strings is provided only nodes that match the string will be considered.
    """

    llm = ChatOpenAI(model= model_version, temperature=0,openai_api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages(get_prompt(allowed_nodes,allowed_rels))
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)




class OpenaiGraphTransformer:

    def __init__(self,openai_api_key: Optional[str] = None):
        self.openai_api_key=openai_api_key

    def extract_and_store_graph(self,
        model_version,
        graph: Neo4jGraph,
        documents: List[Document],
        nodes:Optional[List[str]] = None,
        rels:Optional[List[str]]=None) -> None:
        
        """
        This is a wrapper around OpenAI functions to perform the extraction and 
        store the result into a Neo4jGraph.
        
        Args:
            graph: Neo4j graph to store the data into
            document: Langchain document to extract data from
            nodes: List of nodes to extract ( default : None )
            rels: List of relationships to extract ( default : None )
        
        Returns: 
            The GraphDocument that was extracted and stored into the Neo4jgraph
        """
        graph_documents_list=[]

        for document in documents:
            extract_chain = get_extraction_chain(model_version,nodes, rels,openai_api_key=self.openai_api_key)
            data = extract_chain.invoke(document.page_content)['function']

            graph_document = [GraphDocument(
            nodes = [map_to_base_node(node) for node in data.nodes],
            relationships = [map_to_base_relationship(rel) for rel in data.rels],
            source = document
            )]

            graph.add_graph_documents(graph_document)

            graph_documents_list.extend(graph_document)
        return graph_documents_list   
 
