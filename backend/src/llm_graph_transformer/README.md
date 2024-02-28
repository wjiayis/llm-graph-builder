# llm_graph_transformer

## Purpose
The 'llm_graph_transformer' package enables the generation of knowledge graphs in a Neo4j workspace using Large language models.

## Installation
Using Poetry is a more streamlined way to manage Python dependencies and projects. To manage python dependencies in 'pyproject.toml' file install Poetry using the following command:

curl -sSL https://install.python-poetry.org | python3 -

Install Dependencies specified in the 'pyproject.toml' file using the command:

poetry install

or

pip install -r requirements.txt

## Setting up Environment Variables
Create .env file and update the following env variables.\
OPENAI_API_KEY = ""\
NEO4J_URI = ""\
NEO4J_USERNAME = ""\
NEO4J_PASSWORD = ""\
LLM_MODEL=""

## Importing Modules
To import modules and functions from 'llm_graph_transformer', you can use the following import statement:

from llm_graph_transformer.llm_graph_transformer.llm import *

## Checkout the examples/example.ipynb for usage




