import logging

import time

import yaml

import pymilvus

from pymilvus.client.types import ExtraList

logger = logging.getLogger("feature_api")

class VectorClassBase:

def __init__(self, config) -> None:

# init config

self.tenant = config['tenant']

self.database = config['db_name']

self.collection_name = config['collection_name']

self.host = config['host']

self.port = config['port']

self.uri = f"{config['protocol']}://{self.host}:{self.port}"

self.token = config['token']

self.dim = config['dim']

class Milvus(VectorClassBase):

get_output_fields = ["documents", "code", "name_en", "name_cn"]

search_output_fields = ["documents", "code", "name_en", "name_cn"]

def __init__(self, config) -> None:

super().__init__(config)

# init config

self.client = pymilvus.MilvusClient(

uri=self.uri,

token=self.token,

db_name=self.database,)

logger.debug(f"client inited: {self.client}")

self.get_or_create_collection()

logger.debug(f"collection inited: {self.collection_name}")

self.client.load_collection(collection_name=self.collection_name)

logger.debug(f"collection loaded: {self.collection_name}")

def get_or_create_collection(self):

logger.debug(f"connect to milvus server: {self.uri}")

res = self.client.has_collection(self.collection_name)

logger.debug(f"has collections res: {res}")

if res:

logger.info(f"Collection {self.collection_name} already exists")

else:

logger.info(

f"Collection {self.collection_name} not found, create it")

self.client.create_collection(

collection_name=self.collection_name,

schema=self.schema(),

num_shards=1,

consistency_level="Strong"

)

self.init_index()

def schema(self):

fields = [

pymilvus.FieldSchema(name="ids",

dtype=pymilvus.DataType.VARCHAR,

is_primary=True,

auto_id=False,

max_length=128,

),

pymilvus.FieldSchema(name="documents",

max_length=65530,

dtype=pymilvus.DataType.VARCHAR),

pymilvus.FieldSchema(name="embeddings",

dtype=pymilvus.DataType.FLOAT_VECTOR,

dim=self.dim),

pymilvus.FieldSchema(name="code",

max_length=256,

dtype=pymilvus.DataType.VARCHAR),

pymilvus.FieldSchema(name="name_en",

max_length=256,

dtype=pymilvus.DataType.VARCHAR),

pymilvus.FieldSchema(name="name_cn",

max_length=256,

dtype=pymilvus.DataType.VARCHAR),

]

logger.debug(f"fields:\n\t{fields}")

schema = pymilvus.CollectionSchema(

fields=fields,

description="ebom feature collection",

)

return schema

def init_index(self):

logger.debug("init index")

index_params = self.client.prepare_index_params()

index_params.add_index(

field_name="ids",

index_type="INVERTED"

)

index_params.add_index(

field_name="embeddings",

index_type="HNSW",

metric_type="COSINE",

params={

"M": 32,

"efConstruction": 256,

}

)

self.client.create_index(self.collection_name,

index_params=index_params)

logger.debug("finished index created")

def milvus_get_response2result(self, response):

logger.debug(f"milvus get res-->result:\n\t{response}\n")

res_data = {}

if isinstance(response, ExtraList) and response:

res_data = (list(response))[0]

logger.debug(f"milvus get res-->result ExtraList:\n\t{res_data}\n")

elif isinstance(response, dict):

logger.debug(f"milvus get res-->result dict:\n\t{res_data}\n")

res_data = response

else:

return {}

res_data['metadatas'] = {key: res_data[key] for key in [

"code",

"name_en",

"name_cn",

]

}

result = {key: res_data[key] for key in [

"ids",

"documents",

"metadatas"]

}

return result

def get(self, ids):

# for get data from ids

if isinstance(ids, list):

pass

else:

ids = [ids]

response = self.client.get(

collection_name=self.collection_name,

ids=ids,

output_fields=self.get_output_fields

)

logger.debug(f"\nmilvus get ids: {[ids]}\n")

logger.debug(f"milvus get response:\n\t{response}\n")

result = self.milvus_get_response2result(response=response)

logger.debug(f"milvus get result:\n\t{result}\n")

return result

def milvus_query_response2result(self, response):

result = {'ids': [], 'documents': [], 'metadatas': []}

if response:

for res in response:

result_of_res = self.milvus_get_response2result(response=res)

result['ids'].append(result_of_res['ids'])

result['documents'].append(result_of_res['documents'])

result['metadatas'].append(result_of_res['metadatas'])

return result

def query(self, feature_code):

# for get data from feature code filter

response = self.client.query(

collection_name=self.collection_name,

filter=f"code == '{feature_code}'",

output_fields=self.get_output_fields

)

logger.debug(f"milvus get query response:\n\t{response}\n")

result = self.milvus_query_response2result(response=response)

logger.debug(f"milvus get query result:\n\t{result}\n")

return result

def insert(self, ids, documents, embeddings, metadatas):

logger.debug(f"milvus insert data ids:\n\t{ids}\n")

# print(f"milvus insert data documents: {documents}")

# print(f"milvus insert data embeddings: {embeddings}")

# print(f"milvus insert data metadatas: {metadatas}")

try:

response = self.client.insert(

collection_name=self.collection_name,

data={

"ids": ids,

"documents": documents,

"embeddings": embeddings,

**metadatas[0] # "code", "name_en", "name_cn"

}

)

logger.debug(f"milvus insert response:\n\t{response}\n")

if response['insert_count'] == 1 and response['ids'][0] == ids:

return {'ids': ids,

'documents': documents,

"metadatas": metadatas[0]}

else:

logger.debug(f"milvus insert error: {response}")

return f"milvus insert error: {response}"

except Exception as e:

logger.debug(f"milvus insert error: {e}")

return f"milvus insert error: {e}"

def milvus_search_response2result(self, response):

result = {}

if (list(response))[0]:

res = (list(response))[0][0]

logger.debug(f"milvus search res-->result:\n\t{res}\n")

entity = res['entity']

entity['metadatas'] = {key: entity[key] for key in ["code",

"name_en",

"name_cn",

]

}

entity['ids'] = res['id']

entity['distances'] = 1 - res['distance']

result = {key: entity[key] for key in ["ids",

"distances",

"documents",

"metadatas"]

}

return result

else:

result = {

"ids": "",

"distances": "",

"documents": "",

"metadatas": ""

}

return result

def search(self, query_embeddings, n_results, where=""):

""" only for search 1 embeddings, and return 1 result """

logger.debug(f"start milvus search query_embeddings:"

f"{query_embeddings}"

f"{where}"

)

expr = ""

if where != "":

pairs = [f'{key} == "{value}"' for key, value in where.items()]

if not expr:

expr = " and ".join(pairs)

else:

expr = expr + " and " + " and ".join(pairs)

logger.debug(f"expr all: {expr}")

search_params = {

"metric_type": "COSINE",

"params": {"nprobe": 10, }

}

response = self.client.search(

collection_name=self.collection_name,

data=query_embeddings,

search_params=search_params,

limit=n_results,

filter=expr,

output_fields=self.search_output_fields,

)

logger.debug(f"start milvus search response: {response}")

result = self.milvus_search_response2result(response=response)

return result

def delete(self, ids):

response = self.client.delete(

collection_name=self.collection_name,

ids=ids

)

logger.debug(f"milvus delete response:\n\t{response}\n")

time.sleep(1)

if not self.get(ids):

return response

time.sleep(1)

return self.client.delete(

collection_name=self.collection_name,

ids=ids

)

class VectorDB(VectorClassBase):

def __init__(self):

try:

with open('config.yaml', 'r', encoding='utf8') as f:

config_data = yaml.safe_load(f)

self.config = config_data['vectorDB']

# Use a mapping instead of if-elif to map projects

# to their implementations

db_mapping = {

'milvus': Milvus

}

# Attempt to instantiate the correct database class

self.vectordb_class = db_mapping.get(self.config['project'])

if not self.vectordb_class:

raise ValueError(

f"not supported vector database: {self.config['project']}")

logger.info(

f"Initializing {self.config['project']} vector database")

# init vectordb client and collection name

self.vectordb = self.vectordb_class(config=self.config)

super().__init__(config=self.config)

self.collection_name = self.vectordb.collection_name

except FileNotFoundError as e:

logger.error("config.yaml file not found")

raise e # or some other appropriate handling

except yaml.YAMLError as e:

logger.error("Error in config.yaml YAML format")

raise e # or some other appropriate handling

except Exception as e:

logger.error("An error occurred during initialization")

raise e # or some other appropriate handling

def get(self, ids):

response = self.vectordb.get(ids=ids)

logger.debug(f"vectordb get response: {response}\n")

return response

def insert(self, ids, documents, embeddings, metadatas):

response = self.vectordb.insert(

ids=ids,

documents=documents,

embeddings=embeddings,

metadatas=metadatas,

)

logger.debug(f"vectordb insert response: {response}\n")

return response

def search(self, query_embeddings, n_results, where=""):

response = self.vectordb.search(query_embeddings=query_embeddings,

n_results=n_results, where=where)

logger.debug(f"vectordb search response: {response}\n")

return response

def delete(self, ids):

response = self.vectordb.delete(ids=ids)

logger.debug(f"vectordb delete response: {response}\n")

return response

def query(self, feature_code):

response = self.vectordb.query(code=feature_code)

logger.debug(f"vectordb query feature Code response: {response}\n")

return response