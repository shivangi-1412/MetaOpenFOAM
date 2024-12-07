import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


import openai
import re
import config_path

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content={self.page_content[:30]}..., metadata={self.metadata})"

database_tutorials_summary_path = f'{config_path.Database_PATH}/openfoam_tutorials_summary.txt'
# loader = TextLoader(database_tutorials_summary_path)
# pages = loader.load()

# pattern = re.compile(r"###case begin:(.*?)case end.###", re.DOTALL)
# matches = pattern.findall(pages[0].page_content)
# pages = [Document(page_content=match.strip(),metadata={'source': database_tutorials_summary_path}) for match in matches]

persist_directory = f'{config_path.Database_PATH}/openfoam_tutorials_summary'

embed_dir = f'{config_path.Database_PATH}/openfoam_tutorials_summary/'
print(os.path.isfile(f'{embed_dir}/index.faiss'))

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
#batch_size = config_path.batchsize


if os.path.exists(persist_directory):
    vectordb = FAISS.load_local(
        embed_dir, embeddings, allow_dangerous_deserialization=True
    )
    
# for i in range(0, len(pages), batch_size):
#     print("i:",i)
#     if(i+batch_size<=len(pages)-1):
#         batch = pages[i:i + batch_size]
#     elif(i<=len(pages)-2):
#         batch = pages[i:]

#     try:
#         if(i==0):
#             qdrant = Qdrant.from_documents(
#                 batch,
#                 embeddings,
#                 # location=":memory:",
#                 path="persist_directory",
#                 collection_name="all_run1",
#             )
#         else:
#             qdrant = Qdrant.add_documents(
#                 batch,
#                 embeddings,
#                 # location=":memory:",
#                 path="persist_directory",
#                 collection_name="all_run1",
#             )
#     except Exception as e:  
#         print(f"Error processing batch {i // batch_size + 1}: {e}")
#         break

# if vectordb is not None:
#     vectordb.save_local(persist_directory)
# else:
#     print("vectordb was not initialized due to a previous error. Skipping save.")


