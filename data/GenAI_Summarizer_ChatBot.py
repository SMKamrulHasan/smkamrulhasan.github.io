import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import WikipediaLoader

load_dotenv()
# STEP 1: Load Data into Langchain Documents
# pip install -q pypdf
# pip install -q docx2txt
# pip install -q wikipedia

def load_wiki(query, lang='en', load_max_docs=2):
  loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
  data = loader.load()
  return data

def load_documents(file):
  name, ext = os.path.splitext(file)
  print(name)
  if ext == '.pdf':
    loader = PyPDFLoader(file)
  elif ext == '.docx':
    loader = Docx2txtLoader(file)
  else:
    print('document format is not supported')
    return None
  data = loader.load()
  return data

data = load_documents('2022_SPIE.pdf')
#data = load_documents('the_great_gatsby.docx')
#print(data[0].page_content)
#print(f'you have {len(data)} page in your data')
#print(f' you have {len(data[0].page_content)} character in the page')

 #STEP 2: Chunks data into smaller segments
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_data(data):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=256,
      chunk_overlap=0,
      length_function=len

  )
  chunks = text_splitter.split_documents(data)
  return chunks

chunks = chunk_data(data)
print(len(chunks))

# STEP 3: Embed the chunks into numerical vectors
import pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone
pc = pinecone.Pinecone(api_key='')
pc.list_indexes()


def insert_or_fetch_embeddings(index_name, chunks):
  embeddings = OpenAIEmbeddings( api_key='')
  if index_name in pc.list_indexes().names():
    print('already exist')
    vector_store = Pinecone.from_existing_index(embeddings, index_name)

  else:
    print('creating index..')
    pc.create_index(
        name = index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
  )
    vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    return vector_store

index_name = 'ragbuild'
vector_store = insert_or_fetch_embeddings(index_name, chunks)

# STEP 4: ASK and Get Answer
# 2. Using the question embedding and chunk embeddings,
#rank the vector by similarity search.
# The nearest vector represent the chunk nearest to the question
def ask_and_get_answer(vector_store, ques):
  from langchain_openai import ChatOpenAI
  from langchain.chains import RetrievalQA, ConversationalRetrievalChain
  from langchain.memory import ConversationBufferMemory

  llm = ChatOpenAI(api_key=')
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
  #chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever= retriever)

  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                chain_type='stuff',
                                                retriever= retriever,
                                                memory=memory)

  answer = chain.invoke(ques)
  return answer

# ques = 'what is this whole document about?'
# answer = ask_and_get_answer(vector_store, ques)
# print(answer)

i = 1
while True:
  question = input(f'Question #{i}: ')
  i = i + 1
  if question in ['quit', 'exit']:
    print('Exiting ... Goodbye')
    break
  answer = ask_and_get_answer(vector_store, question)
  print(f'\nAnswer: {answer}')
  print(f'\n {"_" * 50} \n')
