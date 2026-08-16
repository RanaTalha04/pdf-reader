import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")


from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings

load_dotenv()

# Load the documents
loader = PyPDFLoader("data/Week 15.pdf")
document = loader.load()

# Text Splitter - Splitting into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(document)

# Vector Database
vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings())

# Retriever
retriever = vectorstore.as_retriever()

# Query
question = input("Ask your question: ")
retrieved_docs = retriever.invoke(question)

retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = f"Based on the following document, answer questions: {question}\n\n{retrieved_text}"
answer = model.invoke(prompt)

print(f"Answer: {answer.content}")