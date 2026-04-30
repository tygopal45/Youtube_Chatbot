from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Step 1(a) : Indexing(Document Ingestion)

video_id = "Gfr50f6ZBvo"

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US"])

    transcript = " ".join(chunk["text"] for chunk in transcript_list)

except TranscriptsDisabled:
    print("Transcripts are disabled")



# Step 1(b) : Indexing(Text Chunking)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Step 1(c) & 1(d): Indexing(Embedding Creation and Storing in Vector Database)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_store = FAISS.from_documents(chunks, embeddings)


# Step 2 : Retrieval

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever.invoke("What is the video about?")


# Step 3 : Augmentation

llm = GoogleGenerativeAI(model="models/gemini-2.0-pro")

prompt_template = PromptTemplate.from_template(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient to answer the question, say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

question = "If the topic of aliens discussed in the video? If yes, what are the key points mentioned about aliens?"
retriever_docs = retriever.invoke(question)


context_text = "\n\n".join(doc.page_content for doc in retriever_docs)

final_prompt = prompt_template.format(context=context_text, question=question)


# Step 4 : Generation

response = llm.invoke(final_prompt)
print(response)



# Chaining the above steps together

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    context_text = "\n\n".join(doc.page_content for doc in docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke("Who is Demis?")

parser = StrOutputParser()

main_chain = parallel_chain | prompt_template | llm | parser

result = main_chain.invoke("Summarize the video please.")

print(result)

