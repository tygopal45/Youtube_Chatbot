from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

