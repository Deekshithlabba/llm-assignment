from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import string
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import fitz
import csv
import mysql.connector

# Load environment variables from .env file (if any)
load_dotenv()

# Initialize MySQL connection
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Deekshith@1602",
    database="file_storage_db"
)
db_cursor = db_connection.cursor()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocess text function
def preprocess_text(text):
    sentences = sent_tokenize(text)
    translator = str.maketrans('', '', string.punctuation + string.digits)
    cleaned_sentences = [sentence.translate(translator).strip() for sentence in sentences]
    stop_words = set(stopwords.words("english"))
    filtered_sentences = [sentence for sentence in cleaned_sentences if sentence.lower() not in stop_words]
    return " ".join(filtered_sentences)

# Extract text from PDF function
def extract_text_from_pdf(file_contents):
    pdf_document = fitz.open(stream=io.BytesIO(file_contents))
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text

class Response(BaseModel):
    result: str | None

@app.post("/predict", response_model=Response)
async def predict(request: Request, question: str = Form(...), file: UploadFile = File(...)):
    # Check if the uploaded file is a TXT, PDF, DOCX, or CSV file
    if file.content_type not in ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/csv"]:
        return {"result": "Error: Please upload a TXT, PDF, DOCX, or CSV file"}

    # Read the file contents
    file_contents = await file.read()

    # Process the file
    result = None
    if file.content_type == "text/plain":
        text = file_contents.decode("utf-8")
        if question.lower() == "summarize the text":
            cleaned_text = preprocess_text(text)
            parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=3)
            result = " ".join(str(sentence) for sentence in summary)
    elif file.content_type == "application/pdf":
        text = extract_text_from_pdf(file_contents)
        if question.lower() == "summarize the text":
            cleaned_text = preprocess_text(text)
            parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=3)
            result = " ".join(str(sentence) for sentence in summary)
    elif file.content_type == "text/csv":
        file_contents_str = file_contents.decode("utf-8")
        csv_data = []
        reader = csv.reader(io.StringIO(file_contents_str))
        for row in reader:
            if row[0] == question:
                result = row[3]
                break

    # If result is still None, ask the generative AI chat bot
    if result is None:
        result = bot.get_response(question)

    return {"result": result}
