from summarize import Summarizer
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse


app = FastAPI(
    title="Minor Project",
    description="A simple API to summarize text",
    version="0.1.0",  
)

app.get("/", response_class=PlainTextResponse, tags=["Home"])
async def home():
    return "Welcome to the home page"

@app.post("/summarize", tags=["Summarize"])
async def summarize(text: str):
    summarizer = Summarizer()
    summary = summarizer.get_summary(text)
    return summary

@app.post("/summarize_all", tags=["Summarize"])
async def summarize_all(text: str):
    summarizer = Summarizer(text)
    text, summary, per = summarizer.get_all()
    return text, summary, per



