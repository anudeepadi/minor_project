from transformers import pipeline
import logging
import time

logger = logging.getLogger(__name__)

class Summarizer:

    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def get_summary(self, text):
        # GET TIME TAKEN TO SUMMARIZE
        start = time.time()
        sum_text = self.summarizer(text, max_length=100, min_length=30, do_sample=False)
        end = time.time()
        logger.info(f"Time taken to summarize: {end-start}")
        dic = {
            "Original Text": text,
            "Summary Text": sum_text[0]['summary_text'],
            "Length Before Summariztion": len(text),
            "Length After Summarization": len(sum_text[0]['summary_text']),
            "Percentage Reduction": 100 - round((len(sum_text[0]['summary_text'])/len(text))*100),
            "Time Taken": end-start
        }
        return dic
    
    def get_all(self):
        text = self.text
        summary = self.summary[0]['summary_text']
        per = 100 - round((len(summary)/len(text))*100)
        return text, summary, per
