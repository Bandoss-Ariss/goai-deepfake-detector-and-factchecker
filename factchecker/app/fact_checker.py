from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from transformers import pipeline
import numpy as np
import os
from app.utils import *
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize embeddings and FAISS index
def initialize_index():
    #embeddings = OpenAIEmbeddings(api_key = os.getenv("OPENAI_API_KEY"))
    #dimension = embeddings.embed_query("test").shape[0]
    #index = faiss.IndexFlatL2(dimension)
    embeddings = []
    index = 0
    return embeddings, index

# Verify claim using generated headlines
def verify_claim_with_generated_headlines(user_claim, embeddings, index):
    headlines = generate_headlines_gpt(user_claim)
    articles = combine_articles_context(headlines)
    result = query_gpt_with_context(user_claim, articles)

    explanation = ""
    if result.lower() == 'true':
        explanation = "We fact-checked this info and can determine with high confidence it is TRUE."
    elif result.lower() == 'false':
        explanation = "We fact-checked this info and can determine with confidence it is FALSE."
    else:
        explanation = (
            "We were not able to get any evidence for this info being true. We advise you to take it with caution. "
            "We sent it to our manual fact-checkers and they will get back to you ASAP."
        )

    return {
        'status': result,
        'reason': result,
        'explanation': explanation,
        'articles': [
            {'title': article.get('title', 'No title available'), 'url': article.get('link', 'No URL available')}
            for article in articles
        ]
    }