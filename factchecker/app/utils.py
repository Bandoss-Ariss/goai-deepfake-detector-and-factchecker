import pytesseract
from openai import OpenAI
from PIL import Image
from io import BytesIO
import requests
import easyocr
from transformers import pipeline
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import faiss
import os
from dotenv import load_dotenv
load_dotenv()
# Summarization model
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# Load dataset and initialize FAISS index
def create_embeddings(): 
    dataset = load_dataset("roupenminassian/twitter-misinformation", split="train")
    texts = dataset["text"]
    labels = dataset["label"]
    embeddings = OpenAIEmbeddings()
    claim_embeddings = np.array([embeddings.embed_query(text) for text in texts]).astype("float32")
    dimension = claim_embeddings.shape[1]
    cpu_index = faiss.IndexFlatL2(dimension)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(claim_embeddings)



# Extract text from image or image URL
def extract_text_from_image(image=None, image_url=None):
    try:
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'], gpu=False)  # Set `gpu=True` if you have GPU support

        # Load image from URL or file
        if image_url:
            response = requests.get(image_url, timeout=20)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        elif image:
            image = Image.open(image)
        else:
            return "No image provided."

        # Convert the image to a format EasyOCR can process
        image = image.convert("RGB")
        text = reader.readtext(np.array(image), detail=0)  # Extract text (detail=0 returns plain text)
        return " ".join(text)
    except Exception as e:
        return f"Error: {e}"

def process_twilio_image(media_url, account_sid, auth_token):
    """
    Downloads an image from Twilio media URL and extracts text using EasyOCR.
    
    Parameters:
        media_url (str): The Twilio media URL (MediaUrl0).
        account_sid (str): Twilio Account SID for authentication.
        auth_token (str): Twilio Auth Token for authentication.
    
    Returns:
        str: Extracted text from the image.
    """
    try:
        # Step 1: Download the image using Twilio media URL
        response = requests.get(media_url, auth=(account_sid, auth_token), timeout=20)
        if response.status_code != 200:
            return f"Failed to download media. Status code: {response.status_code}, Error: {response.text}"

        # Step 2: Open the image
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Step 3: Use EasyOCR to extract text
        reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR reader
        image_np = np.array(image)  # Convert PIL image to NumPy array
        text = reader.readtext(image_np, detail=0)  # Extract text (detail=0 returns plain text)
        
        # Step 4: Return extracted text
        return " ".join(text)
    
    except Exception as e:
        return f"Error processing media: {e}"

# Functions
def extract_claim(paragraph):
    sentences = sent_tokenize(paragraph)
    summarized_text = summarizer(paragraph, max_length=50, min_length=10, do_sample=False)
    summary = summarized_text[0]["summary_text"]

    best_sentence = None
    highest_similarity = 0
    for sentence in sentences:
        similarity_score = len(set(summary.split()) & set(sentence.split())) / len(set(summary.split()))
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            best_sentence = sentence

    refined_claim = refine_claim_with_gpt(paragraph)
    return refined_claim.strip() if refined_claim else "No claim found."

def refine_claim_with_gpt(claim):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    prompt = f"Refine the following paragraph into a newspaper Headline. Be short.Paragrah: {claim}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def similarity_search(query, top_k=5):
    query_embedding = np.array(embeddings.embed_query(query)).astype("float32").reshape(1, -1)
    distances, indices = gpu_index.search(query_embedding, top_k)
    return [(texts[i], labels[i], distances[0][j]) for j, i in enumerate(indices[0])]

def generate_headlines_gpt(user_claim):
    prompt = (
        f"Generate 5 possible headlines for the following claim: \"{user_claim}\"."
    )
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return [line.strip("- ").strip() for line in response.choices[0].message.content.strip().split("\n") if line.strip()]

def search_google_news(query):
    api_key = os.getenv("GOOGLE_NEWS_API_KEY")
    endpoint = "https://google.serper.dev/search"
    data = {"q": query}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("organic", [])
    else:
        return []

def combine_articles_context(headlines):
    all_articles = []
    for headline in headlines:
        articles = search_google_news(headline)
        all_articles.extend(articles[:2])
    return all_articles

def query_gpt_with_context(user_claim, articles):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    articles_text = "\n\n".join(
        f"Title: {article.get('title', 'No title available')}" for article in articles
    )
    
    input_prompt = (
        f"The following claim needs fact-checking: \"{user_claim}\"\n\n"
        f"THESE ARE HEADLINES AND CONTENT FROM TRUSTED SOURCES:\n\n{articles_text}\n\n"
        f"Based on the provided articles and your general knowledge, reply in ONE word only: True, False, or Unverifiable. "
        f"BE CAUTIOUS WHEN IT IS ABOUT ATTACKS OR MURDERS"
        f"Do not provide any additional information or commentary. Ensure your response considers both the articles and your internal knowledge."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_prompt}]
    )
    return response.choices[0].message.content.strip()