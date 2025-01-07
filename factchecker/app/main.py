from fastapi import FastAPI, UploadFile, Form, File, Request, status
from fastapi.responses import JSONResponse
import requests
import os
from typing import Union
from app.utils import extract_text_from_image, process_twilio_image
from app.fact_checker import verify_claim_with_generated_headlines, initialize_index
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
app = FastAPI()

# Allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings, index = initialize_index()

@app.post("/verify-claim/")
async def verify_claim(
    text: str = Form(None),
    image: Union[UploadFile, str, None] = File(None, media_type="image/*"),
    image_url: str = Form(None)
):
    if image_url:
        extracted_text = extract_text_from_image(image_url=image_url)
    elif image:
        extracted_text = extract_text_from_image(image=image.file)
    elif text:
        extracted_text = text
    else:
        return {"error": "Provide text, an image, or an image URL."}

    result = verify_claim_with_generated_headlines(extracted_text, embeddings, index)
    return result

twilio_ssid = os.getenv("TWILIO_SSID")
TWILIO_WHATSAPP_URL = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_ssid}/Messages.json"
TWILIO_AUTH = (os.getenv("TWILIO_SSID"), os.getenv("TWILIO_SECRET_KEY"))

@app.post("/whatsapp-webhook/")
async def whatsapp_webhook(request: Request):
    data = await request.form()
    incoming_message = data.get("Body", "").strip()
    media_url = data.get("MediaUrl0", None)
    from_number = data.get("From", "")

    if media_url:
        extracted_text = process_twilio_image(media_url, twilio_ssid, os.getenv("TWILIO_SECRET_KEY"))
    elif incoming_message:
        extracted_text = incoming_message
    else:
        return JSONResponse({"error": "No valid input received"}, status_code=400)
    print(extracted_text)
    result = verify_claim_with_generated_headlines(extracted_text, embeddings, index)
    explanation = result["explanation"]
    articles = result['articles']
    articles = [
            {'title': article.get('title', 'No title available'), 'url': article['url']}
            for article in articles
        ]
    articles = articles[:3]

    # Reply to the user
    response_message = explanation + "\n"
    if articles:
        response_message += '*Few relevant sources*: \n'
    for article in articles:
        title = article.get('title', 'No title availaible')
        url = article.get('url', 'No URL available')
        #response_message += title + ":"
        response_message += url + "\n"
    response_data = {
        "From": "whatsapp:+14155238886",
        "To": from_number,
        "Body": response_message,
    }
    requests.post(TWILIO_WHATSAPP_URL, data=response_data, auth=TWILIO_AUTH)

    return JSONResponse({"status": "success"})

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return health_check_handler()

@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def health_check_handler():
    return "200 OK"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Use PORT environment variable or default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)