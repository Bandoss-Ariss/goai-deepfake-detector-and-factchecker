import requests
import os
from dotenv import load_dotenv

import io
load_dotenv()
twilio_ssid = os.getenv("TWILIO_SSID")
url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_ssid}/Messages.json"
auth = (os.getenv("TWILIO_SSID"), os.getenv("TWILIO_SECRET_KEY"))
data = {
    'From': 'whatsapp:+14155238886',  # Twilio's WhatsApp sandbox number
    'To': 'whatsapp:+22666628303',
    'Body': 'Hello from Twilio!'
}

response = requests.post(url, data=data, auth=auth)

print(response.status_code)
print(response.text)
