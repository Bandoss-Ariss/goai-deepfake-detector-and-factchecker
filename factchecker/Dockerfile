FROM python:3.9


WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

ENV OPENAI_API_KEY=
ENV GOOGLE_NEWS_API_KEY=
ENV TWILIO_SSID=
ENV TWILIO_SECRET_KEY=

ENV PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# README.md
# FactChecker API