FROM python:3.7.6-buster

WORKDIR /app
ADD . /app

RUN apt-get update && apt-get install -y \
	build-essential \
	curl \
	apt-utils \
	tesseract-ocr \
	tesseract-ocr-deu \
	libsm6 \
	libxext6 \
	libxrender-dev


RUN pip install -r requirements.txt
RUN pip install gunicorn


CMD ["gunicorn", "-b", "0.0.0.0:8000", "server:app"]
