FROM python:3.7
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /workdir
