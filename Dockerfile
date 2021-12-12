FROM python:3.6
RUN pip3 install --upgrade pip setuptools
COPY requirements.txt / 
RUN pip3 install -r requirements.txt
COPY src/ /src