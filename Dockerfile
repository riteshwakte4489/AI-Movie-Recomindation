# base image 
FROM python:3.11-slim

#workdir
WORKDIR /app

#copy


#run
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


#port
EXPOSE 8501

#command
CMD ["streamlit","run","app.py","--server.address=0.0.0.0"]