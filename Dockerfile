# pull python base image
FROM python:3.11

# specify working directory
WORKDIR /llm_app

# copy required files
ADD ./llm_app  /llm_app
ADD ./chroma_db /chroma_db
ADD ./data /data


# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt  --no-cache-dir

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "main.py"]