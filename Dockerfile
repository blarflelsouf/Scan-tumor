FROM python:3.10.6-buster

# COPY -> WHAT folder/ file <space> HOW to name it in the image
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt requirements.txt
COPY api /api
COPY interface /interface
COPY ml_logic /ml_logic
COPY utils.py /utils.py

RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn api.fastapi:app --host 0.0.0.0 --port $PORT

# $DEL_END
