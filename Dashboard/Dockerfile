FROM python:3.10-slim-buster

RUN apt-get update && rm -rf /var/lib/apt/lists/*

ADD . /Dasboard

WORKDIR /Dasboard

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5002

# Run gunnicord for the dashboard
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "dashboard:app"]




