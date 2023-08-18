FROM python:3.10-slim-buster

# Install nginx and supervisord from Debian repo
RUN apt-get update && apt-get install -y nginx supervisor && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

ADD . /app 

WORKDIR /app

RUN pip install --no-cache-dir -r combined_requirements.txt
#RUN pip install --no-cache-dir -r mlruns/1/450ae60519ee43fda2402ae292be69d4/artifacts/model/requirements.txt

# Copy nginx configuration file
COPY nginx.conf /etc/nginx/sites-enabled/default

# Copy supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Remember to post port 80
EXPOSE 80

# Run supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]




