FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# timezone data + cron (for daily scheduled trading)
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata cron \
    && rm -rf /var/lib/apt/lists/*

# install dependencies first (leverage docker layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy project source and configs
COPY . .

# entrypoint for cron scheduling
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# default: start daily cron; manual: docker run ... python run_daily.py --today
CMD ["sh", "/usr/local/bin/docker-entrypoint.sh", "cron"]
