FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# timezone data + cron (for daily scheduled trading)
# 国内服务器使用阿里云 apt 镜像源, 避免 Ubuntu 官方源超时
RUN sed -i 's@//archive.ubuntu.com@//mirrors.aliyun.com@g; s@//security.ubuntu.com@//mirrors.aliyun.com@g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata cron fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# install dependencies first (leverage docker layer cache)
# pip 镜像源可配置: 国内服务器构建时用 --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=$PIP_INDEX_URL
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy project source and configs
COPY . .

# entrypoint for cron scheduling
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# default: start daily cron; manual: docker run ... python run_daily.py --today
CMD ["sh", "/usr/local/bin/docker-entrypoint.sh", "cron"]
