#!/bin/sh
set -e

# 初始化日志目录
mkdir -p /app/logs /app/data

if [ "$1" = "cron" ]; then
  echo "启动定时任务: 工作日 16:00(Asia/Shanghai) 运行每日交易"
  echo "0 16 * * 1-5 cd /app && /usr/local/bin/python run_daily.py --today >> /app/logs/daily.log 2>&1" | crontab -
  crond -f
else
  exec "$@"
fi
