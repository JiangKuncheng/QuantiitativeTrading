#!/bin/sh
set -e

# 初始化日志目录
mkdir -p /app/logs /app/data

if [ "$1" = "cron" ]; then
  echo "启动定时任务(Asia/Shanghai): 9:20 读计划并等待开盘价成交(最长15分钟), 16:00 结算+日报+生成明日计划"
  echo "20 9 * * 1-5 cd /app && /usr/local/bin/python run_daily.py --execute >> /app/logs/execute.log 2>&1
0 16 * * 1-5 cd /app && /usr/local/bin/python run_daily.py --settle >> /app/logs/daily.log 2>&1" | crontab -
  crond -f
else
  exec "$@"
fi
