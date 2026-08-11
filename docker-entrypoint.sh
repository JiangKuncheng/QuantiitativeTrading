#!/bin/sh
set -e

# 初始化日志目录
mkdir -p /app/logs /app/data

if [ "$1" = "cron" ]; then
  echo "启动定时任务(Asia/Shanghai): 9:20 读计划, 9:25 取开盘价(失败每10分钟重试, 最多5次), 16:00 结算(行情未发布则每15分钟重试, 最多20次)"
  echo "20 9 * * 1-5 cd /app && /usr/local/bin/python run_daily.py --execute >> /app/logs/execute.log 2>&1
0 16 * * 1-5 cd /app && /usr/local/bin/python run_daily.py --settle >> /app/logs/daily.log 2>&1" | crontab -
  # Debian 的 cron 守护进程是 cron, CentOS/Arch 是 crond
  if command -v crond >/dev/null 2>&1; then
    exec crond -f
  else
    exec cron -f
  fi
else
  exec "$@"
fi
