#!/usr/bin/env bash
# 监控 POET 降 lr 重跑（24 个 run）。每条 stdout 行 = 一条通知。
# 检测三件事：
#   1) 进程数 < 24 → 有 run 退出/挂了
#   2) 任一 log 出现 Traceback → 报错
#   3) 任一 run 最近 30 个 eval 全 0 且历史曾 >0.4 → 疑似 Neumann 爆炸崩了（每文件只报一次）
#      （MetaWorld 任务有 ~10-20 eval 的"从 0 爬起"暖机期，30 才越过暖机期、只报真崩）
# 全部进程结束（进程数=0）时退出。
L="C:/Users/杨斯杰/Desktop/第四周/parseval_reg/parseval_reg/logs/mw_poet_neumann_lowlr"

count_py() {
  powershell.exe -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count" 2>/dev/null | tr -d '[:space:]'
}

seen=""

while true; do
  n=$(count_py)
  if [ "$n" = "0" ]; then
    echo "全部进程结束，监控退出"
    break
  fi
  if [ -n "$n" ] && [ "$n" -lt 24 ] 2>/dev/null; then
    echo "进程数=$n（<24），有 run 退出，需排查"
  fi

  tb=$(grep -l "Traceback" "$L"/*.log 2>/dev/null)
  if [ -n "$tb" ]; then
    echo "发现 Traceback：$(echo "$tb" | tr '\n' ' ')"
  fi

  for f in "$L"/*.log; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    case " $seen " in *" $b "*) continue ;; esac   # 已报过的不再报
    vals=$(grep -oE "success [0-9.]+" "$f" 2>/dev/null | grep -oE "[0-9.]+$" | tail -30)
    nv=$(echo "$vals" | grep -c .)
    zeros=$(echo "$vals" | awk '$1==0{c++} END{print c+0}')
    mx=$(grep -oE "success [0-9.]+" "$f" 2>/dev/null | grep -oE "[0-9.]+$" | awk 'BEGIN{m=0}{if($1>m)m=$1}END{print m}')
    if [ "$nv" -ge 30 ] && [ "$zeros" -ge 30 ] && awk "BEGIN{exit !($mx > 0.4)}"; then
      echo "疑似崩（最近 30 eval 全 0，历史最高 $mx）：$b"
      seen=" $seen $b "
    fi
  done

  sleep 180
done
