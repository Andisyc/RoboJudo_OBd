#!/bin/bash
set -e

# 获取执行脚本时传入的第一个参数。如果没有传入，则默认使用 "sync update"
COMMIT_MSG=${1:-"sync update"}
CURRENT_BRANCH=$(git branch --show-current)

if [ -z "$CURRENT_BRANCH" ]; then
  echo "❌ 当前不是一个命名分支，无法安全推送。"
  exit 1
fi

echo "🚀 开始同步代码到远程仓库..."
# 1. 暂存所有更改
echo "📦 正在执行: git add ."
git add .
# 2. 提交更改
echo "📝 正在执行: git commit -m \"$COMMIT_MSG\""
if git diff --cached --quiet; then
  echo "ℹ️ 没有需要提交的更改，跳过 commit。"
else
  git commit -m "$COMMIT_MSG"
fi
# 3. 推送到当前分支
echo "☁️ 正在执行: git push origin $CURRENT_BRANCH"
git push -u origin "$CURRENT_BRANCH"
