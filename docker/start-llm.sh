#!/bin/bash
set -e

echo "🤖 Starting Phi-3 LLM Service..."

# Ollamaサーバーをバックグラウンドで起動
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Ollamaが起動するまで待機
echo "Waiting for Ollama to start..."
sleep 15

# Ollamaサーバーが正常に起動したか確認
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama server is ready"
        break
    fi
    echo "Waiting for Ollama... ($i/30)"
    sleep 2
done

# Phi-3 miniモデルをダウンロード（存在しない場合のみ）
echo "Checking for Phi-3 mini model..."
if ! ollama list | grep -q "phi3:mini"; then
    echo "Downloading Phi-3 mini model... (this may take several minutes)"
    ollama pull phi3:mini
else
    echo "Phi-3 mini model already exists"
fi

# FastAPIアプリケーションを起動
echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 8002