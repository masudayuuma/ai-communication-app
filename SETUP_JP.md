# 🎌 AI英会話アプリ - 日本語セットアップガイド

## 📝 概要
このアプリは、AI（人工知能）と英語で会話練習ができるデスクトップアプリケーションです。
- あなたが英語で話すと、AIが音声で返答します
- リアルタイムで音声認識・音声合成を行います
- Llama-3とSeamlessM4Tという最新技術を使用

## 🚀 初心者向けセットアップ手順

### ステップ1: Homebrewのインストール
Homebrewは、Macでソフトウェアを簡単にインストールするためのツールです。

```bash
# ターミナルで以下のコマンドを実行
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### ステップ2: Pythonとその他の必要ソフトのインストール

```bash
# Python 3.11とPortAudio（音声処理用）をインストール
brew install python@3.11 portaudio

# Poetryをインストール（Pythonのパッケージ管理ツール）
curl -sSL https://install.python-poetry.org | python3 -

# Poetryのパスを通す
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### ステップ3: Ollamaのインストール
OllamaはAI言語モデルをローカルで動かすためのツールです。

```bash
# Ollamaをインストール
curl -fsSL https://ollama.ai/install.sh | sh

# Ollamaを起動（バックグラウンドで動作）
ollama serve &

# 数秒待ってから、AI言語モデルをダウンロード（約5GB、10-15分程度）
sleep 10
ollama pull llama3:8b-instruct
```

### ステップ4: アプリケーションのセットアップ

```bash
# 既にクローン済みのディレクトリに移動
cd /Users/masudayuuma/Development/ai-communication-app

# 依存関係をインストール
poetry install

# SeamlessM4Tをインストール（音声処理用、約3GB）
poetry run pip install git+https://github.com/facebookresearch/seamless_communication.git
```

## 🎯 動作確認

### テスト1: 依存関係の確認

```bash
# 依存関係をチェック
poetry run python main.py --check-deps
```

**期待される出力:**
```
INFO - PyTorch version: 2.x.x
INFO - CUDA available: True/False
INFO - Streamlit version: 1.35.x
INFO - Sounddevice available
INFO - Dependencies check completed successfully
```

### テスト2: オーディオデバイスの確認

```bash
# オーディオデバイスをテスト
poetry run python main.py --cli
```

**期待される出力:**
```
INFO - Available audio devices: X input, Y output
INFO - LLM manager initialized successfully
INFO - S2S manager initialized successfully
INFO - CLI mode testing completed
```

### テスト3: アプリケーションの起動

```bash
# アプリケーションを起動
poetry run python main.py
```

**期待される出力:**
```
INFO - Starting AI Communication App
INFO - Starting Streamlit app on http://localhost:8501
```

ブラウザが自動で開き、`http://localhost:8501` でアプリにアクセスできます。

## 🎤 使い方

### 基本的な使用手順

1. **ブラウザでアプリを開く**: http://localhost:8501
2. **システム初期化**: 「🚀 Initialize System」ボタンをクリック
3. **音声認識開始**: 「🎤 Start」ボタンをクリック
4. **英語で話す**: マイクに向かって英語で話しかける
5. **AIの応答**: AIが英語で音声返答します
6. **停止**: 「⏹️ Stop」ボタンで一時停止

### 会話例

**あなた**: "Hello, how are you today?"
**AI**: "Hello! I'm doing well, thank you for asking. How are you doing? What would you like to talk about today?"

**あなた**: "I want to practice English conversation."
**AI**: "That's great! I'm here to help you practice. What topics interest you? We could talk about hobbies, travel, food, or anything else you'd like."

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. 「マイクが見つかりません」エラー

```bash
# マイクの設定を確認
poetry run python -c "import sounddevice; print(sounddevice.query_devices())"

# システム環境設定 > セキュリティとプライバシー > マイク で、ターミナルやPythonにマイクアクセスを許可
```

#### 2. 「Ollama接続エラー」

```bash
# Ollamaの状態確認
ollama list

# Ollamaが起動していない場合
ollama serve

# 別のターミナルで確認
curl http://localhost:11434/api/tags
```

#### 3. 「メモリ不足」エラー

```bash
# 軽量モデルに変更
ollama pull llama3:8b-instruct-q4_0

# アプリでモデルを変更: サイドバーのModel Settingsで「llama3:8b-instruct-q4_0」を選択
```

#### 4. 音声が聞こえない

1. **音量確認**: スピーカー音量をチェック
2. **デバイス確認**: システム環境設定 > サウンド で出力デバイスを確認
3. **アプリ再起動**: ブラウザでページを更新

## 📱 操作方法詳細

### メイン画面の説明

- **🟢 Status Ready**: システムが正常に動作中
- **🔴 Status Error**: エラーが発生、再初期化が必要
- **🎤 Start/Stop**: 音声認識の開始/停止
- **🗑️ Clear**: 会話履歴をクリア

### サイドバーの機能

- **🤖 Model Settings**: AI言語モデルの変更
- **🌍 Language Settings**: 音声認識の言語設定（現在は英語のみ推奨）
- **📱 Audio Devices**: 利用可能なオーディオデバイス表示

### デバッグ情報

画面下部の「🔧 Debug Information」で以下を確認可能:
- システム状態
- 処理時間
- 会話履歴数

## 🎯 動作確認チェックリスト

完全な動作確認のために、以下をすべて実行してください:

### ✅ 基本機能テスト

1. **アプリ起動**: `poetry run python main.py` でエラーなく起動
2. **ブラウザアクセス**: http://localhost:8501 でUIが表示
3. **システム初期化**: 「Initialize System」でStatus: Ready
4. **マイクテスト**: 「Start」で"🎤 Listening..."表示
5. **音声認識**: 英語で話すと会話ログに表示
6. **音声出力**: AIの返答が音声で再生

### ✅ 会話テスト

簡単な会話例:

```
あなた: "Hello"
期待される応答: AIが挨拶を返す

あなた: "What's your name?"
期待される応答: AIが自己紹介

あなた: "Tell me about Japan"
期待される応答: 日本について説明
```

## 🚨 重要な注意事項

1. **初回起動**: モデルダウンロードで時間がかかる（15-25分）
2. **メモリ使用量**: 8GB以上のRAMを推奨
3. **ネットワーク**: 初回のみインターネット接続必要
4. **音声品質**: クリアな発音で、背景ノイズを避ける
5. **プライバシー**: すべての処理はローカルで実行（外部送信なし）

## 📞 サポート

問題が解決しない場合:

1. **ログ確認**: ターミナルのエラーメッセージをコピー
2. **GitHub Issues**: https://github.com/masudayuuma/ai-communication-app/issues
3. **詳細ログ**: `poetry run python main.py --debug --log-level DEBUG` で詳細情報取得

---

**🎉 セットアップ完了！英語会話練習を楽しんでください！**