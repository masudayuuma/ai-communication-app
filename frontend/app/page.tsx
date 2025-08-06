'use client';

import { useState, useRef, useEffect } from 'react';

// 会話履歴型定義
interface ChatMessage {
  user: string;
  ai: string;
  timestamp: string;
}

// システム状態型定義
interface SystemStatus {
  ollama_connected: boolean;
  microphone_available: boolean;
  speaker_available: boolean;
  available_voices: number;
  available_microphones: number;
}

export default function VoiceConversationApp() {
  // 状態管理
  const [isRecording, setIsRecording] = useState(false);
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);

  // 音声処理用 Ref
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // APIベースURL
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // マイクアクセス許可
  const requestMicrophoneAccess = async (): Promise<MediaStream | null> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      return stream;
    } catch (error) {
      console.error('マイクアクセスエラー:', error);
      alert('マイクへのアクセスが必要です');
      return null;
    }
  };

  // 音声会話処理
  const processVoiceConversation = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      // FormDataで音声ファイルを送信（拡張子をMIMEタイプに合わせる）
      const formData = new FormData();
      const extension = audioBlob.type.split('/')[1] || 'webm'; // webm, ogg, wav
      formData.append('audio_file', audioBlob, `recording.${extension}`);
      
      const response = await fetch(`${API_BASE}/api/speech/chat`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      // 応答構造に応じて会話履歴を更新
      if (result.conversation_history) {
        setConversation(result.conversation_history);
      } else if (result.transcription && result.response) {
        // 単一応答の場合、会話履歴に追加
        const newMessage: ChatMessage = {
          user: result.transcription,
          ai: result.response,
          timestamp: new Date().toISOString()
        };
        setConversation(prev => [...prev, newMessage]);
      }
      
      // AI応答を音声で再生
      const aiResponse = result.response || result.ai_response;
      if (aiResponse) {
        await playTextAsSpeech(aiResponse);
      }
      
    } catch (error) {
      console.error('音声会話エラー:', error);
      
      // より詳細なエラーメッセージ
      let errorMessage = '音声処理中にエラーが発生しました';
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        errorMessage = 'バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。';
      } else if (error instanceof Error) {
        if (error.message.includes('404')) {
          errorMessage = 'APIエンドポイントが見つかりません。';
        } else if (error.message.includes('500')) {
          errorMessage = 'サーバー内部エラーが発生しました。';
        }
      }
      
      alert(errorMessage);
      
      // システム状態を再取得
      fetchSystemStatus();
    } finally {
      setIsProcessing(false);
    }
  };

  // テキストを音声で再生
  const playTextAsSpeech = async (text: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/speech/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error('音声合成に失敗しました');
      }
      
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      await audio.play();
      
      // メモリリークを防ぐためにURLを解放
      audio.addEventListener('ended', () => {
        URL.revokeObjectURL(audioUrl);
      });
      
    } catch (error) {
      console.error('音声再生エラー:', error);
      // フォールバック: ブラウザのTTSを使用
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        window.speechSynthesis.speak(utterance);
      }
    }
  };

  // 録音開始
  const startRecording = async () => {
    try {
      const stream = audioStream || await requestMicrophoneAccess();
      if (!stream) {
        alert('マイクロホンアクセスが必要です');
        return;
      }
      
      audioChunksRef.current = [];
      
      // より多くのブラウザで対応する音声形式を選択（WebM優先）
      let mimeType = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/webm')) {
        mimeType = 'audio/webm';
      } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
        mimeType = 'audio/ogg';
      } else if (MediaRecorder.isTypeSupported('audio/wav')) {
        mimeType = 'audio/wav';
      }
      
      console.log('🎵 使用する音声形式:', mimeType);
      
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current.onstop = async () => {
        try {
          const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
          console.log('🎤 録音完了:', audioBlob.size, 'bytes,', mimeType);
          await processVoiceConversation(audioBlob);
        } catch (error) {
          console.error('音声処理エラー:', error);
          alert('音声処理中にエラーが発生しました: ' + (error instanceof Error ? error.message : '不明なエラー'));
        }
      };
      
      mediaRecorderRef.current.onerror = (event) => {
        console.error('MediaRecorder エラー:', event);
        alert('録音中にエラーが発生しました');
        setIsRecording(false);
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      console.log('🎤 録音開始');
      
    } catch (error) {
      console.error('録音開始エラー:', error);
      alert('録音を開始できません: ' + (error instanceof Error ? error.message : '不明なエラー'));
    }
  };

  // 録音停止
  const stopRecording = () => {
    try {
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
        setIsProcessing(true); // 処理中状態に
        console.log('⏹️ 録音停止');
      }
    } catch (error) {
      console.error('録音停止エラー:', error);
      setIsRecording(false);
      setIsProcessing(false);
    }
  };

  // 会話履歴リセット
  const resetConversation = () => {
    setConversation([]);
  };

  // システム状態取得
  const fetchSystemStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      const status = await response.json();
      
      // バックエンドヘルスチェック応答形式に合わせて変換
      setSystemStatus({
        ollama_connected: status.services?.llm || false,
        microphone_available: true, // ブラウザで判定
        speaker_available: status.services?.tts || false,
        available_voices: 1,
        available_microphones: 1
      });
    } catch (error) {
      console.error('システム状態取得エラー:', error);
      // エラー時のフォールバック状態
      setSystemStatus({
        ollama_connected: false,
        microphone_available: true,
        speaker_available: false,
        available_voices: 0,
        available_microphones: 1
      });
    }
  };

  // 初期化
  useEffect(() => {
    fetchSystemStatus();
    
    return () => {
      if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // ステータス色分け
  const getStatusColor = (status: boolean) => {
    return status ? 'text-green-600' : 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* ヘッダー */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-slate-800 mb-2">
            🎤 AI English Conversation
          </h1>
          <p className="text-xl text-slate-600">
            Speech Recognition + Llama3 + Text-to-Speech
          </p>
        </div>

        {/* ステータスバー */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className={`p-4 rounded-lg shadow-md ${systemStatus?.ollama_connected ? 'bg-green-100' : 'bg-red-100'}`}>
            <div className="text-sm font-medium">Ollama AI</div>
            <div className={getStatusColor(systemStatus?.ollama_connected || false)}>
              {systemStatus?.ollama_connected ? '✅ 接続中' : '❌ 切断'}
            </div>
          </div>

          <div className={`p-4 rounded-lg shadow-md ${systemStatus?.microphone_available ? 'bg-green-100' : 'bg-red-100'}`}>
            <div className="text-sm font-medium">マイク</div>
            <div className={getStatusColor(systemStatus?.microphone_available || false)}>
              {systemStatus?.microphone_available ? '✅ 利用可能' : '❌ 利用不可'}
            </div>
          </div>

          <div className="p-4 rounded-lg shadow-md bg-purple-100">
            <div className="text-sm font-medium">会話履歴</div>
            <div className="text-purple-800 font-mono">
              {conversation.length} turns
            </div>
          </div>

          <div className="p-4 rounded-lg shadow-md bg-gray-100">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="w-full text-gray-700 hover:text-gray-900"
            >
              ⚙️ 設定
            </button>
          </div>
        </div>

        {/* 設定パネル */}
        {showSettings && systemStatus && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 className="text-2xl font-semibold mb-4">🛠️ システム状態</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className={`p-3 rounded ${systemStatus.ollama_connected ? 'bg-green-100' : 'bg-red-100'}`}>
                <div className="text-sm font-medium">Ollama AI</div>
                <div className={getStatusColor(systemStatus.ollama_connected)}>
                  {systemStatus.ollama_connected ? '✅ OK' : '❌ NG'}
                </div>
              </div>
              <div className={`p-3 rounded ${systemStatus.speaker_available ? 'bg-green-100' : 'bg-red-100'}`}>
                <div className="text-sm font-medium">音声合成</div>
                <div className={getStatusColor(systemStatus.speaker_available)}>
                  {systemStatus.speaker_available ? '✅ OK' : '❌ NG'}
                </div>
              </div>
              <div className="p-3 rounded bg-blue-100">
                <div className="text-sm font-medium">利用可能音声</div>
                <div className="text-blue-800">
                  {systemStatus.available_voices} voices
                </div>
              </div>
            </div>
            <div className="mt-4 flex gap-2">
              <button
                onClick={fetchSystemStatus}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                🔄 更新
              </button>
              <button
                onClick={resetConversation}
                className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
              >
                🗑️ 会話履歴リセット
              </button>
            </div>
          </div>
        )}

        {/* メイン会話エリア */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* 録音コントロール */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4">🎙️ 音声録音</h2>
            
            <div className="space-y-4">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
                className={`w-full py-4 px-6 rounded-lg font-semibold text-xl transition-all ${
                  isRecording 
                    ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse' 
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isRecording ? '⏹️ 録音停止' : (isProcessing ? '🤖 処理中...' : '🎤 録音開始')}
              </button>

              {/* 処理中インジケータ */}
              {isProcessing && (
                <div className="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-400">
                  <div className="text-blue-700 flex items-center">
                    <div className="animate-spin mr-2">🤖</div>
                    AI が応答を処理中...
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 会話履歴 */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-semibold">💬 会話履歴</h2>
              <div className="text-sm text-gray-500">
                最新のターンが上に表示されます
              </div>
            </div>

            <div className="space-y-4 max-h-96 overflow-y-auto">
              {conversation.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <div className="text-6xl mb-4">🎤</div>
                  <div className="text-xl">録音ボタンを押して英会話を始めましょう！</div>
                  <div className="text-sm mt-2">音声認識 + AI応答</div>
                </div>
              ) : (
                conversation.slice().reverse().map((msg, index) => (
                  <div key={index} className="border-l-4 border-blue-200 pl-4 py-3">
                    <div className="mb-2">
                      <span className="font-medium text-blue-600">👤 あなた:</span>
                      <span className="ml-2 text-lg">{msg.user}</span>
                    </div>
                    <div className="mb-1">
                      <span className="font-medium text-green-600">🤖 AI:</span>
                      <span className="ml-2 text-lg">{msg.ai}</span>
                    </div>
                    <div className="text-xs text-gray-400 mt-2">
                      {msg.timestamp}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* システム情報 */}
        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">📊 システム情報</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-mono text-blue-600">
                {systemStatus?.ollama_connected ? '✅' : '❌'}
              </div>
              <div className="text-sm text-gray-600">AI Model (Llama3)</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-mono text-green-600">
                {systemStatus?.microphone_available ? '✅' : '❌'}
              </div>
              <div className="text-sm text-gray-600">マイク</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-mono text-purple-600">
                {systemStatus?.speaker_available ? '✅' : '❌'}
              </div>
              <div className="text-sm text-gray-600">音声合成</div>
            </div>
          </div>
          <div className="mt-4 text-center text-sm text-gray-500">
            音声認識: faster-whisper | AI: Ollama Llama3 8B | 音声合成: XTTS-v2
          </div>
        </div>
      </div>
    </div>
  );
}