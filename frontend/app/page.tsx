'use client';

import { useState, useRef, useEffect } from 'react';

interface ChatMessage {
  user: string;
  ai: string;
  timestamp: string;
}

interface SystemStatus {
  ollama_connected: boolean;
  microphone_available: boolean;
  speaker_available: boolean;
  available_voices: number;
  available_microphones: number;
}

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const [debugMode, setDebugMode] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  const API_BASE = 'http://localhost:8000';

  // システム状態を取得
  useEffect(() => {
    checkSystemStatus();
    
    // ブラウザの音声サポート情報をログ出力
    console.log('MediaRecorder support check:');
    console.log('audio/wav:', MediaRecorder.isTypeSupported('audio/wav'));
    console.log('audio/webm:', MediaRecorder.isTypeSupported('audio/webm'));
    console.log('audio/webm;codecs=opus:', MediaRecorder.isTypeSupported('audio/webm;codecs=opus'));
    console.log('audio/mp4:', MediaRecorder.isTypeSupported('audio/mp4'));
    console.log('audio/ogg;codecs=opus:', MediaRecorder.isTypeSupported('audio/ogg;codecs=opus'));
    
    // Web Speech API のサポート確認
    console.log('Web Speech API support:');
    console.log('SpeechRecognition:', 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window);
    console.log('SpeechSynthesis:', 'speechSynthesis' in window);
    
    // 音声合成の声をロード
    if ('speechSynthesis' in window) {
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        console.log('利用可能な音声:', voices.length);
        voices.forEach(voice => {
          if (voice.lang.startsWith('en')) {
            console.log(`英語音声: ${voice.name} (${voice.lang})`);
          }
        });
      };
      
      if (window.speechSynthesis.getVoices().length > 0) {
        loadVoices();
      } else {
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }
    }
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/system/status`);
      const status = await response.json();
      setSystemStatus(status);
    } catch (error) {
      console.error('システム状態取得エラー:', error);
    }
  };

  // マイクアクセス許可を取得
  const requestMicrophoneAccess = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      return stream;
    } catch (error) {
      console.error('マイクアクセスエラー:', error);
      alert('マイクアクセスが拒否されました。ブラウザの設定を確認してください。');
      return null;
    }
  };

  // WebAudio APIを使用してWAV形式で録音
  const startRecording = async () => {
    const stream = audioStream || await requestMicrophoneAccess();
    if (!stream) return;

    try {
      // AudioContextを作成
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      
      let audioData: Float32Array[] = [];
      
      processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        audioData.push(new Float32Array(inputData));
      };
      
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      // 録音停止時の処理を設定
      const stopRecordingInternal = () => {
        processor.disconnect();
        source.disconnect();
        audioContext.close();
        
        // Float32ArrayをWAVファイルに変換
        const wavBlob = createWavBlob(audioData, audioContext.sampleRate);
        console.log(`WAV blob created: size=${wavBlob.size}, type=${wavBlob.type}`);
        
        if (wavBlob.size === 0) {
          alert('音声が録音されませんでした。マイクの権限を確認してください。');
          return;
        }
        
        processSpeechChat(wavBlob, 'wav');
      };
      
      // グローバルに停止関数を保存
      (window as any).stopRecordingInternal = stopRecordingInternal;
      
      setIsRecording(true);
      console.log('WAV録音開始');
      
    } catch (error) {
      console.error('録音開始エラー:', error);
      alert('録音の開始に失敗しました');
    }
  };

  // Float32ArrayからWAVファイルを作成
  const createWavBlob = (audioData: Float32Array[], sampleRate: number): Blob => {
    const length = audioData.reduce((acc, chunk) => acc + chunk.length, 0);
    
    if (length === 0) {
      console.error('音声データが空です');
      return new Blob([], { type: 'audio/wav' });
    }
    
    console.log(`WAV作成: サンプル数=${length}, サンプルレート=${sampleRate}Hz, 長さ=${length/sampleRate}秒`);
    
    const buffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(buffer);
    
    // WAVヘッダーを書き込み（16bit PCM, モノラル）
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);  // ファイルサイズ - 8
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);              // PCMフォーマットサイズ
    view.setUint16(20, 1, true);               // フォーマットタイプ（PCM）
    view.setUint16(22, 1, true);               // チャンネル数（モノラル）
    view.setUint32(24, sampleRate, true);      // サンプルレート
    view.setUint32(28, sampleRate * 2, true);  // バイトレート
    view.setUint16(32, 2, true);               // ブロックアラインメント
    view.setUint16(34, 16, true);              // ビット深度
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);      // データサイズ
    
    // 音声データを書き込み（16bit）
    let offset = 44;
    for (const chunk of audioData) {
      for (let i = 0; i < chunk.length; i++) {
        const sample = Math.max(-1, Math.min(1, chunk[i]));
        view.setInt16(offset, sample * 0x7FFF, true);
        offset += 2;
      }
    }
    
    console.log(`WAVファイル生成完了: ${buffer.byteLength} bytes`);
    return new Blob([buffer], { type: 'audio/wav' });
  };

  // 音声録音停止
  const stopRecording = () => {
    if ((window as any).stopRecordingInternal) {
      (window as any).stopRecordingInternal();
      (window as any).stopRecordingInternal = null;
    }
    setIsRecording(false);
  };

  // 音声チャット処理
  const processSpeechChat = async (audioBlob: Blob, fileExtension: string = 'webm') => {
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, `recording.${fileExtension}`);

      const response = await fetch(`${API_BASE}/speech/chat`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        const newMessage: ChatMessage = {
          user: result.user_text,
          ai: result.ai_response,
          timestamp: new Date().toLocaleTimeString(),
        };
        setConversation(prev => [...prev, newMessage]);

        // AI応答を音声で再生
        if (result.ai_response) {
          console.log('AI応答の音声再生開始:', result.ai_response);
          await playTextToSpeech(result.ai_response);
          console.log('AI応答の音声再生完了');
        }
      } else {
        const errorData = await response.text();
        console.error('音声処理エラー詳細:', errorData);
        alert(`音声処理でエラーが発生しました: ${errorData}`);
      }
    } catch (error) {
      console.error('音声チャットエラー:', error);
      alert(`音声処理でエラーが発生しました: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsLoading(false);
    }
  };

  // テキストチャット
  const processTextChat = async () => {
    if (!textInput.trim()) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/chat/text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: textInput }),
      });

      if (response.ok) {
        const result = await response.json();
        const newMessage: ChatMessage = {
          user: textInput,
          ai: result.response,
          timestamp: new Date().toLocaleTimeString(),
        };
        setConversation(prev => [...prev, newMessage]);
        setTextInput('');

        // AI応答を音声で再生
        if (result.response) {
          await playTextToSpeech(result.response);
        }
      } else {
        alert('テキスト処理でエラーが発生しました');
      }
    } catch (error) {
      console.error('テキストチャットエラー:', error);
      alert('テキスト処理でエラーが発生しました');
    } finally {
      setIsLoading(false);
    }
  };

  // 音声合成・再生（ブラウザのWeb Speech API使用）
  const playTextToSpeech = async (text: string) => {
    try {
      console.log('TTS開始（Web Speech API使用）:', text);
      
      // 現在の音声を停止
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
      
      // Web Speech APIが使用可能かチェック
      if ('speechSynthesis' in window) {
        // 現在の発話を停止
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        
        // 英語音声を設定
        const voices = window.speechSynthesis.getVoices();
        const englishVoice = voices.find(voice => 
          voice.lang.startsWith('en') && 
          (voice.name.includes('English') || voice.name.includes('US') || voice.name.includes('GB'))
        );
        
        if (englishVoice) {
          utterance.voice = englishVoice;
          console.log('使用する音声:', englishVoice.name, englishVoice.lang);
        }
        
        // 音声設定
        utterance.rate = 0.9;    // 話速
        utterance.pitch = 1.0;   // 音程
        utterance.volume = 0.8;  // 音量
        
        // イベントリスナー
        utterance.onstart = () => {
          console.log('音声再生開始');
          setIsPlaying(true);
        };
        utterance.onend = () => {
          console.log('音声再生終了');
          setIsPlaying(false);
        };
        utterance.onerror = (e) => {
          console.error('音声再生エラー:', e);
          setIsPlaying(false);
        };
        
        // 音声再生
        window.speechSynthesis.speak(utterance);
      } else {
        console.error('Web Speech API（音声合成）がサポートされていません');
        // フォールバック: サーバーTTSを試行
        await playTextToSpeechFallback(text);
      }
    } catch (error) {
      console.error('音声合成エラー:', error);
      setIsPlaying(false);
    }
  };

  // フォールバック: サーバーTTS
  const playTextToSpeechFallback = async (text: string) => {
    try {
      console.log('フォールバックTTS開始:', text);
      const response = await fetch(`${API_BASE}/speech/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (response.ok) {
        const audioBlob = await response.blob();
        console.log('TTS音声ファイル受信:', audioBlob.size, 'bytes, type:', audioBlob.type);
        
        if (audioBlob.size === 0) {
          console.error('音声ファイルが空です');
          return;
        }
        
        // 現在再生中の音声があれば停止
        if (currentAudioRef.current) {
          currentAudioRef.current.pause();
          currentAudioRef.current = null;
        }
        
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        currentAudioRef.current = audio;
        
        audio.onloadeddata = () => console.log('音声データ読み込み完了');
        audio.onplay = () => {
          console.log('音声再生開始');
          setIsPlaying(true);
        };
        audio.onended = () => {
          console.log('音声再生終了');
          setIsPlaying(false);
          URL.revokeObjectURL(audioUrl);
          currentAudioRef.current = null;
        };
        audio.onerror = (e) => {
          console.error('音声再生エラー:', e);
          setIsPlaying(false);
          currentAudioRef.current = null;
        };
        
        await audio.play();
      } else {
        const errorText = await response.text();
        console.error('TTS APIエラー:', response.status, errorText);
      }
    } catch (error) {
      console.error('フォールバックTTS エラー:', error);
    }
  };

  // 会話履歴クリア
  const clearConversation = async () => {
    try {
      await fetch(`${API_BASE}/conversation/history`, { method: 'DELETE' });
      setConversation([]);
    } catch (error) {
      console.error('履歴クリアエラー:', error);
    }
  };

  // スピーカーテスト
  const testSpeaker = async () => {
    try {
      const response = await fetch(`${API_BASE}/test/speaker`);
      if (response.ok) {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
      }
    } catch (error) {
      console.error('スピーカーテストエラー:', error);
    }
  };

  // 音声ファイルアップロードテスト
  const testAudioUpload = async (audioBlob: Blob, fileExtension: string) => {
    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, `test.${fileExtension}`);
      
      const response = await fetch(`${API_BASE}/test/upload`, {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.json();
      console.log('アップロードテスト結果:', result);
      return result;
    } catch (error) {
      console.error('アップロードテストエラー:', error);
      return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* ヘッダー */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🎤 AI音声英会話アプリ
          </h1>
          <p className="text-lg text-gray-600">
            音声でAIと英語会話を練習しよう！
          </p>
        </div>

        {/* システム状態 */}
        {systemStatus && (
          <div className="bg-white rounded-lg shadow-md p-4 mb-6">
            <h2 className="text-xl font-semibold mb-3">🛠️ システム状態</h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
              <div className={`p-2 rounded ${systemStatus.ollama_connected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                <div>Ollama</div>
                <div>{systemStatus.ollama_connected ? '✅ 接続' : '❌ 未接続'}</div>
              </div>
              <div className={`p-2 rounded ${systemStatus.microphone_available ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                <div>マイク</div>
                <div>{systemStatus.microphone_available ? '✅ 利用可能' : '❌ 未利用'}</div>
              </div>
              <div className={`p-2 rounded ${systemStatus.speaker_available ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                <div>スピーカー</div>
                <div>{systemStatus.speaker_available ? '✅ 利用可能' : '❌ 未利用'}</div>
              </div>
              <div className="p-2 rounded bg-blue-100 text-blue-800">
                <div>音声種類</div>
                <div>{systemStatus.available_voices}種類</div>
              </div>
              <div className="p-2 rounded bg-blue-100 text-blue-800">
                <div>マイク数</div>
                <div>{systemStatus.available_microphones}個</div>
              </div>
            </div>
            <div className="mt-3 flex gap-2">
              <button
                onClick={checkSystemStatus}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                🔄 状態更新
              </button>
              <button
                onClick={testSpeaker}
                className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
              >
                🔊 スピーカーテスト
              </button>
              <button
                onClick={() => setDebugMode(!debugMode)}
                className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600"
              >
                🐛 {debugMode ? 'デバッグOFF' : 'デバッグON'}
              </button>
            </div>
          </div>
        )}

        {debugMode && (
          <div className="bg-yellow-50 rounded-lg shadow-md p-4 mb-6">
            <h2 className="text-xl font-semibold mb-3">🐛 デバッグ情報</h2>
            <div className="text-sm space-y-1">
              <div><strong>音声録音:</strong></div>
              <div>　MediaRecorder.isTypeSupported('audio/wav'): {MediaRecorder.isTypeSupported('audio/wav').toString()}</div>
              <div>　MediaRecorder.isTypeSupported('audio/webm'): {MediaRecorder.isTypeSupported('audio/webm').toString()}</div>
              <div>　MediaRecorder.isTypeSupported('audio/webm;codecs=opus'): {MediaRecorder.isTypeSupported('audio/webm;codecs=opus').toString()}</div>
              <div>　現在のオーディオストリーム: {audioStream ? '有効' : '無効'}</div>
              <div><strong>音声再生:</strong></div>
              <div>　SpeechSynthesis API: {'speechSynthesis' in window ? '対応' : '未対応'}</div>
              <div>　利用可能な音声数: {typeof window !== 'undefined' && 'speechSynthesis' in window ? window.speechSynthesis.getVoices().length : 0}</div>
              <div>　英語音声: {typeof window !== 'undefined' && 'speechSynthesis' in window ? 
                window.speechSynthesis.getVoices().filter(v => v.lang.startsWith('en')).length : 0}個</div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* 音声会話セクション */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">🎙️ 音声会話</h2>
            <div className="space-y-4">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isLoading}
                className={`w-full py-3 px-6 rounded-lg font-semibold transition-colors ${ 
                  isRecording 
                    ? 'bg-red-500 hover:bg-red-600 text-white' 
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                } disabled:opacity-50`}
              >
                {isRecording ? '⏹️ 録音停止' : '🎤 録音開始'}
              </button>
              {isRecording && (
                <div className="text-center text-red-600 font-medium">
                  🔴 録音中... 英語で話してください
                </div>
              )}
              {isLoading && (
                <div className="text-center text-blue-600 font-medium">
                  ⏳ 処理中...
                </div>
              )}
              {isPlaying && (
                <div className="text-center text-green-600 font-medium">
                  🔊 音声再生中...
                </div>
              )}
            </div>
          </div>

          {/* テキスト会話セクション */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">💬 テキスト会話</h2>
            <div className="space-y-4">
              <input
                type="text"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && processTextChat()}
                placeholder="英語で入力してください..."
                className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={processTextChat}
                disabled={isLoading || !textInput.trim()}
                className="w-full py-3 px-6 bg-green-500 hover:bg-green-600 text-white rounded-lg font-semibold disabled:opacity-50"
              >
                📤 送信
              </button>
            </div>
          </div>
        </div>

        {/* 会話履歴 */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">📝 会話履歴</h2>
            <button
              onClick={clearConversation}
              className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
            >
              🗑️ クリア
            </button>
          </div>
          
          {conversation.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              まだ会話がありません。音声またはテキストで話しかけてみましょう！
            </div>
          ) : (
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {conversation.map((msg, index) => (
                <div key={index} className="border-b border-gray-200 pb-4">
                  <div className="mb-2">
                    <span className="font-medium text-blue-600">👤 あなた:</span>
                    <span className="ml-2">{msg.user}</span>
                    <button
                      onClick={() => playTextToSpeech(msg.user)}
                      className="ml-2 text-sm text-blue-500 hover:text-blue-700"
                    >
                      🔊
                    </button>
                  </div>
                  <div className="mb-1">
                    <span className="font-medium text-green-600">🤖 AI:</span>
                    <span className="ml-2">{msg.ai}</span>
                    <button
                      onClick={() => playTextToSpeech(msg.ai)}
                      className="ml-2 text-sm text-green-500 hover:text-green-700"
                    >
                      🔊
                    </button>
                  </div>
                  <div className="text-xs text-gray-400">
                    {msg.timestamp}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
