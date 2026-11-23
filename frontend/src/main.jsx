import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = 'http://127.0.0.1:8000';

const LANGUAGES = {
  en: 'English (English)',
  hi: 'Hindi (हिन्दी)',
  kn: 'Kannada (ಕನ್ನಡ)',
  ml: 'Malayalam (മലയാളം)',
  ta: 'Tamil (தமிழ்)',
  te: 'Telugu (తెలుగు)',
  bn: 'Bengali (বাংলা)',
  mr: 'Marathi (मराठी)',
  gu: 'Gujarati (ગુજરાતી)',
  pa: 'Punjabi (ਪੰਜਾਬੀ)',
  or: 'Odia (ଓଡ଼ିଆ)',
};

function App() {
  const [source, setSource] = useState('en');
  const [target, setTarget] = useState('ml');
  const [mode, setMode] = useState('general');
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [warnings, setWarnings] = useState([]);
  const [flags, setFlags] = useState({});
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [continuousMode, setContinuousMode] = useState(false);
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('indic_translator_history') || '[]');
    } catch {
      return [];
    }
  });

  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const audioURLRef = useRef(null);
  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);

  // Poll backend status on mount
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 8000);
    return () => clearInterval(interval);
  }, []);

  // Save history to local storage
  useEffect(() => {
    localStorage.setItem('indic_translator_history', JSON.stringify(history.slice(-50)));
  }, [history]);

  async function fetchStatus() {
    try {
      const res = await fetch(`${API}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        setError('');
      } else {
        setError('Backend reported an error. Check server logs.');
      }
    } catch {
      setError('Cannot connect to local backend (http://127.0.0.1:8000). Ensure uvicorn is running.');
    }
  }

  function swapLanguages() {
    setSource(target);
    setTarget(source);
    if (output) {
      setInput(output);
      setOutput('');
      setWarnings([]);
      setFlags({});
    }
  }

  async function handleTranslate(text = input) {
    if (!text.trim()) return;
    setBusy('Translating text…');
    setError('');
    setOutput('');
    setWarnings([]);
    setFlags({});

    try {
      const res = await fetch(`${API}/api/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source, target, mode }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Translation request failed');
      }

      const data = await res.json();
      setOutput(data.text);
      setWarnings(data.warnings || []);
      setFlags(data.flags || {});

      addToHistory({
        id: Date.now(),
        source,
        target,
        mode,
        original: text,
        translated: data.text,
        warnings: data.warnings || [],
        timestamp: new Date().toLocaleTimeString(),
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy('');
    }
  }

  async function handleSpeak(textToSpeak = output, lang = target) {
    if (!textToSpeak.trim()) return;
    setBusy('Synthesizing speech…');
    setError('');

    try {
      const res = await fetch(`${API}/api/speak`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToSpeak, lang }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Speech synthesis failed');
      }

      const blob = await res.blob();
      if (audioURLRef.current) URL.revokeObjectURL(audioURLRef.current);
      audioURLRef.current = URL.createObjectURL(blob);
      const audio = new Audio(audioURLRef.current);
      await audio.play();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy('');
    }
  }

  function downloadAudio() {
    if (!audioURLRef.current) return;
    const a = document.createElement('a');
    a.href = audioURLRef.current;
    a.download = `translation_${target}_${Date.now()}.wav`;
    a.click();
  }

  function copyTranslation() {
    if (!output) return;
    navigator.clipboard.writeText(output);
  }

  function addToHistory(item) {
    setHistory((prev) => [item, ...prev]);
  }

  function clearHistory() {
    setHistory([]);
    localStorage.removeItem('indic_translator_history');
  }

  function exportTranscript() {
    const content = history
      .map(
        (h) =>
          `[${h.timestamp}] (${LANGUAGES[h.source]} -> ${LANGUAGES[h.target]} | Mode: ${h.mode})\n` +
          `Original: ${h.original}\n` +
          `Translated: ${h.translated}\n` +
          (h.warnings.length ? `Advisories: ${h.warnings.join('; ')}\n` : '') +
          '--------------------------------------------------\n'
      )
      .join('\n');

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript_export_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Handle single-utterance microphone recording
  async function toggleRecording() {
    if (isRecording) {
      if (recorderRef.current && recorderRef.current.state === 'recording') {
        recorderRef.current.stop();
      }
      setIsRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
      mediaStreamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      const mr = new MediaRecorder(stream, { mimeType });
      recorderRef.current = mr;
      chunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setBusy('Transcribing & translating audio…');
        try {
          const blob = new Blob(chunksRef.current, { type: mimeType });
          const formData = new FormData();
          formData.append('file', blob, 'recording.webm');
          formData.append('source', source);
          formData.append('target', target);
          formData.append('mode', mode);

          const res = await fetch(`${API}/api/speech-to-speech`, {
            method: 'POST',
            body: formData,
          });

          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Speech translation failed');
          }

          const data = await res.json();
          setInput(data.original_text);
          setOutput(data.translated_text);
          setWarnings(data.warnings || []);
          setFlags(data.flags || {});

          if (data.original_text) {
            addToHistory({
              id: Date.now(),
              source,
              target,
              mode,
              original: data.original_text,
              translated: data.translated_text,
              warnings: data.warnings || [],
              timestamp: new Date().toLocaleTimeString(),
            });
          }

          if (data.audio_base64) {
            const byteChars = atob(data.audio_base64);
            const byteNumbers = new Array(byteChars.length);
            for (let i = 0; i < byteChars.length; i++) {
              byteNumbers[i] = byteChars.charCodeAt(i);
            }
            const audioBlob = new Blob([new Uint8Array(byteNumbers)], { type: 'audio/wav' });
            if (audioURLRef.current) URL.revokeObjectURL(audioURLRef.current);
            audioURLRef.current = URL.createObjectURL(audioBlob);
            new Audio(audioURLRef.current).play();
          }
        } catch (e) {
          setError(e.message);
        } finally {
          setBusy('');
        }
      };

      mr.start();
      setIsRecording(true);
      setBusy('Recording speech… Click Stop when finished.');
    } catch (e) {
      setError(`Microphone error: ${e.message}`);
    }
  }

  // Handle continuous conversation mode
  useEffect(() => {
    if (!continuousMode) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    const wsUrl = `ws://127.0.0.1:8000/ws/stream?source=${source}&target=${target}&mode=${mode}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
        });
        mediaStreamRef.current = stream;
        const mr = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

        mr.ondataavailable = (e) => {
          if (e.data && e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            ws.send(e.data);
          }
        };

        mr.start(800); // Send chunks every 800ms
        recorderRef.current = mr;
      } catch (e) {
        setError(`Continuous mode audio error: ${e.message}`);
        setContinuousMode(false);
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'utterance') {
          setInput(msg.original);
          setOutput(msg.translated);
          setWarnings(msg.warnings || []);
          setFlags(msg.flags || {});

          addToHistory({
            id: Date.now(),
            source,
            target,
            mode,
            original: msg.original,
            translated: msg.translated,
            warnings: msg.warnings || [],
            timestamp: new Date().toLocaleTimeString(),
          });

          if (msg.audio_base64) {
            const byteChars = atob(msg.audio_base64);
            const byteNumbers = new Array(byteChars.length);
            for (let i = 0; i < byteChars.length; i++) {
              byteNumbers[i] = byteChars.charCodeAt(i);
            }
            const audioBlob = new Blob([new Uint8Array(byteNumbers)], { type: 'audio/wav' });
            if (audioURLRef.current) URL.revokeObjectURL(audioURLRef.current);
            audioURLRef.current = URL.createObjectURL(audioBlob);
            new Audio(audioURLRef.current).play();
          }
        }
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    ws.onerror = () => {
      setError('Continuous WebSocket connection error');
      setContinuousMode(false);
    };

    return () => {
      if (recorderRef.current && recorderRef.current.state === 'recording') {
        recorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [continuousMode, source, target, mode]);

  return (
    <div className="app-container">
      <header>
        <div className="header-top">
          <div className="brand-eyebrow">Offline Indic Translator</div>
          <div className="status-pills">
            <span className={`pill ${status?.ready ? 'pill-success' : ''}`}>
              <span className="pill-dot"></span>
              {status?.ready ? 'Models Ready (Offline)' : 'Loading Models'}
            </span>
            {status?.gpu_name && status.gpu_name !== 'None' && (
              <span className="pill pill-gpu">
                GPU: {status.gpu_name.replace('NVIDIA GeForce ', '')} ({status.device?.toUpperCase()})
              </span>
            )}
            {status?.vram?.free_mb !== undefined && (
              <span className="pill">
                VRAM: {status.vram.free_mb} MB Free / {status.vram.total_mb} MB
              </span>
            )}
          </div>
        </div>
        <h1>Speech & Text Translator</h1>
        <p className="subtitle">
          Direct multilingual translation across 11 Indian languages powered by local AI.
        </p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="panel">
        <div className="mode-bar">
          <div className="modes">
            {['general', 'medical', 'office'].map((m) => (
              <button
                key={m}
                className={`mode-btn ${mode === m ? `active mode-${m}` : ''}`}
                onClick={() => setMode(m)}
              >
                {m === 'general' ? '💬 General' : m === 'medical' ? '⚕️ Medical' : '💼 Office'}
              </button>
            ))}
          </div>

          <label className="continuous-toggle">
            <input
              type="checkbox"
              checked={continuousMode}
              onChange={(e) => setContinuousMode(e.target.checked)}
            />
            ⚡ Continuous Conversation Mode
          </label>
        </div>

        <div className="lang-selector-bar">
          <div className="lang-select-group">
            <span className="lang-label">From Language</span>
            <select value={source} onChange={(e) => setSource(e.target.value)}>
              {Object.entries(LANGUAGES).map(([code, name]) => (
                <option key={code} value={code}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          <button className="swap-btn" onClick={swapLanguages} title="Swap languages">
            ⇄
          </button>

          <div className="lang-select-group">
            <span className="lang-label">To Language</span>
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              {Object.entries(LANGUAGES).map(([code, name]) => (
                <option key={code} value={code}>
                  {name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="translation-grid">
          {/* Source Card */}
          <div className="text-card">
            <div className="card-header">
              <span className="card-title">Original Speech & Text</span>
              {input && (
                <button
                  style={{ border: 'none', background: 'transparent', padding: '2px', fontSize: '11px', color: '#94a3b8' }}
                  onClick={() => setInput('')}
                >
                  Clear
                </button>
              )}
            </div>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type text or click 'Record Voice' to translate spoken speech…"
            />
            <div className="card-footer">
              <span className="char-count">{input.length} characters</span>
              <div className="card-actions">
                <button
                  className={`record-btn ${isRecording ? 'recording' : ''}`}
                  onClick={toggleRecording}
                  disabled={continuousMode || (busy !== '' && !busy.startsWith('Recording'))}
                >
                  {isRecording ? '■ Stop Recording' : '● Record Voice'}
                </button>
                <button
                  className="primary"
                  onClick={() => handleTranslate()}
                  disabled={!input.trim() || !!busy || isRecording}
                >
                  Translate Text
                </button>
              </div>
            </div>
          </div>

          {/* Target Card */}
          <div className="text-card">
            <div className="card-header">
              <span className="card-title">Translated Result</span>
              {output && (
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    style={{ border: 'none', background: 'transparent', padding: '2px', fontSize: '11px', color: '#3b82f6' }}
                    onClick={copyTranslation}
                  >
                    Copy
                  </button>
                </div>
              )}
            </div>
            <textarea
              readOnly
              value={output}
              placeholder="Translated speech and text will appear here…"
            />
            <div className="card-footer">
              <span className="char-count">{output.length} characters</span>
              <div className="card-actions">
                {audioURLRef.current && (
                  <button onClick={downloadAudio} title="Download synthesized speech as WAV">
                    ⬇ Save WAV
                  </button>
                )}
                <button
                  disabled={!output.trim() || !!busy}
                  onClick={() => handleSpeak()}
                >
                  ▶ Speak Translation
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Warnings / Clinical Review Alerts */}
        {flags?.negation_alert && (
          <div className="danger-box">
            <strong>CRITICAL CLINICAL DIRECTIVE ALERT:</strong>
            Negation directive ('no', 'not', 'stop') was detected in source instruction. Verify patient translation manually.
          </div>
        )}

        {warnings.length > 0 && (
          <div className="warning-box">
            <strong>{mode === 'medical' ? '⚕️ Clinical Verification Review:' : '⚠️ Advisory Notes:'}</strong>
            <ul style={{ paddingLeft: '18px', marginTop: '4px' }}>
              {warnings.map((w, idx) => (
                <li key={idx}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {busy && (
          <div className="busy-indicator">
            <span className="spinner"></span>
            <span>{busy}</span>
          </div>
        )}
      </section>

      {/* Local Conversation History */}
      {history.length > 0 && (
        <section className="history-section">
          <div className="history-header">
            <h3>Recent Conversation History ({history.length})</h3>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button onClick={exportTranscript}>📄 Export Transcript</button>
              <button onClick={clearHistory}>Clear History</button>
            </div>
          </div>
          <div className="history-list">
            {history.map((turn) => (
              <div key={turn.id} className="history-item">
                <div className="history-meta">
                  <span>
                    {LANGUAGES[turn.source]} → {LANGUAGES[turn.target]} • Mode:{' '}
                    <strong>{turn.mode}</strong>
                  </span>
                  <span>{turn.timestamp}</span>
                </div>
                <div className="history-content">
                  <div className="history-src">{turn.original}</div>
                  <div className="history-tgt">{turn.translated}</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
