/* Jarvis voice console: always-listening mic, spoken replies, RAM-only history.
 *
 * Mic: getUserMedia @16kHz mono -> AudioWorklet -> int16 WS frames.
 * Playback: scheduled WebAudio queue (stoppable for barge-in).
 * Barge-in: mic energy above threshold while speaking -> stop frame.
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const orb = $('orb'), orbGlyph = $('orbGlyph'), stateLine = $('stateLine');
  const liveLine = $('liveLine'), vuFill = $('vuFill'), turns = $('turns');
  const btnListen = $('btnListen'), btnMute = $('btnMute');
  const textIn = $('textIn'), btnSend = $('btnSend'), threshold = $('threshold');
  const conn = $('conn');

  const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/voice';

  let ws = null;
  let listening = false;
  let muted = false;
  let micStream = null;
  let audioCtx = null;
  let analyser = null;
  let playCtx = null;
  let playQueue = [];      // AudioBufferSourceNodes currently scheduled
  let speaking = false;    // server says audio is coming / playing
  let backoff = 500;

  const STATES = {
    idle: ['○', 'press Start and talk'],
    listening: ['◉', 'listening…'],
    endpointing: ['◉', 'got it, thinking…'],
    thinking: ['✦', 'thinking…'],
    speaking: ['♪', 'speaking… (talk to interrupt)'],
  };

  function setState(s) {
    const [glyph, label] = STATES[s] || STATES.idle;
    orb.className = 'orb ' + s;
    orbGlyph.textContent = glyph;
    stateLine.textContent = label;
  }

  function addTurn(cls, text) {
    const div = document.createElement('div');
    div.className = cls;
    div.textContent = text;
    turns.appendChild(div);
    while (turns.children.length > 30) turns.firstChild.remove();
  }

  function floatTo16(buf) {
    const out = new Int16Array(buf.length);
    for (let i = 0; i < buf.length; i++) {
      const s = Math.max(-1, Math.min(1, buf[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out.buffer;
  }

  function micLevel() {
    if (!analyser) return 0;
    const arr = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(arr);
    let peak = 0;
    for (let i = 0; i < arr.length; i += 4) {
      const v = Math.abs(arr[i]);
      if (v > peak) peak = v;
    }
    return peak;
  }

  function connect() {
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      conn.textContent = '● CONNECTED';
      conn.className = 'conn on';
      backoff = 500;
      if (listening) ws.send(JSON.stringify({ type: 'start' }));
    };
    ws.onclose = () => {
      conn.textContent = '● OFFLINE';
      conn.className = 'conn off';
      stopPlayback();
      setTimeout(() => {
        backoff = Math.min(backoff * 2, 8000);
        if (listening) connect();
      }, backoff);
    };
    ws.onmessage = async (ev) => {
      if (typeof ev.data === 'string') {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'state') setState(msg.state);
        else if (msg.type === 'transcript') {
          liveLine.textContent = '“' + msg.text + '”';
          addTurn('you', msg.text);
        } else if (msg.type === 'reply') {
          addTurn('jarvis', msg.text);
          liveLine.textContent = '';
        } else if (msg.type === 'error') {
          stateLine.textContent = '⚠️ ' + msg.message;
        }
        return;
      }
      // Binary: one spoken sentence (WAV). Schedule gaplessly.
      if (muted) return;
      await playWav(ev.data);
    };
  }

  async function ensurePlayCtx() {
    if (!playCtx) playCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (playCtx.state === 'suspended') await playCtx.resume();
    return playCtx;
  }

  async function playWav(buf) {
    const ctx = await ensurePlayCtx();
    const audio = await ctx.decodeAudioData(buf.slice(0));
    const src = ctx.createBufferSource();
    src.buffer = audio;
    src.connect(ctx.destination);
    speaking = true;
    setState('speaking');
    src.onended = () => {
      playQueue = playQueue.filter((s) => s !== src);
      if (playQueue.length === 0 && speaking) {
        speaking = false;
        setState('listening');
      }
    };
    playQueue.push(src);
    src.start();
  }

  function stopPlayback() {
    for (const src of playQueue) {
      try { src.stop(); } catch (_) { /* already ended */ }
    }
    playQueue = [];
    speaking = false;
  }

  async function startListening() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      stateLine.textContent = '⚠️ Mic blocked — allow microphone access and retry.';
      return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    await audioCtx.audioWorklet.addModule('/static/worklet.js');
    const src = audioCtx.createMediaStreamSource(micStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    const worklet = new AudioWorkletNode(audioCtx, 'mic-capture');
    src.connect(analyser);
    src.connect(worklet);
    worklet.port.onmessage = (e) => {
      if (ws && ws.readyState === WebSocket.OPEN && listening) {
        ws.send(floatTo16(e.data));
      }
    };
    window._worklet = worklet;

    if (!ws || ws.readyState !== WebSocket.OPEN) connect();
    else ws.send(JSON.stringify({ type: 'start' }));
    listening = true;
    btnListen.textContent = '■ Stop listening';
    btnListen.classList.add('live');
    setState('listening');
    requestAnimationFrame(vuLoop);
  }

  function stopListening() {
    listening = false;
    stopPlayback();
    if (window._worklet) {
      try { window._worklet.port.postMessage({ cmd: 'stop' }); } catch (_) {}
      window._worklet = null;
    }
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (audioCtx) {
      audioCtx.close().catch(() => {});
      audioCtx = null;
      analyser = null;
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }));
    }
    btnListen.textContent = '▶ Start listening';
    btnListen.classList.remove('live');
    setState('idle');
    vuFill.style.width = '0%';
  }

  function vuLoop() {
    if (!listening || !analyser) return;
    const level = micLevel();
    vuFill.style.width = Math.min(100, Math.round(level * 220)) + '%';
    // Barge-in: audible mic energy while the reply is playing.
    const thresh = (threshold.value / 32768) * 4;
    if (speaking && level > Math.max(thresh, 0.08)) {
      stopPlayback();
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'barge' }));
      }
      setState('listening');
    }
    requestAnimationFrame(vuLoop);
  }

  btnListen.addEventListener('click', () => (listening ? stopListening() : startListening()));

  btnMute.addEventListener('click', () => {
    muted = !muted;
    btnMute.textContent = muted ? '🔈 Unmute speaker' : '🔇 Mute speaker';
    if (muted) stopPlayback();
  });

  function sendText() {
    const text = textIn.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    addTurn('you', text);
    ws.send(JSON.stringify({ type: 'text', text }));
    textIn.value = '';
  }
  btnSend.addEventListener('click', sendText);
  textIn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendText();
  });

  connect();
})();
