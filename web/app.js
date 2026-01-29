const TARGET_SAMPLE_RATE = 16000;
const CHUNK_SIZE = 512;
const SILERO_THRESHOLD = 0.5;
const SMART_TURN_THRESHOLD = 0.5;

const toggleBtn = document.getElementById("toggleBtn");
const statusText = document.getElementById("statusText");
const statusDot = document.getElementById("statusDot");

const vadState = document.getElementById("vadState");
const vadProb = document.getElementById("vadProb");
const vadSilence = document.getElementById("vadSilence");
const turnMs = document.getElementById("turnMs");

const stEnd = document.getElementById("stEnd");
const stContinue = document.getElementById("stContinue");
const stLatency = document.getElementById("stLatency");
const stDecision = document.getElementById("stDecision");
const smartPanel = document.getElementById("smartPanel");

const logEl = document.getElementById("log");
const logHint = document.getElementById("logHint");
const chartSilero = document.getElementById("chartSilero");
const chartSmartTurn = document.getElementById("chartSmartTurn");
const chartSpectrum = document.getElementById("chartSpectrum");

let ws = null;
let audioCtx = null;
let mediaStream = null;
let workletNode = null;
let gainNode = null;
let analyserNode = null;
let resampleQueue = [];
let workletUrl = null;
let startTime = 0;
let sileroSeries = [];
let smartSeries = [];
let chartTimer = null;

const MAX_POINTS = 300;
const CHART_RANGE_MS = 10000;
const FFT_SIZE = 1024;
let spectrumBins = null;

function log(message) {
  const line = document.createElement("div");
  line.textContent = `${new Date().toLocaleTimeString()} · ${message}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(text, live = false) {
  statusText.textContent = text;
  statusDot.classList.toggle("live", live);
}

function pushPoint(series, timeMs, value) {
  series.push({ t: timeMs, v: value });
  if (series.length > MAX_POINTS) {
    series.splice(0, series.length - MAX_POINTS);
  }
}

function drawSeries(canvas, series, color, threshold) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = "#d7d1c9";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height * 0.2);
  ctx.lineTo(width, height * 0.2);
  ctx.moveTo(0, height * 0.5);
  ctx.lineTo(width, height * 0.5);
  ctx.moveTo(0, height * 0.8);
  ctx.lineTo(width, height * 0.8);
  ctx.stroke();

  if (typeof threshold === "number") {
    const y = height - Math.max(0, Math.min(1, threshold)) * height;
    ctx.save();
    ctx.strokeStyle = "#2d2a27";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
    ctx.restore();
  }

  if (!series.length) return;
  const latest = series[series.length - 1].t;
  const minT = latest - CHART_RANGE_MS;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  for (const point of series) {
    if (point.t < minT) continue;
    const x = ((point.t - minT) / CHART_RANGE_MS) * width;
    const y = height - Math.max(0, Math.min(1, point.v)) * height;
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

function renderCharts() {
  drawSeries(chartSilero, sileroSeries, "#e76f51", SILERO_THRESHOLD);
  drawSeries(chartSmartTurn, smartSeries, "#2a9d8f", SMART_TURN_THRESHOLD);
  if (analyserNode && chartSpectrum) {
    if (!spectrumBins || spectrumBins.length !== analyserNode.frequencyBinCount) {
      spectrumBins = new Uint8Array(analyserNode.frequencyBinCount);
    }
    analyserNode.getByteFrequencyData(spectrumBins);
    const ctx = chartSpectrum.getContext("2d");
    const width = chartSpectrum.width;
    const height = chartSpectrum.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f6f2ea";
    ctx.fillRect(0, 0, width, height);
    const barCount = spectrumBins.length;
    const barWidth = width / barCount;
    ctx.fillStyle = "#264653";
    for (let i = 0; i < barCount; i += 1) {
      const value = spectrumBins[i] / 255;
      const barHeight = value * height;
      ctx.fillRect(i * barWidth, height - barHeight, barWidth * 0.9, barHeight);
    }
  }
  chartTimer = requestAnimationFrame(renderCharts);
}

function wsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws`;
}

function downsampleBuffer(buffer, inRate, outRate) {
  if (outRate === inRate) return buffer;
  if (outRate > inRate) return buffer;
  const sampleRateRatio = inRate / outRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function queueSamples(samples) {
  for (let i = 0; i < samples.length; i += 1) {
    resampleQueue.push(samples[i]);
  }
  while (resampleQueue.length >= CHUNK_SIZE) {
    const chunk = new Float32Array(resampleQueue.splice(0, CHUNK_SIZE));
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(chunk);
    }
  }
}

function handlePCM(pcm) {
  if (!audioCtx) return;
  const resampled = downsampleBuffer(pcm, audioCtx.sampleRate, TARGET_SAMPLE_RATE);
  queueSamples(resampled);
}

function handleMessage(data) {
  if (data.type === "vad") {
    vadState.textContent = data.state;
    vadProb.textContent = data.p.toFixed(2);
    vadSilence.textContent = `${data.silence_ms} ms`;
    turnMs.textContent = `${data.turn_ms} ms`;
    const now = performance.now() - startTime;
    pushPoint(sileroSeries, now, data.p);
  } else if (data.type === "smart_turn") {
    stEnd.textContent = data.scores.end.toFixed(2);
    stContinue.textContent = data.scores.continue.toFixed(2);
    stLatency.textContent = `${data.latency_ms} ms`;
    stDecision.textContent = data.decision.toLowerCase();
    const now = performance.now() - startTime;
    pushPoint(smartSeries, now, data.scores.end);
    if (data.decision === "END") {
      smartPanel.classList.remove("flash");
      void smartPanel.offsetWidth;
      smartPanel.classList.add("flash");
      log(`END turn detected (score ${data.scores.end.toFixed(2)})`);
    } else {
      log(`CONTINUE turn (score ${data.scores.end.toFixed(2)})`);
    }
  } else if (data.type === "error") {
    log(`Server error: ${data.message}`);
    setStatus("Error", false);
  }
}

async function start() {
  resampleQueue = [];
  logHint.textContent = "listening";
  setStatus("Connecting...", false);
  startTime = performance.now();
  sileroSeries = [];
  smartSeries = [];
  if (!chartTimer) {
    chartTimer = requestAnimationFrame(renderCharts);
  }

  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    ws.send(
      JSON.stringify({ type: "init", sample_rate: TARGET_SAMPLE_RATE, format: "f32le", channels: 1 })
    );
    setStatus("Live", true);
    log("WebSocket connected");
  };
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const data = JSON.parse(event.data);
      handleMessage(data);
    }
  };
  ws.onclose = () => {
    setStatus("Disconnected", false);
    log("WebSocket closed");
  };
  ws.onerror = () => {
    setStatus("Error", false);
    log("WebSocket error");
  };

  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  if (!workletUrl) {
    const workletCode = `
      class PCMCaptureProcessor extends AudioWorkletProcessor {
        process(inputs) {
          const input = inputs[0];
          if (input && input[0]) {
            this.port.postMessage(input[0]);
          }
          return true;
        }
      }
      registerProcessor('pcm-capture', PCMCaptureProcessor);
    `;
    workletUrl = URL.createObjectURL(new Blob([workletCode], { type: "application/javascript" }));
  }
  await audioCtx.audioWorklet.addModule(workletUrl);

  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, "pcm-capture");
  workletNode.port.onmessage = (event) => handlePCM(event.data);

  gainNode = audioCtx.createGain();
  gainNode.gain.value = 0;

  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = FFT_SIZE;
  analyserNode.smoothingTimeConstant = 0.7;

  source.connect(workletNode);
  source.connect(analyserNode);
  workletNode.connect(gainNode);
  analyserNode.connect(gainNode);
  gainNode.connect(audioCtx.destination);
  if (audioCtx.state === "suspended") {
    await audioCtx.resume();
  }
}

async function stop() {
  logHint.textContent = "ready";
  setStatus("Stopped", false);
  if (chartTimer) {
    cancelAnimationFrame(chartTimer);
    chartTimer = null;
  }

  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (gainNode) {
    gainNode.disconnect();
    gainNode = null;
  }
  if (analyserNode) {
    analyserNode.disconnect();
    analyserNode = null;
  }
  if (audioCtx) {
    await audioCtx.close();
    audioCtx = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}

let running = false;

toggleBtn.addEventListener("click", async () => {
  if (!running) {
    toggleBtn.textContent = "Stop";
    running = true;
    try {
      await start();
    } catch (err) {
      log(`Start failed: ${err.message || err}`);
      toggleBtn.textContent = "Start";
      running = false;
      await stop();
    }
  } else {
    toggleBtn.textContent = "Start";
    running = false;
    await stop();
  }
});
