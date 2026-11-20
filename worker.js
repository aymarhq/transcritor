// Motor de transcrição: Whisper rodando no navegador (WebGPU, com WASM de reserva).
import { pipeline, env, WhisperTextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

env.allowLocalModels = false;

const MODELOS = {
  rapido: 'onnx-community/whisper-base',
  equilibrado: 'onnx-community/whisper-small',
  precisao: 'onnx-community/whisper-large-v3-turbo',
  teste: 'onnx-community/whisper-tiny',
};

let pipe = null, modoAtual = null, device = null;

async function temWebGPU() {
  try {
    if (!navigator.gpu) return false;
    const a = await navigator.gpu.requestAdapter();
    return !!a && a.features.has('shader-f16');
  } catch { return false; }
}

async function carregar(modo) {
  if (pipe && modoAtual === modo) return pipe;
  if (pipe) { try { await pipe.dispose(); } catch {} pipe = null; }
  if (!device) device = (await temWebGPU()) ? 'webgpu' : 'wasm';
  const progress_callback = (p) => {
    if (p.status === 'progress_total') postMessage({ type: 'download', pct: Math.round(p.progress), loaded: p.loaded, total: p.total });
  };
  postMessage({ type: 'loading', modo, device });
  const opts = device === 'webgpu'
    ? { device, dtype: { encoder_model: 'fp16', decoder_model_merged: 'q4' }, progress_callback }
    : { device, dtype: 'q8', progress_callback };
  try {
    pipe = await pipeline('automatic-speech-recognition', MODELOS[modo], opts);
  } catch (e) {
    if (device !== 'webgpu') throw e;
    device = 'wasm';
    postMessage({ type: 'loading', modo, device });
    pipe = await pipeline('automatic-speech-recognition', MODELOS[modo], { device, dtype: 'q8', progress_callback });
  }
  modoAtual = modo;
  postMessage({ type: 'ready', modo, device });
  return pipe;
}

async function transcrever({ id, audio, idioma, modo }) {
  const p = await carregar(modo);
  const dur = audio.length / 16000;
  const janelas = 1 + Math.max(0, Math.ceil((dur - 30) / 20)); // chunk 30s, passo 20s
  let feitas = 0, parcial = '';
  const streamer = new WhisperTextStreamer(p.tokenizer, {
    skip_prompt: true,
    callback_function: (t) => {
      parcial = (parcial + t).slice(-140);
      postMessage({ type: 'progress', id, pct: Math.min(99, Math.round((feitas / janelas) * 100)), parcial: parcial.trim() });
    },
    on_finalize: () => {
      feitas++;
      postMessage({ type: 'progress', id, pct: Math.min(99, Math.round((feitas / janelas) * 100)), parcial: parcial.trim() });
    },
  });
  const opts = { task: 'transcribe', chunk_length_s: 30, stride_length_s: 5, return_timestamps: true, streamer };
  if (idioma && idioma !== 'auto') opts.language = idioma;
  const out = await p(audio, opts);
  const segs = (out.chunks || [])
    .map((c) => ({
      start: c.timestamp?.[0] ?? 0,
      end: c.timestamp?.[1] ?? c.timestamp?.[0] ?? dur,
      text: (c.text || '').trim(),
    }))
    .filter((s) => s.text);
  if (!segs.length && out.text?.trim()) segs.push({ start: 0, end: dur, text: out.text.trim() });
  postMessage({ type: 'done', id, segs, duracao: dur, device });
}

let fila = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  fila = fila.then(async () => {
    try {
      if (m.type === 'run') await transcrever(m);
    } catch (err) {
      postMessage({ type: 'error', id: m.id, message: String(err?.message || err).slice(0, 300) });
    }
  });
};
