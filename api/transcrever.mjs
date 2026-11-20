// Função da Vercel: recebe um pedaço de áudio (WAV) e repassa para a Groq (Whisper large-v3 turbo).
// Variáveis de ambiente na Vercel: GROQ_API_KEY e SITE_SENHA. A chave nunca vai para o navegador.
import { createHash, timingSafeEqual } from 'node:crypto';

const GROQ_URL = process.env.GROQ_URL || 'https://api.groq.com/openai/v1/audio/transcriptions';
const json = (data, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' } });

function senhaOk(recebida) {
  const certa = process.env.SITE_SENHA || '';
  if (!certa || !recebida) return false;
  const a = createHash('sha256').update(String(recebida)).digest();
  const b = createHash('sha256').update(certa).digest();
  return timingSafeEqual(a, b);
}

export function GET() {
  return json({ configurado: Boolean(process.env.GROQ_API_KEY && process.env.SITE_SENHA) });
}

export async function POST(request) {
  if (!process.env.GROQ_API_KEY || !process.env.SITE_SENHA) {
    return json({ erro: 'A nuvem ainda não foi configurada na Vercel (GROQ_API_KEY e SITE_SENHA).' }, 503);
  }
  if (!senhaOk(request.headers.get('x-senha'))) return json({ erro: 'Senha incorreta.' }, 401);

  const idioma = new URL(request.url).searchParams.get('idioma');
  const audio = await request.arrayBuffer();
  if (!audio.byteLength) return json({ erro: 'Áudio vazio.' }, 400);
  if (audio.byteLength > 4_400_000) return json({ erro: 'Pedaço de áudio grande demais.' }, 413);

  const fd = new FormData();
  fd.append('file', new Blob([audio], { type: 'audio/wav' }), 'audio.wav');
  fd.append('model', 'whisper-large-v3-turbo');
  fd.append('response_format', 'verbose_json');
  fd.append('temperature', '0');
  if (idioma && idioma !== 'auto') fd.append('language', idioma);

  let r;
  try {
    r = await fetch(GROQ_URL, { method: 'POST', headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` }, body: fd });
  } catch {
    return json({ erro: 'Não consegui falar com a Groq.' }, 502);
  }
  if (r.status === 429) {
    return json({ erro: 'Limite da Groq atingido.', retryAfter: Number(r.headers.get('retry-after')) || 20 }, 429);
  }
  if (!r.ok) {
    let msg = '';
    try { msg = (await r.json())?.error?.message || ''; } catch {}
    return json({ erro: `Groq recusou (${r.status}). ${msg}`.trim() }, r.status === 401 ? 500 : 502);
  }
  const d = await r.json();
  return json({
    language: d.language || null,
    text: (d.text || '').trim(),
    segments: (d.segments || []).map((s) => ({ start: s.start, end: s.end, text: String(s.text || '').trim() })).filter((s) => s.text),
  });
}
