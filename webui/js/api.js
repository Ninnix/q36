// Client for the q36-server HTTP API (OpenAI-compatible chat completions with
// SSE streaming, plus /props for server facts).

export class ApiError extends Error {
  constructor(message, { status = 0, code = '', info = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.info = info;
  }
}

async function errorFromResponse(res) {
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  const e = body && body.error ? body.error : {};
  return new ApiError(e.message || `HTTP ${res.status}`, { status: res.status, code: e.code || '', info: e });
}

export async function fetchServerInfo(base = '') {
  const info = { model: '', n_ctx: 0, vision: null, default_max_tokens: 0, slots: 1 };
  try {
    const res = await fetch(base + '/props', { cache: 'no-store' });
    if (res.ok) Object.assign(info, await res.json());
  } catch { /* try /v1/models below */ }
  if (!info.model || !info.n_ctx) {
    const res = await fetch(base + '/v1/models', { cache: 'no-store' });
    if (!res.ok) throw new ApiError(`HTTP ${res.status}`, { status: res.status });
    const m = (await res.json()).data?.[0];
    if (m) {
      info.model = info.model || m.id;
      info.n_ctx = info.n_ctx || m.context_length || 0;
    }
  }
  return info;
}

export async function ping(base = '') {
  try {
    const res = await fetch(base + '/health', { cache: 'no-store' });
    return res.ok;
  } catch {
    return false;
  }
}

// Incremental SSE parser: feed() decoded text, get complete `data:` payloads.
export class SseParser {
  constructor() { this.buf = ''; }
  feed(text) {
    this.buf += text;
    const events = [];
    let idx;
    while ((idx = this.buf.search(/\r?\n\r?\n/)) >= 0) {
      const block = this.buf.slice(0, idx);
      this.buf = this.buf.slice(idx + (this.buf[idx] === '\r' ? 4 : 2));
      const data = block.split(/\r?\n/).filter((l) => l.startsWith('data:')).map((l) => l.slice(5).replace(/^ /, '')).join('\n');
      if (data) events.push(data);
    }
    return events;
  }
}

/**
 * Streams one chat completion.  Callbacks receive text deltas as they arrive.
 * Resolves with { finish, usage }.  Rejects with ApiError, or AbortError when
 * `signal` fires (the caller keeps whatever was streamed so far).
 */
export async function streamChat({ base = '', body, signal, onReasoning, onContent, onFirstToken }) {
  const res = await fetch(base + '/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, stream: true, stream_options: { include_usage: true } }),
    signal,
  });
  if (!res.ok) throw await errorFromResponse(res);
  if (!res.body) throw new ApiError('Streaming is not supported by this browser');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const sse = new SseParser();
  let finish = null;
  let usage = null;
  let first = true;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const data of sse.feed(decoder.decode(value, { stream: true }))) {
      if (data === '[DONE]') continue;
      let chunk;
      try { chunk = JSON.parse(data); } catch { continue; }
      if (chunk.error) throw new ApiError(chunk.error.message || 'Server error', { code: chunk.error.code || '', info: chunk.error });
      if (chunk.usage) usage = chunk.usage;
      const choice = chunk.choices && chunk.choices[0];
      if (!choice) continue;
      const delta = choice.delta || {};
      const reasoning = delta.reasoning_content ?? delta.reasoning;
      if (reasoning) {
        if (first) { first = false; onFirstToken && onFirstToken(); }
        onReasoning && onReasoning(reasoning);
      }
      if (delta.content) {
        if (first) { first = false; onFirstToken && onFirstToken(); }
        onContent && onContent(delta.content);
      }
      if (choice.finish_reason) finish = choice.finish_reason;
    }
  }
  return { finish, usage };
}

export async function completeChat({ base = '', body, signal }) {
  const res = await fetch(base + '/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, stream: false }),
    signal,
  });
  if (!res.ok) throw await errorFromResponse(res);
  const json = await res.json();
  return json.choices?.[0]?.message?.content || '';
}

// ------------------------------------------------------------- requests ----

function attachmentText(att) {
  return `<attachment name=${JSON.stringify(att.name)}>\n${att.data}\n</attachment>`;
}

export function userContent(node) {
  const files = (node.attachments || []).filter((a) => a.kind === 'text');
  const images = (node.attachments || []).filter((a) => a.kind === 'image');
  let text = node.content || '';
  if (files.length) text = files.map(attachmentText).join('\n\n') + (text ? '\n\n' + text : '');
  if (!images.length) return text;
  const parts = images.map((a) => ({ type: 'image_url', image_url: { url: a.data } }));
  parts.push({ type: 'text', text: text || ' ' });
  return parts;
}

/** Turns the visible path into OpenAI messages. */
export function buildMessages(path, { system, sendReasoning }) {
  const messages = [];
  if (system && system.trim()) messages.push({ role: 'system', content: system });
  for (const n of path) {
    if (n.role === 'user') {
      messages.push({ role: 'user', content: userContent(n) });
    } else if (n.role === 'assistant') {
      if (!n.content && !n.reasoning) continue;
      const m = { role: 'assistant', content: n.content || '' };
      if (sendReasoning && n.reasoning) m.reasoning_content = n.reasoning;
      messages.push(m);
    }
  }
  return messages;
}

const THINKING_EFFORT = new Set(['low', 'medium', 'high', 'xhigh', 'max']);

/** Sampling and reasoning fields for the request; unset values use server defaults. */
export function buildParams(settings) {
  const p = {};
  const g = settings.gen;
  for (const k of ['temperature', 'top_p', 'min_p', 'presence_penalty', 'frequency_penalty']) {
    if (typeof g[k] === 'number' && Number.isFinite(g[k])) p[k] = g[k];
  }
  if (Number.isInteger(g.top_k)) p.top_k = g.top_k;
  if (Number.isInteger(g.max_tokens) && g.max_tokens > 0) p.max_tokens = g.max_tokens;
  if (Number.isInteger(g.seed) && g.seed > 0) p.seed = g.seed;
  const stops = (g.stop || '').split('\n').map((s) => s.replace(/\\n/g, '\n')).filter((s) => s !== '');
  if (stops.length) p.stop = stops;
  if (settings.thinking === 'off') p.chat_template_kwargs = { enable_thinking: false };
  else if (THINKING_EFFORT.has(settings.thinking)) p.reasoning_effort = settings.thinking;
  return p;
}
