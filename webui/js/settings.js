// User settings, kept in localStorage.  Unset sampling values (null) are not
// sent, so the server's own defaults apply.

const KEY = 'q36-webui-settings';

export const THINKING_MODES = ['default', 'off', 'low', 'medium', 'high', 'xhigh', 'max'];

export const DEFAULTS = {
  lang: 'auto',
  theme: 'auto',
  thinking: 'default',
  sendReasoning: true,
  thinkingOpen: 'streaming', // streaming | always | never
  enterToSend: true,
  fontSize: 15,
  wide: false,
  autoTitle: false,
  showStats: true,
  system: '',
  gen: {
    temperature: null, top_p: null, top_k: null, min_p: null,
    presence_penalty: null, frequency_penalty: null,
    max_tokens: null, seed: null, stop: '',
  },
};

function merge(base, over) {
  const out = { ...base };
  if (!over || typeof over !== 'object') return out;
  for (const k of Object.keys(base)) {
    if (!(k in over)) continue;
    if (base[k] && typeof base[k] === 'object') out[k] = merge(base[k], over[k]);
    else if (typeof over[k] === typeof base[k] || base[k] === null) out[k] = over[k];
  }
  return out;
}

export function loadSettings(storage = globalThis.localStorage) {
  try {
    const raw = storage && storage.getItem(KEY);
    return merge(DEFAULTS, raw ? JSON.parse(raw) : null);
  } catch {
    return merge(DEFAULTS, null);
  }
}

export function saveSettings(settings, storage = globalThis.localStorage) {
  try { storage.setItem(KEY, JSON.stringify(settings)); } catch { /* private mode */ }
}

export function resetGen(settings) {
  settings.gen = { ...DEFAULTS.gen };
}
