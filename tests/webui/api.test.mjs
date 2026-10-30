import test from 'node:test';
import assert from 'node:assert/strict';
import { SseParser, buildMessages, buildParams, userContent } from '../../webui/js/api.js';
import { DEFAULTS, loadSettings, saveSettings } from '../../webui/js/settings.js';

test('SseParser handles split chunks and CRLF', () => {
  const p = new SseParser();
  assert.deepEqual(p.feed('data: {"a":1}\n\ndata: {"b"'), ['{"a":1}']);
  assert.deepEqual(p.feed(':2}\n\n'), ['{"b":2}']);
  assert.deepEqual(p.feed('data: [DONE]\r\n\r\n'), ['[DONE]']);
  assert.deepEqual(p.feed(': keepalive\n\n'), []);
});

test('buildMessages resends reasoning_content only when enabled', () => {
  const path = [
    { role: 'root' },
    { role: 'user', content: 'q', attachments: [] },
    { role: 'assistant', content: 'a', reasoning: 'r' },
    { role: 'assistant', content: '', reasoning: '' },
  ];
  const on = buildMessages(path, { system: 'S', sendReasoning: true });
  assert.deepEqual(on.map((m) => m.role), ['system', 'user', 'assistant']);
  assert.equal(on[2].reasoning_content, 'r');
  const off = buildMessages(path, { system: '', sendReasoning: false });
  assert.equal(off[1].reasoning_content, undefined);
  assert.equal(off[0].role, 'user');
});

test('userContent wraps text files and emits image parts', () => {
  const text = userContent({ content: 'ask', attachments: [{ kind: 'text', name: 'a.txt', data: 'BODY' }] });
  assert.match(text, /<attachment name="a.txt">/);
  assert.match(text, /BODY/);
  assert.match(text, /ask$/);
  const parts = userContent({ content: 'look', attachments: [{ kind: 'image', name: 'i', data: 'data:image/png;base64,AA' }] });
  assert.equal(parts[0].type, 'image_url');
  assert.deepEqual(parts[parts.length - 1], { type: 'text', text: 'look' });
});

test('buildParams sends only configured values', () => {
  const s = structuredClone(DEFAULTS);
  assert.deepEqual(buildParams(s), {});
  s.gen.temperature = 0.3; s.gen.top_k = 20; s.gen.max_tokens = 512; s.gen.stop = 'END\n\\n\\n';
  s.thinking = 'high';
  assert.deepEqual(buildParams(s), { temperature: 0.3, top_k: 20, max_tokens: 512, stop: ['END', '\n\n'], reasoning_effort: 'high' });
  s.thinking = 'off';
  assert.deepEqual(buildParams(s).chat_template_kwargs, { enable_thinking: false });
  assert.equal(buildParams(s).reasoning_effort, undefined);
});

test('settings merge ignores unknown keys and wrong types', () => {
  const store = new Map();
  const storage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
  assert.deepEqual(loadSettings(storage), DEFAULTS);
  const s = loadSettings(storage);
  s.fontSize = 18; s.gen.temperature = 0.5;
  saveSettings(s, storage);
  store.set('q36-webui-settings', JSON.stringify({ ...JSON.parse(store.get('q36-webui-settings')), theme: 5, evil: 1 }));
  const back = loadSettings(storage);
  assert.equal(back.fontSize, 18);
  assert.equal(back.gen.temperature, 0.5);
  assert.equal(back.theme, 'auto');
  assert.equal(back.evil, undefined);
  store.set('q36-webui-settings', '{not json');
  assert.deepEqual(loadSettings(storage), DEFAULTS);
});
