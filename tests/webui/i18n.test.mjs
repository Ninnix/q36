import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { DICTIONARIES } from '../../webui/js/i18n.js';

const dir = new URL('../../webui/js/', import.meta.url);

test('every language has the same keys', () => {
  const en = Object.keys(DICTIONARIES.en).sort();
  for (const [code, dict] of Object.entries(DICTIONARIES)) {
    assert.deepEqual(Object.keys(dict).sort(), en, `language ${code} differs from en`);
  }
});

test('every literal t() key used in the code exists', () => {
  const missing = [];
  for (const f of readdirSync(dir).filter((n) => n.endsWith('.js') && n !== 'i18n.js')) {
    const src = readFileSync(new URL(f, dir), 'utf8');
    for (const m of src.matchAll(/\b(?:t|tr)\('([A-Za-z0-9_]+)'\s*[,)]/g)) {
      if (!(m[1] in DICTIONARIES.en)) missing.push(`${f}: ${m[1]}`);
    }
  }
  assert.deepEqual(missing, []);
});
