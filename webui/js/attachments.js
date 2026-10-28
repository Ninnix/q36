// Turns files the user picks, pastes or drops into attachments.  Images are
// downscaled and re-encoded so a phone photo does not become a 20 MB request;
// text files are read as UTF-8 and sent inline.

import { uid } from './util.js';

export const MAX_IMAGE_EDGE = 1536;
export const MAX_TEXT_BYTES = 2 * 1024 * 1024;

const TEXT_EXT = new Set(`txt md markdown rst log csv tsv json jsonl yaml yml toml ini cfg conf env xml html htm css scss less js mjs cjs ts tsx jsx c h cc cpp cxx hpp hh m mm
  java kt kts scala go rs py rb php pl pm lua sh bash zsh fish ps1 bat sql r swift dart cs vb fs ex exs erl hs clj lisp el vim diff patch make mk cmake gradle tex bib
  svg proto graphql gql tf hcl dockerfile gitignore editorconfig`.split(/\s+/));

export function isImageFile(file) {
  return /^image\/(png|jpe?g|webp|gif|bmp)$/i.test(file.type);
}

export function looksLikeText(file) {
  if (file.type.startsWith('text/')) return true;
  if (/json|xml|javascript|yaml|x-sh|toml|sql/.test(file.type)) return true;
  const name = file.name.toLowerCase();
  const ext = name.includes('.') ? name.split('.').pop() : name;
  return TEXT_EXT.has(ext);
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Cannot decode image')); };
    img.src = url;
  });
}

async function processImage(file) {
  const img = await loadImage(file);
  const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext('2d');
  // JPEG has no alpha; flatten onto white so transparent PNGs stay readable.
  g.fillStyle = '#fff';
  g.fillRect(0, 0, w, h);
  g.drawImage(img, 0, 0, w, h);
  const data = canvas.toDataURL('image/jpeg', 0.9);
  const base = file.name.replace(/\.[^.]+$/, '') || 'image';
  return { id: uid(), kind: 'image', name: base + '.jpg', data, w, h };
}

async function processText(file) {
  if (file.size > MAX_TEXT_BYTES) throw new Error('too large');
  const text = await file.text();
  if (text.includes('\u0000')) throw new Error('binary');
  return { id: uid(), kind: 'text', name: file.name, data: text };
}

/** Resolves to { attachment } or { error: 'unsupported'|'noVision'|'tooLarge'|'binary'|'decode' }. */
export async function processFile(file, { vision }) {
  try {
    if (isImageFile(file)) {
      if (!vision) return { error: 'noVision' };
      return { attachment: await processImage(file) };
    }
    if (file.type.startsWith('image/')) return { error: 'unsupported' };
    if (looksLikeText(file) || file.type === '') return { attachment: await processText(file) };
    return { error: 'unsupported' };
  } catch (e) {
    if (e.message === 'too large') return { error: 'tooLarge' };
    if (e.message === 'binary') return { error: 'binary' };
    return { error: 'decode' };
  }
}
