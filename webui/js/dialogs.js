// Small modal / popover toolkit plus the app's dialogs (settings, preview,
// shortcuts, attachment viewers).

import { h, icon, iconButton } from './util.js';
import { THINKING_MODES, DEFAULTS } from './settings.js';
import { LANGUAGES, t as tr } from './i18n.js';

let openCount = 0;

export function openModal({ title, body, wide = false, className = '', onClose }) {
  const overlay = h('div', { class: 'overlay' });
  const dlg = h('div', { class: 'dialog ' + (wide ? 'wide ' : '') + className, role: 'dialog', 'aria-modal': 'true', 'aria-label': title || '' });
  const close = () => {
    if (!overlay.isConnected) return;
    overlay.remove();
    document.removeEventListener('keydown', onKey, true);
    openCount--;
    if (previous && previous.isConnected) previous.focus();
    if (onClose) onClose();
  };
  const onKey = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); close(); }
  };
  const header = h('div', { class: 'dialog-head' }, h('h2', { text: title || '' }), h('span', { class: 'grow' }),
    iconButton('x', tr('close'), close));
  dlg.append(header, body);
  overlay.append(dlg);
  overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });
  const previous = document.activeElement;
  document.body.append(overlay);
  document.addEventListener('keydown', onKey, true);
  openCount++;
  const first = dlg.querySelector('input, textarea, select, button.primary');
  if (first) first.focus();
  return { el: dlg, close };
}

export const hasOpenModal = () => openCount > 0;

export function confirmDialog(t, message, { okLabel, danger = false, title = '' } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v) => { if (!done) { done = true; resolve(v); } };
    const body = h('div', { class: 'dialog-body' }, h('p', { class: 'dialog-text', text: message }));
    const m = openModal({ title, body, className: 'small', onClose: () => finish(false) });
    const ok = h('button', { class: 'btn ' + (danger ? 'danger' : 'primary'), type: 'button', text: okLabel || t('ok'), onclick: () => { finish(true); m.close(); } });
    body.append(h('div', { class: 'dialog-buttons' },
      h('button', { class: 'btn', type: 'button', text: t('cancel'), onclick: () => m.close() }), ok));
    ok.focus();
  });
}

export function promptDialog(t, title, value = '') {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v) => { if (!done) { done = true; resolve(v); } };
    const input = h('input', { type: 'text', class: 'text-input', value });
    const body = h('div', { class: 'dialog-body' }, input);
    const m = openModal({ title, body, className: 'small', onClose: () => finish(null) });
    const submit = () => { finish(input.value.trim()); m.close(); };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); submit(); } });
    body.append(h('div', { class: 'dialog-buttons' },
      h('button', { class: 'btn', type: 'button', text: t('cancel'), onclick: () => m.close() }),
      h('button', { class: 'btn primary', type: 'button', text: t('save'), onclick: submit })));
    input.focus();
    input.select();
  });
}

// ---------------------------------------------------------------- popover ----

let activeMenu = null;

export function closeMenu() {
  if (activeMenu) { activeMenu.remove(); activeMenu = null; }
}

/** items: [{label, icon, active, danger, hint, onclick} | {separator:true}] */
export function menu(anchor, items, { align = 'start', up = false } = {}) {
  closeMenu();
  const el = h('div', { class: 'menu', role: 'menu' });
  for (const it of items) {
    if (it.separator) { el.append(h('div', { class: 'menu-sep' })); continue; }
    const b = h('button', { class: 'menu-item' + (it.danger ? ' danger' : '') + (it.active ? ' active' : ''), type: 'button', role: 'menuitem' });
    if (it.icon) b.append(icon(it.icon, 16));
    b.append(h('span', { class: 'menu-label' }, it.label));
    if (it.hint) b.append(h('span', { class: 'menu-hint', text: it.hint }));
    if (it.active) b.append(icon('check', 16));
    b.addEventListener('click', () => { closeMenu(); it.onclick(); });
    el.append(b);
  }
  document.body.append(el);
  const r = anchor.getBoundingClientRect();
  const mw = el.offsetWidth;
  const mh = el.offsetHeight;
  let left = align === 'end' ? r.right - mw : r.left;
  left = Math.max(8, Math.min(left, window.innerWidth - mw - 8));
  let top = up ? r.top - mh - 6 : r.bottom + 6;
  if (top + mh > window.innerHeight - 8) top = Math.max(8, r.top - mh - 6);
  if (top < 8) top = 8;
  el.style.left = left + 'px';
  el.style.top = top + 'px';
  activeMenu = el;
  const off = (e) => {
    if (e.type === 'keydown' && e.key !== 'Escape') return;
    if (e.type === 'mousedown' && el.contains(e.target)) return;
    closeMenu();
    document.removeEventListener('mousedown', off, true);
    document.removeEventListener('keydown', off, true);
    window.removeEventListener('resize', off);
  };
  setTimeout(() => {
    document.addEventListener('mousedown', off, true);
    document.addEventListener('keydown', off, true);
    window.addEventListener('resize', off);
  }, 0);
  const firstItem = el.querySelector('.menu-item');
  if (firstItem) firstItem.focus();
  el.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    const list = [...el.querySelectorAll('.menu-item')];
    const i = list.indexOf(document.activeElement);
    list[(i + (e.key === 'ArrowDown' ? 1 : list.length - 1)) % list.length].focus();
  });
  return el;
}

// ------------------------------------------------------------ viewers ----

export function openImageViewer(att) {
  const img = h('img', { class: 'lightbox-img', src: att.data, alt: att.name });
  openModal({ title: att.name, body: h('div', { class: 'dialog-body lightbox' }, img), wide: true });
}

export function openTextViewer(att) {
  openModal({ title: att.name, body: h('div', { class: 'dialog-body' }, h('pre', { class: 'text-view', text: att.data })), wide: true });
}

// HTML/SVG preview runs in a sandboxed frame served without the app's CSP.  The
// frame has an opaque origin (no allow-same-origin), so it cannot touch chats.
export function openPreview(t, code, lang) {
  const frame = h('iframe', { class: 'preview-frame', sandbox: 'allow-scripts allow-forms allow-modals allow-popups', title: t('preview') });
  const isSvg = (lang || '').toLowerCase() === 'svg';
  const payload = isSvg ? `<!doctype html><meta charset="utf-8"><style>html,body{margin:0;height:100%;display:grid;place-items:center;background:#fff}svg{max-width:100%;max-height:100%}</style>${code}` : code;
  frame.addEventListener('load', () => { frame.contentWindow.postMessage({ q36Preview: payload }, '*'); }, { once: true });
  frame.src = '/preview.html';
  const m = openModal({ title: t('preview'), body: h('div', { class: 'dialog-body preview' }, frame), wide: true, className: 'tall' });
  return m;
}

// ------------------------------------------------------------ shortcuts ----

export function openShortcuts(t) {
  const mod = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl';
  const rows = [
    [`${mod} + K`, t('scNewChat')],
    [`${mod} + Shift + O`, t('scNewChat')],
    [`${mod} + B`, t('scSidebar')],
    [`${mod} + /`, t('scFocusInput')],
    [`${mod} + ,`, t('scSettings')],
    [`${mod} + Shift + F`, t('scSearch')],
    ['Enter / Shift+Enter', t('scSend')],
    [`${mod} + Enter`, t('scSaveEdit')],
    ['Esc', t('scStop')],
    ['?', t('scHelp')],
  ];
  const table = h('table', { class: 'kbd-table' });
  for (const [k, d] of rows) table.append(h('tr', null, h('td', null, h('kbd', { text: k })), h('td', { text: d })));
  openModal({ title: t('shortcuts'), body: h('div', { class: 'dialog-body' }, table), className: 'small' });
}

// --------------------------------------------------------------- settings ----

function field(label, control, hint) {
  return h('label', { class: 'field' },
    h('span', { class: 'field-label', text: label }), control,
    hint ? h('span', { class: 'field-hint', text: hint }) : null);
}

function toggle(checked, onchange) {
  const input = h('input', { type: 'checkbox', class: 'switch' });
  input.checked = !!checked;
  input.addEventListener('change', () => onchange(input.checked));
  return input;
}

function select(options, value, onchange) {
  const s = h('select', { class: 'select' });
  for (const [v, label] of options) s.append(h('option', { value: v, text: label }));
  s.value = value;
  s.addEventListener('change', () => onchange(s.value));
  return s;
}

function numberInput(value, { min, max, step, placeholder, integer }, onchange) {
  const input = h('input', { type: 'number', class: 'text-input num', placeholder, min, max, step });
  input.value = value === null || value === undefined ? '' : value;
  input.addEventListener('change', () => {
    if (input.value.trim() === '') { onchange(null); return; }
    let v = Number(input.value);
    if (!Number.isFinite(v)) { onchange(null); input.value = ''; return; }
    if (integer) v = Math.round(v);
    if (min !== undefined) v = Math.max(min, v);
    if (max !== undefined) v = Math.min(max, v);
    input.value = v;
    onchange(v);
  });
  return input;
}

export function openSettings({ t, settings, info, save, actions }) {
  const tabs = ['general', 'chat', 'sampling', 'data', 'about'];
  const nav = h('div', { class: 'tabs', role: 'tablist' });
  const panel = h('div', { class: 'tab-panel' });
  const body = h('div', { class: 'dialog-body settings' }, nav, panel);
  let current = 'general';

  const set = (path, value) => {
    if (path.length === 2) settings[path[0]][path[1]] = value; else settings[path[0]] = value;
    save();
  };

  const builders = {
    general() {
      return [
        field(t('language'), select([['auto', t('auto')], ...LANGUAGES.map((l) => [l.code, l.name])], settings.lang, (v) => { set(['lang'], v); actions.relang(); })),
        field(t('theme'), select([['auto', t('auto')], ['light', t('light')], ['dark', t('dark')]], settings.theme, (v) => { set(['theme'], v); actions.retheme(); })),
        field(t('fontSize'), (() => {
          const r = h('input', { type: 'range', min: 12, max: 20, step: 1, class: 'range' });
          r.value = settings.fontSize;
          const val = h('span', { class: 'range-val', text: settings.fontSize + 'px' });
          r.addEventListener('input', () => { set(['fontSize'], +r.value); val.textContent = r.value + 'px'; actions.retheme(); });
          return h('span', { class: 'range-row' }, r, val);
        })()),
        h('div', { class: 'field row' }, h('span', { class: 'field-label', text: t('wideLayout') }), toggle(settings.wide, (v) => { set(['wide'], v); actions.retheme(); })),
        h('div', { class: 'field row' }, h('span', { class: 'field-label', text: t('enterToSend') }), toggle(settings.enterToSend, (v) => set(['enterToSend'], v))),
        h('div', { class: 'field row' }, h('span', { class: 'field-label', text: t('showStats') }), toggle(settings.showStats, (v) => { set(['showStats'], v); actions.rerender(); })),
      ];
    },
    chat() {
      const sys = h('textarea', { class: 'text-input', rows: 6, placeholder: t('systemPlaceholder') });
      sys.value = settings.system;
      sys.addEventListener('input', () => set(['system'], sys.value));
      return [
        field(t('defaultSystem'), sys, t('systemHint')),
        field(t('thinkingMode'), select(THINKING_MODES.map((m) => [m, t('think_' + m)]), settings.thinking, (v) => { set(['thinking'], v); actions.thinking(); }), t('thinkingHint')),
        field(t('thinkingDisplay'), select([['streaming', t('thinkOpenStreaming')], ['always', t('thinkOpenAlways')], ['never', t('thinkOpenNever')]], settings.thinkingOpen, (v) => { set(['thinkingOpen'], v); actions.rerender(); })),
        h('div', { class: 'field row' }, h('span', { class: 'field-label' }, t('sendReasoning'), h('span', { class: 'field-hint', text: t('sendReasoningHint') })),
          toggle(settings.sendReasoning, (v) => set(['sendReasoning'], v))),
        h('div', { class: 'field row' }, h('span', { class: 'field-label' }, t('autoTitle'), h('span', { class: 'field-hint', text: t('autoTitleHint') })),
          toggle(settings.autoTitle, (v) => set(['autoTitle'], v))),
      ];
    },
    sampling() {
      const g = settings.gen;
      const num = (key, opts) => field(t('gen_' + key), numberInput(g[key], { ...opts, placeholder: t('serverDefault') }, (v) => set(['gen', key], v)));
      const stop = h('textarea', { class: 'text-input', rows: 2, placeholder: t('stopPlaceholder') });
      stop.value = g.stop;
      stop.addEventListener('input', () => set(['gen', 'stop'], stop.value));
      const reset = h('button', { class: 'btn', type: 'button', text: t('resetDefaults'), onclick: () => { settings.gen = { ...DEFAULTS.gen }; save(); show('sampling'); } });
      return [
        h('p', { class: 'dialog-text muted', text: t('samplingHint') }),
        h('div', { class: 'grid2' },
          num('temperature', { min: 0, max: 5, step: 0.05 }),
          num('top_p', { min: 0, max: 1, step: 0.01 }),
          num('top_k', { min: 0, step: 1, integer: true }),
          num('min_p', { min: 0, max: 1, step: 0.01 }),
          num('presence_penalty', { min: -2, max: 2, step: 0.1 }),
          num('frequency_penalty', { min: -2, max: 2, step: 0.1 }),
          num('max_tokens', { min: 1, step: 1, integer: true }),
          num('seed', { min: 1, step: 1, integer: true })),
        field(t('gen_stop'), stop, t('stopHint')),
        reset,
      ];
    },
    data() {
      const file = h('input', { type: 'file', accept: '.json,application/json', hidden: true, multiple: true });
      file.addEventListener('change', async () => { await actions.importFiles([...file.files]); file.value = ''; });
      return [
        h('div', { class: 'data-actions' },
          h('button', { class: 'btn', type: 'button', onclick: () => actions.exportAll() }, icon('download', 16), t('exportAll')),
          h('button', { class: 'btn', type: 'button', onclick: () => file.click() }, icon('upload', 16), t('importChats')),
          file,
          h('button', { class: 'btn danger', type: 'button', onclick: () => actions.deleteAll() }, icon('trash', 16), t('deleteAll'))),
        h('p', { class: 'dialog-text muted', text: t('dataHint') }),
      ];
    },
    about() {
      const rows = [
        [t('model'), info.model || '–'],
        [t('contextSize'), info.n_ctx ? info.n_ctx.toLocaleString() : '–'],
        [t('vision'), info.vision ? t('enabled') : t('disabled')],
        [t('parallelSlots'), String(info.slots ?? 1)],
        [t('storage'), actions.persistent() ? t('storageLocal') : t('storageMemory')],
      ];
      const table = h('table', { class: 'kbd-table about' });
      for (const [k, v] of rows) table.append(h('tr', null, h('td', { text: k }), h('td', { text: v })));
      return [table, h('button', { class: 'btn', type: 'button', onclick: () => actions.shortcuts() }, icon('keyboard', 16), t('shortcuts'))];
    },
  };

  function show(name) {
    current = name;
    for (const b of nav.children) b.classList.toggle('active', b.dataset.tab === name);
    panel.replaceChildren(...builders[name]());
  }
  for (const name of tabs) {
    nav.append(h('button', { class: 'tab', type: 'button', role: 'tab', dataset: { tab: name }, text: t('tab_' + name), onclick: () => show(name) }));
  }
  show(current);
  return openModal({ title: t('settings'), body, wide: true, className: 'settings-dialog' });
}

