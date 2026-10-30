// Message composer: auto-growing input, attachments (picker, paste, drag and
// drop), thinking-mode picker, context meter and the send/stop button.

import { h, icon, toast, fmtNumber } from './util.js';
import { processFile } from './attachments.js';
import { menu } from './dialogs.js';
import { THINKING_MODES } from './settings.js';

const MAX_ATTACHMENTS = 12;

export function createComposer(root, ctx) {
  const { t } = ctx;
  let attachments = [];
  let busy = false;
  let dragDepth = 0;

  const input = h('textarea', { class: 'composer-input', rows: 1, placeholder: t('placeholder'), 'aria-label': t('placeholder'), enterkeyhint: 'send' });
  const files = h('input', { type: 'file', multiple: true, hidden: true });
  const attRow = h('div', { class: 'att-row' });
  const attachBtn = h('button', { class: 'icon-btn', type: 'button', title: t('attach'), 'aria-label': t('attach'), onclick: () => files.click() }, icon('clip'));
  const thinkBtn = h('button', { class: 'chip-btn', type: 'button', title: t('thinkingMode') });
  const meter = h('span', { class: 'ctx-meter', hidden: true });
  const meterBar = h('span', { class: 'ctx-bar' }, h('span'));
  const meterText = h('span', { class: 'ctx-text' });
  meter.append(meterBar, meterText);
  const sendBtn = h('button', { class: 'send-btn', type: 'button' });
  const box = h('div', { class: 'composer-box' }, attRow, input,
    h('div', { class: 'composer-bar' }, attachBtn, thinkBtn, h('span', { class: 'grow' }), meter, sendBtn));
  const hint = h('div', { class: 'composer-hint' });
  root.replaceChildren(box, hint, files);

  function fit() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 260) + 'px';
  }

  function refresh() {
    const can = !!input.value.trim() || attachments.length > 0;
    sendBtn.replaceChildren(icon(busy ? 'stop' : 'send', 18));
    sendBtn.classList.toggle('stop', busy);
    sendBtn.disabled = !busy && !can;
    sendBtn.title = busy ? t('stop') : t('send');
    sendBtn.setAttribute('aria-label', sendBtn.title);
    attachBtn.hidden = false;
    hint.textContent = ctx.settings().enterToSend ? t('hintEnter') : t('hintCtrlEnter');
    const mode = ctx.settings().thinking;
    thinkBtn.replaceChildren(icon('brain', 15), h('span', { text: t('think_' + mode) }), icon('chevronDown', 14));
    thinkBtn.classList.toggle('active', mode !== 'default');
  }

  function renderAttachments() {
    attRow.replaceChildren();
    attRow.hidden = attachments.length === 0;
    for (const a of attachments) {
      const remove = h('button', { class: 'att-remove', type: 'button', title: t('remove'), 'aria-label': t('remove'), onclick: () => { attachments = attachments.filter((x) => x !== a); renderAttachments(); refresh(); } }, icon('x', 12));
      if (a.kind === 'image') {
        attRow.append(h('div', { class: 'att-card image' }, h('img', { src: a.data, alt: a.name }), remove));
      } else {
        attRow.append(h('div', { class: 'att-card file', title: a.name }, icon('file', 16), h('span', { class: 'att-name', text: a.name }), remove));
      }
    }
  }

  async function addFiles(list) {
    for (const file of list) {
      if (attachments.length >= MAX_ATTACHMENTS) { toast(t('tooManyFiles'), 'error'); break; }
      const r = await processFile(file, { vision: ctx.vision() });
      if (r.error) {
        toast(t('attachError_' + r.error, { name: file.name }), 'error', 4000);
        continue;
      }
      attachments.push(r.attachment);
    }
    renderAttachments();
    refresh();
    input.focus();
  }

  function submit() {
    const text = input.value.trim();
    if (!text && !attachments.length) return;
    const payload = { text, attachments };
    input.value = '';
    attachments = [];
    renderAttachments();
    fit();
    refresh();
    ctx.onSubmit(payload);
  }

  input.addEventListener('input', () => { fit(); refresh(); ctx.onDraft && ctx.onDraft(input.value); });
  input.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.isComposing || e.keyCode === 229) return;
    const modifier = e.ctrlKey || e.metaKey;
    const send = ctx.settings().enterToSend ? !e.shiftKey && !e.altKey : modifier;
    if (!send) return;
    e.preventDefault();
    if (!busy) submit();
  });
  input.addEventListener('paste', (e) => {
    const items = [...(e.clipboardData?.files || [])];
    if (items.length) { e.preventDefault(); addFiles(items); }
  });
  sendBtn.addEventListener('click', () => { if (busy) ctx.onStop(); else submit(); });
  files.addEventListener('change', () => { addFiles([...files.files]); files.value = ''; });
  thinkBtn.addEventListener('click', () => {
    menu(thinkBtn, THINKING_MODES.map((m) => ({
      label: t('think_' + m), hint: t('thinkHint_' + m), active: ctx.settings().thinking === m,
      onclick: () => ctx.onThinking(m),
    })), { up: true });
  });

  // Drop zone covers the whole window.
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes('Files');
  window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { dragDepth++; document.body.classList.add('dragging'); } });
  window.addEventListener('dragleave', (e) => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove('dragging');
    addFiles([...e.dataTransfer.files]);
  });

  fit();
  refresh();

  return {
    focus: () => input.focus(),
    setBusy(b) { busy = b; refresh(); },
    refresh,
    getDraft: () => ({ text: input.value, attachments }),
    setDraft(text, atts = []) { input.value = text || ''; attachments = atts; renderAttachments(); fit(); refresh(); },
    insert(text) { input.value = text; fit(); refresh(); input.focus(); },
    setContext(used, total) {
      if (!total) { meter.hidden = true; return; }
      meter.hidden = false;
      const frac = Math.min(1, used / total);
      meterBar.firstChild.style.width = (frac * 100).toFixed(1) + '%';
      meter.classList.toggle('warn', frac > 0.8);
      meter.classList.toggle('full', frac > 0.95);
      meterText.textContent = `${fmtNumber(used)} / ${fmtNumber(total)}`;
      meter.title = t('contextUsage', { pct: Math.round(frac * 100) });
    },
  };
}
