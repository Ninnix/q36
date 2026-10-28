// Message list view.  Renders the visible branch of a conversation and updates
// the streaming reply in place so long answers do not re-layout everything.

import { h, icon, iconButton, copyText, toast, fmtDuration, fmtNumber, fmtBytes } from './util.js';
import { renderBlocks } from './markdown.js';
import { pathTo, siblingsOf } from './tree.js';

const LONG_USER_MESSAGE = 1600;

// Replace only the blocks whose HTML changed; earlier blocks of a streaming
// reply are final, so usually just the last one is touched.
function patchBlocks(container, state, text) {
  const blocks = renderBlocks(text);
  const prev = state.html;
  for (let i = 0; i < blocks.length; i++) {
    if (prev[i] === blocks[i]) continue;
    let el = state.els[i];
    if (!el) {
      el = h('div', { class: 'blk' });
      container.append(el);
      state.els[i] = el;
    }
    el.innerHTML = blocks[i];
  }
  for (let i = blocks.length; i < state.els.length; i++) state.els[i].remove();
  state.els.length = blocks.length;
  state.html = blocks;
}

export function createMessageList(root, ctx) {
  const { t } = ctx;
  const entries = new Map();
  let editing = null;
  let lastNodes = null;
  let lastConv = null;
  let streamingId = null;

  function action(name, id, extra) { ctx.action(name, id, extra); }

  function branchNav(nodes, node) {
    const sibs = siblingsOf(nodes, node.id);
    if (sibs.length < 2) return null;
    const idx = sibs.indexOf(node.id);
    const prev = iconButton('chevronLeft', t('prevBranch'), () => action('branch', node.id, -1), 'tiny');
    const next = iconButton('chevronRight', t('nextBranch'), () => action('branch', node.id, 1), 'tiny');
    prev.disabled = idx === 0;
    next.disabled = idx === sibs.length - 1;
    return h('span', { class: 'branch-nav' }, prev, h('span', { class: 'branch-count', text: `${idx + 1} / ${sibs.length}` }), next);
  }

  function attachmentsView(node) {
    if (!node.attachments || !node.attachments.length) return null;
    const box = h('div', { class: 'attachments' });
    for (const a of node.attachments) {
      if (a.kind === 'image') {
        const img = h('img', { class: 'att-img', src: a.data, alt: a.name, title: a.name });
        img.addEventListener('click', () => ctx.openImage(a));
        box.append(img);
      } else {
        const chip = h('button', { class: 'att-chip', type: 'button', title: t('viewAttachment') }, icon('file', 16),
          h('span', { class: 'att-name', text: a.name }), h('span', { class: 'att-size', text: fmtBytes(a.data.length) }));
        chip.addEventListener('click', () => ctx.openText(a));
        box.append(chip);
      }
    }
    return box;
  }

  function actionsRow(nodes, node, items) {
    const row = h('div', { class: 'msg-actions' });
    for (const it of items) row.append(it);
    const nav = branchNav(nodes, node);
    if (nav) row.prepend(nav);
    return row;
  }

  function copyButton(node) {
    const b = iconButton('copy', t('copy'), async () => {
      const ok = await copyText(node.content || '');
      toast(ok ? t('copied') : t('copyFailed'), ok ? 'info' : 'error');
    });
    return b;
  }

  // ------------------------------------------------------------- user ------

  function buildUser(nodes, node) {
    const el = h('div', { class: 'msg user', dataset: { id: node.id } });
    if (editing === node.id) {
      const ta = h('textarea', { class: 'edit-area', rows: 3 });
      ta.value = node.content;
      const fit = () => { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 420) + 'px'; };
      ta.addEventListener('input', fit);
      ta.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') { editing = null; action('cancelEdit', node.id); }
        else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); action('saveEdit', node.id, ta.value); }
      });
      el.append(h('div', { class: 'edit-box' }, ta,
        h('div', { class: 'edit-buttons' },
          h('button', { class: 'btn', type: 'button', text: t('cancel'), onclick: () => { editing = null; action('cancelEdit', node.id); } }),
          h('button', { class: 'btn primary', type: 'button', text: t('saveSubmit'), onclick: () => action('saveEdit', node.id, ta.value) }))));
      setTimeout(() => { fit(); ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }, 0);
      return el;
    }
    const att = attachmentsView(node);
    if (att) el.append(att);
    if (node.content) {
      const long = node.content.length > LONG_USER_MESSAGE;
      const bubble = h('div', { class: 'bubble' + (long ? ' clamped' : ''), text: node.content });
      el.append(bubble);
      if (long) {
        const more = h('button', { class: 'link-btn', type: 'button', text: t('showMore') });
        more.addEventListener('click', () => {
          const c = bubble.classList.toggle('clamped');
          more.textContent = c ? t('showMore') : t('showLess');
        });
        el.append(more);
      }
    }
    el.append(actionsRow(nodes, node, [
      copyButton(node),
      iconButton('edit', t('edit'), () => { editing = node.id; action('startEdit', node.id); }),
      iconButton('trash', t('delete'), () => action('delete', node.id)),
    ]));
    el.title = new Date(node.ts).toLocaleString();
    return el;
  }

  // -------------------------------------------------------- assistant ------

  function buildAssistant(nodes, node) {
    const el = h('div', { class: 'msg assistant', dataset: { id: node.id } });
    const entry = {
      el, node, userToggled: false, open: false,
      contentState: { els: [], html: [] }, thinkState: { els: [], html: [] },
    };
    entry.think = h('div', { class: 'think', hidden: true });
    entry.thinkHead = h('button', { class: 'think-head', type: 'button', 'aria-expanded': 'false' });
    entry.thinkHead.addEventListener('click', () => {
      entry.userToggled = true;
      setThinkOpen(entry, !entry.open);
    });
    entry.thinkLabel = h('span', { class: 'think-label' });
    entry.thinkChevron = icon('chevronRight', 16);
    entry.thinkHead.append(icon('brain', 16), entry.thinkLabel, entry.thinkChevron);
    entry.thinkBody = h('div', { class: 'think-body md' });
    entry.think.append(entry.thinkHead, entry.thinkBody);
    entry.content = h('div', { class: 'md content' });
    entry.typing = h('div', { class: 'typing', hidden: true }, h('span'), h('span'), h('span'));
    entry.notice = h('div', { class: 'msg-notice', hidden: true });
    entry.error = h('div', { class: 'msg-error', hidden: true });
    entry.foot = h('div', { class: 'msg-foot' });
    el.append(entry.think, entry.content, entry.typing, entry.notice, entry.error, entry.foot);
    entries.set(node.id, entry);
    return entry;
  }

  function setThinkOpen(entry, open) {
    entry.open = open;
    entry.think.classList.toggle('open', open);
    entry.thinkHead.setAttribute('aria-expanded', String(open));
    if (open) patchBlocks(entry.thinkBody, entry.thinkState, entry.node.reasoning || '');
  }

  function thinkLabel(entry, streaming) {
    const n = entry.node;
    if (streaming && !n.content) {
      const ms = n.reasoningStart ? Date.now() - n.reasoningStart : 0;
      return t('thinking') + (ms > 1000 ? ' ' + fmtDuration(ms) : '…');
    }
    return n.reasoningMs ? t('thoughtFor', { time: fmtDuration(n.reasoningMs) }) : t('thought');
  }

  function statsText(node) {
    const s = node.stats;
    if (!s) return '';
    const parts = [];
    if (s.tps) parts.push(`${s.tps.toFixed(1)} ${t('tokPerSec')}`);
    if (s.completion) parts.push(`${fmtNumber(s.completion)} ${t('tokens')}`);
    if (s.prompt) parts.push(`${t('prompt')} ${fmtNumber(s.prompt)}${s.cached ? ` (${fmtNumber(s.cached)} ${t('cached')})` : ''}`);
    if (s.ttft) parts.push(`${t('firstToken')} ${fmtDuration(s.ttft)}`);
    if (s.total) parts.push(fmtDuration(s.total));
    return parts.join(' · ');
  }

  function fillAssistant(entry, nodes, node, streaming) {
    entry.node = node;
    const settings = ctx.settings();
    entry.el.classList.toggle('streaming', streaming);

    const hasReasoning = !!node.reasoning;
    entry.think.hidden = !hasReasoning;
    if (hasReasoning) {
      entry.thinkLabel.textContent = thinkLabel(entry, streaming);
      if (!entry.userToggled) {
        const mode = settings.thinkingOpen;
        const want = mode === 'always' || (mode === 'streaming' && streaming && !node.content);
        if (want !== entry.open) setThinkOpen(entry, want);
      }
      if (entry.open) patchBlocks(entry.thinkBody, entry.thinkState, node.reasoning);
    }

    patchBlocks(entry.content, entry.contentState, node.content || '');
    entry.typing.hidden = !(streaming && !node.content && !node.reasoning);

    const length = node.finish === 'length';
    entry.notice.hidden = !(length || node.aborted);
    entry.notice.textContent = node.aborted ? t('stopped') : length ? t('hitLimit') : '';
    entry.error.hidden = !node.error;
    entry.error.textContent = node.error || '';

    if (!streaming) {
      entry.foot.replaceChildren();
      if (editing !== node.id) {
        const items = [copyButton(node)];
        if (node.content || node.reasoning) items.push(iconButton('edit', t('edit'), () => { editing = node.id; action('startEdit', node.id); }));
        items.push(iconButton('refresh', t('regenerate'), () => action('regenerate', node.id)));
        items.push(iconButton('trash', t('delete'), () => action('delete', node.id)));
        entry.foot.append(actionsRow(nodes, node, items));
        if (ctx.settings().showStats) {
          const st = statsText(node);
          if (st) entry.foot.append(h('div', { class: 'msg-stats', text: st }));
        }
      }
    }
  }

  function buildAssistantEditor(entry, node) {
    const ta = h('textarea', { class: 'edit-area', rows: 6 });
    ta.value = node.content;
    const fit = () => { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 520) + 'px'; };
    ta.addEventListener('input', fit);
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { editing = null; action('cancelEdit', node.id); }
    });
    entry.content.replaceChildren(h('div', { class: 'edit-box' }, ta,
      h('div', { class: 'edit-buttons' },
        h('button', { class: 'btn', type: 'button', text: t('cancel'), onclick: () => { editing = null; action('cancelEdit', node.id); } }),
        h('button', { class: 'btn primary', type: 'button', text: t('save'), onclick: () => action('saveEdit', node.id, ta.value) }))));
    setTimeout(() => { fit(); ta.focus(); }, 0);
  }

  // ----------------------------------------------------------- public ------

  function render(conv, nodes, streamNode = null) {
    lastConv = conv;
    lastNodes = nodes;
    streamingId = streamNode;
    entries.clear();
    root.replaceChildren();
    for (const node of pathTo(nodes, conv.currNode)) {
      if (node.role === 'user') {
        root.append(buildUser(nodes, node));
      } else if (node.role === 'assistant') {
        const entry = buildAssistant(nodes, node);
        const streaming = node.id === streamNode;
        if (editing === node.id) {
          entry.el.classList.add('editing');
          buildAssistantEditor(entry, node);
          entry.foot.replaceChildren();
        } else {
          fillAssistant(entry, nodes, node, streaming);
        }
        root.append(entry.el);
      }
    }
  }

  function update(node) {
    const entry = entries.get(node.id);
    if (entry) fillAssistant(entry, lastNodes, node, node.id === streamingId);
  }

  // Live timers for the "Thinking 12s" label.
  function tick() {
    const entry = streamingId && entries.get(streamingId);
    if (entry && entry.node.reasoning && !entry.node.content) entry.thinkLabel.textContent = thinkLabel(entry, true);
  }

  root.addEventListener('click', async (e) => {
    const btn = e.target.closest('.code-btn');
    if (!btn) return;
    const block = btn.closest('.codeblock');
    const code = block.querySelector('pre code');
    const act = btn.dataset.action;
    if (act === 'copy') {
      const ok = await copyText(code.textContent);
      btn.textContent = ok ? t('copied') : t('copyFailed');
      setTimeout(() => { btn.textContent = t('copy'); }, 1400);
    } else if (act === 'wrap') {
      block.classList.toggle('wrap');
    } else if (act === 'preview') {
      ctx.preview(code.textContent, block.dataset.lang);
    }
  });

  return {
    render, update, tick,
    isEditing: () => editing,
    stopEditing: () => { editing = null; },
    get conv() { return lastConv; },
  };
}
