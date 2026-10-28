// Application controller: owns state, wires the store, API and views together.

import { h, icon, iconButton, toast, debounce, download, safeFilename } from './util.js';
import { t, setLang, resolveLang } from './i18n.js';
import { loadSettings, saveSettings } from './settings.js';
import { openStore } from './store.js';
import { fetchServerInfo, ping, streamChat, completeChat, buildMessages, buildParams, ApiError } from './api.js';
import {
  createConversation, newNode, addChild, pathTo, select, switchSibling, removeSubtree, leafFrom,
  searchText, deriveTitle, serialize, deserialize, toMarkdown,
} from './tree.js';
import { setMarkdownLabels } from './markdown.js';
import { createMessageList } from './messages.js';
import { createComposer } from './composer.js';
import { createSidebar } from './sidebar.js';
import {
  openSettings, openShortcuts, openPreview, openImageViewer, openTextViewer, confirmDialog, promptDialog,
  openModal, hasOpenModal, menu, closeMenu,
} from './dialogs.js';

const state = {
  store: null,
  settings: loadSettings(),
  info: { model: '', n_ctx: 0, vision: null, default_max_tokens: 0, slots: 1 },
  online: true,
  convs: new Map(), // id -> conversation record (shared object with `loaded` when open)
  loaded: new Map(), // id -> { conv, nodes } for the open chat and any chat still generating
  cur: null, // { conv, nodes } or null for the empty "new chat" screen
  gens: new Map(), // convId -> { node, abort, t0, tFirst }
  drafts: new Map(),
  stick: true,
};

const ui = {};
const SIDEBAR_KEY = 'q36-webui-sidebar';

// ------------------------------------------------------------ appearance ----

function applyAppearance() {
  const s = state.settings;
  const root = document.documentElement;
  const dark = s.theme === 'dark' || (s.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  root.dataset.theme = dark ? 'dark' : 'light';
  root.style.setProperty('--font-size', s.fontSize + 'px');
  root.classList.toggle('wide', !!s.wide);
}

function applyLang() {
  setLang(resolveLang(state.settings.lang));
  setMarkdownLabels({ copy: t('copy'), preview: t('preview'), wrap: t('wrap') });
  document.documentElement.lang = resolveLang(state.settings.lang);
}

const saveSettingsNow = () => { saveSettings(state.settings); };

matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyAppearance);

// ------------------------------------------------------------ persistence ----

let storageWarned = false;
async function persist(loaded, nodes = []) {
  if (!state.convs.has(loaded.conv.id)) return; // deleted while a reply was streaming
  try {
    await state.store.saveNodes(loaded.conv.id, nodes);
    await state.store.saveConversation(loaded.conv);
  } catch (e) {
    console.error(e);
    if (!storageWarned) { storageWarned = true; toast(t('storageError'), 'error', 5000); }
  }
}

function touch(loaded) {
  const { conv, nodes } = loaded;
  conv.updatedAt = Date.now();
  conv.text = searchText(nodes);
  if (!conv.title) {
    const first = pathTo(nodes, conv.currNode).find((n) => n.role === 'user');
    if (first) conv.title = deriveTitle(first.content) || (first.attachments[0] && first.attachments[0].name) || '';
  }
}

async function loadConversation(id) {
  const cached = state.loaded.get(id);
  if (cached) return cached;
  const data = await state.store.loadConversation(id);
  if (!data) return null;
  for (const n of data.nodes.values()) {
    if (n.incomplete) { delete n.incomplete; n.aborted = true; }
  }
  if (!data.nodes.has(data.conv.currNode)) data.conv.currNode = leafFrom(data.nodes, data.conv.root);
  state.convs.set(id, data.conv);
  state.loaded.set(id, data);
  return data;
}

function trimCache() {
  for (const id of [...state.loaded.keys()]) {
    if (state.cur && state.cur.conv.id === id) continue;
    if (state.gens.has(id)) continue;
    state.loaded.delete(id);
  }
}

// ------------------------------------------------------------------ views ----

const isCurrent = (convId) => !!state.cur && state.cur.conv.id === convId;
const currentGen = () => (state.cur ? state.gens.get(state.cur.conv.id) : null);

function refreshSidebar() {
  ui.sidebar.render([...state.convs.values()], state.cur ? state.cur.conv.id : null, new Set(state.gens.keys()));
}

function stickToBottom(force = false) {
  if (!force && !state.stick) return;
  ui.scroll.scrollTop = ui.scroll.scrollHeight;
}

function contextUsed() {
  if (!state.cur) return 0;
  const path = pathTo(state.cur.nodes, state.cur.conv.currNode);
  let used = 0;
  let after = 0;
  for (const n of path) {
    if (n.role === 'assistant' && n.stats && n.stats.prompt) {
      used = n.stats.prompt + (n.stats.completion || 0);
      after = 0;
    } else {
      after += (n.content.length + (n.reasoning || '').length) / 3.2;
      for (const a of n.attachments || []) after += a.kind === 'text' ? a.data.length / 3.2 : 700;
    }
  }
  if (!used) used = (state.cur.conv.system || '').length / 3.2;
  return Math.round(used + after);
}

function updateContext() {
  const used = contextUsed();
  ui.composer.setContext(state.cur && used ? used : 0, state.info.n_ctx);
}

function updateHeader() {
  const conv = state.cur && state.cur.conv;
  ui.title.textContent = conv ? (conv.title || t('untitled')) : t('newChat');
  ui.title.classList.toggle('static', !conv);
  document.title = (conv && conv.title ? conv.title + ' · ' : '') + 'QuarkStar';
  ui.modelName.textContent = state.info.model || '';
  ui.status.classList.toggle('offline', !state.online);
  ui.status.title = state.online ? t('serverOnline') : t('serverOffline');
  ui.banner.hidden = state.online;
}

function renderThread() {
  const loaded = state.cur;
  const gen = currentGen();
  const hasMessages = !!loaded && pathTo(loaded.nodes, loaded.conv.currNode).length > 0;
  ui.empty.hidden = hasMessages;
  ui.messages.hidden = !hasMessages;
  if (loaded) ui.list.render(loaded.conv, loaded.nodes, gen ? gen.node.id : null);
  else ui.messages.replaceChildren();
  ui.composer.setBusy(!!gen);
  updateHeader();
  updateContext();
  stickToBottom();
}

function renderEmpty() {
  const suggestions = ['sug1', 'sug2', 'sug3', 'sug4'].map((k) => {
    const b = h('button', { class: 'suggestion', type: 'button', text: t(k) });
    b.addEventListener('click', () => send({ text: t(k), attachments: [] }));
    return b;
  });
  ui.empty.replaceChildren(
    h('div', { class: 'logo-mark' }, icon('sparkle', 28)),
    h('h1', { text: t('welcome') }),
    h('p', { class: 'muted', text: state.info.model ? t('welcomeModel', { model: state.info.model }) : t('welcomeSub') }),
    h('div', { class: 'suggestions' }, suggestions));
}

// -------------------------------------------------------- conversation ops ----

function setRoute(id) {
  const want = id ? '#/c/' + id : '';
  if (location.hash === want || (!want && !location.hash)) return;
  history.pushState(null, '', want || location.pathname + location.search);
}

function routeId() {
  const m = /^#\/c\/([\w-]+)$/.exec(location.hash);
  return m ? m[1] : null;
}

function saveDraft() {
  if (!ui.composer) return;
  const d = ui.composer.getDraft();
  const key = state.cur ? state.cur.conv.id : '';
  if (d.text || d.attachments.length) state.drafts.set(key, d); else state.drafts.delete(key);
}

async function showConversation(id, { push = true } = {}) {
  saveDraft();
  ui.list.stopEditing();
  let loaded = null;
  if (id) {
    loaded = await loadConversation(id);
    if (!loaded) { toast(t('chatNotFound'), 'error'); id = null; }
  }
  state.cur = loaded;
  trimCache();
  if (push) setRoute(id);
  state.stick = true;
  const d = state.drafts.get(id || '') || { text: '', attachments: [] };
  ui.composer.setDraft(d.text, d.attachments);
  renderThread();
  stickToBottom(true);
  refreshSidebar();
  closeSidebarOnMobile();
  if (!matchMedia('(pointer: coarse)').matches) ui.composer.focus();
}

function newChat() {
  if (!state.cur && !ui.composer.getDraft().text) { ui.composer.focus(); return; }
  return showConversation(null);
}

async function ensureConversation() {
  if (state.cur) return state.cur;
  const loaded = createConversation(state.settings.system);
  state.cur = loaded;
  state.convs.set(loaded.conv.id, loaded.conv);
  state.loaded.set(loaded.conv.id, loaded);
  setRoute(loaded.conv.id);
  return loaded;
}

async function deleteConversation(id) {
  const conv = state.convs.get(id);
  if (!conv) return;
  const ok = await confirmDialog(t, t('confirmDeleteChat', { title: conv.title || t('untitled') }), { danger: true, okLabel: t('delete') });
  if (!ok) return;
  const gen = state.gens.get(id);
  state.convs.delete(id);
  if (gen) gen.abort.abort();
  await state.store.deleteConversation(id);
  state.loaded.delete(id);
  state.drafts.delete(id);
  if (state.cur && state.cur.conv.id === id) await showConversation(null);
  else refreshSidebar();
}

async function renameConversation(id) {
  const conv = state.convs.get(id);
  if (!conv) return;
  const name = await promptDialog(t, t('rename'), conv.title);
  if (name === null || !name) return;
  conv.title = name;
  await state.store.saveConversation(conv);
  refreshSidebar();
  updateHeader();
}

async function exportConversation(id, kind) {
  const loaded = await loadConversation(id);
  if (!loaded) return;
  const name = safeFilename(loaded.conv.title);
  if (kind === 'md') {
    const labels = { untitled: t('untitled'), user: t('roleUser'), assistant: t('roleAssistant'), thinking: t('thinking'), attachment: t('attachment') };
    download(name + '.md', toMarkdown(loaded.conv, loaded.nodes, labels), 'text/markdown');
  } else {
    download(name + '.json', JSON.stringify(serialize(loaded.conv, loaded.nodes)), 'application/json');
  }
  trimCache();
}

async function exportAll() {
  const chats = [];
  for (const id of state.convs.keys()) {
    const l = await loadConversation(id);
    if (l) chats.push(serialize(l.conv, l.nodes));
  }
  trimCache();
  download('q36-chats.json', JSON.stringify({ version: 1, chats }), 'application/json');
}

async function importFiles(files) {
  let count = 0;
  for (const file of files) {
    try {
      const data = JSON.parse(await file.text());
      for (const item of Array.isArray(data.chats) ? data.chats : [data]) {
        const { conv, nodes } = deserialize(item);
        await state.store.saveConversation(conv);
        await state.store.saveNodes(conv.id, [...nodes.values()]);
        state.convs.set(conv.id, conv);
        count++;
      }
    } catch (e) {
      toast(t('importFailed', { name: file.name }), 'error', 4000);
    }
  }
  if (count) toast(t('imported', { n: count }));
  refreshSidebar();
}

async function deleteAll() {
  const ok = await confirmDialog(t, t('confirmDeleteAll'), { danger: true, okLabel: t('deleteAll') });
  if (!ok) return;
  state.convs.clear();
  for (const gen of state.gens.values()) gen.abort.abort();
  await state.store.clear();
  state.loaded.clear();
  state.drafts.clear();
  await showConversation(null);
  toast(t('allDeleted'));
}

async function togglePin(id) {
  const conv = state.convs.get(id);
  if (!conv) return;
  conv.pinned = !conv.pinned;
  await state.store.saveConversation(conv);
  refreshSidebar();
}

async function editSystemPrompt() {
  const conv = state.cur && state.cur.conv;
  const area = h('textarea', { class: 'text-input', rows: 10, placeholder: t('systemPlaceholder') });
  area.value = conv ? conv.system : state.settings.system;
  const body = h('div', { class: 'dialog-body' },
    h('p', { class: 'dialog-text muted', text: conv ? t('systemChatHint') : t('systemNewHint') }), area);
  const m = openModal({ title: t('systemPrompt'), body });
  body.append(h('div', { class: 'dialog-buttons' },
    h('button', { class: 'btn', type: 'button', text: t('cancel'), onclick: () => m.close() }),
    h('button', {
      class: 'btn primary', type: 'button', text: t('save'),
      onclick: async () => {
        if (conv) { conv.system = area.value; await persist(state.cur); updateContext(); }
        else { state.settings.system = area.value; saveSettingsNow(); }
        m.close();
      },
    })));
}

// ------------------------------------------------------------- generation ----

function describeError(e) {
  if (e instanceof ApiError) {
    if (e.code === 'context_length_exceeded') {
      const n = e.info && e.info.n_prompt_tokens;
      const c = e.info && e.info.n_ctx;
      return n && c ? t('errContext', { prompt: n.toLocaleString(), ctx: c.toLocaleString() }) : e.message;
    }
    return e.message;
  }
  if (e instanceof TypeError) return t('errNetwork');
  return String(e.message || e);
}

function rafThrottle(fn) {
  let queued = false;
  return () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fn(); });
  };
}

async function generate(loaded, parentId, fresh = []) {
  const { conv, nodes } = loaded;
  if (state.gens.has(conv.id)) return;
  const node = addChild(nodes, parentId, newNode('assistant', '', parentId, { incomplete: true }));
  select(nodes, conv, node.id);
  const gen = { node, abort: new AbortController(), t0: performance.now(), tFirst: 0 };
  state.gens.set(conv.id, gen);
  touch(loaded);
  persist(loaded, [...fresh, nodes.get(parentId), node]);
  if (isCurrent(conv.id)) renderThread();
  refreshSidebar();

  const paint = rafThrottle(() => {
    if (!isCurrent(conv.id)) return;
    ui.list.update(node);
    stickToBottom();
  });
  const checkpoint = debounce(() => persist(loaded, [node]), 1500);

  const body = {
    ...(state.info.model ? { model: state.info.model } : {}),
    messages: buildMessages(pathTo(nodes, parentId), { system: conv.system, sendReasoning: state.settings.sendReasoning }),
    ...buildParams(state.settings),
  };

  try {
    const r = await streamChat({
      body,
      signal: gen.abort.signal,
      onFirstToken: () => { gen.tFirst = performance.now(); },
      onReasoning: (d) => {
        if (!node.reasoningStart) node.reasoningStart = Date.now();
        node.reasoning += d;
        paint();
        checkpoint();
      },
      onContent: (d) => {
        if (node.reasoningStart && !node.reasoningMs) node.reasoningMs = Date.now() - node.reasoningStart;
        node.content += d;
        paint();
        checkpoint();
      },
    });
    node.finish = r.finish;
    const u = r.usage;
    if (u) {
      const now = performance.now();
      const genMs = gen.tFirst ? now - gen.tFirst : 0;
      node.stats = {
        prompt: u.prompt_tokens || 0,
        completion: u.completion_tokens || 0,
        cached: (u.prompt_tokens_details && u.prompt_tokens_details.cached_tokens) || 0,
        ttft: gen.tFirst ? gen.tFirst - gen.t0 : 0,
        total: now - gen.t0,
        tps: genMs > 200 && u.completion_tokens > 1 ? (u.completion_tokens - 1) / (genMs / 1000) : 0,
      };
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      node.aborted = true;
    } else {
      node.error = describeError(e);
      if (!(e instanceof ApiError)) checkHealth();
    }
  } finally {
    if (node.reasoningStart && !node.reasoningMs) node.reasoningMs = Date.now() - node.reasoningStart;
    delete node.reasoningStart;
    delete node.incomplete;
    state.gens.delete(conv.id);
    touch(loaded);
    await persist(loaded, [node]);
    if (isCurrent(conv.id)) {
      renderThread();
      if (!ui.list.isEditing()) ui.composer.focus();
    } else {
      trimCache();
    }
    refreshSidebar();
    if (state.settings.autoTitle && !node.error && !node.aborted && !conv.autoTitled) autoTitle(loaded);
  }
}

async function autoTitle(loaded) {
  const { conv, nodes } = loaded;
  conv.autoTitled = true;
  const path = pathTo(nodes, conv.currNode);
  if (path.length < 2) return;
  try {
    const text = await completeChat({
      body: {
        ...(state.info.model ? { model: state.info.model } : {}),
        messages: [
          { role: 'system', content: 'Reply with a title of at most 6 words for this conversation. No quotes, no punctuation at the end.' },
          { role: 'user', content: path[0].content.slice(0, 600) + '\n\n' + path[1].content.slice(0, 600) },
        ],
        max_tokens: 24,
        chat_template_kwargs: { enable_thinking: false },
      },
    });
    const title = text.replace(/^["'\s]+|["'.\s]+$/g, '').split('\n')[0].slice(0, 80);
    if (title) { conv.title = title; await persist(loaded); refreshSidebar(); updateHeader(); }
  } catch { /* the derived title stays */ }
}

async function send({ text, attachments }) {
  if (currentGen()) return;
  const loaded = await ensureConversation();
  state.drafts.delete(loaded.conv.id);
  state.drafts.delete('');
  const { conv, nodes } = loaded;
  const user = addChild(nodes, conv.currNode, newNode('user', text, conv.currNode, { attachments }));
  select(nodes, conv, user.id);
  state.stick = true;
  await generate(loaded, user.id, [nodes.get(user.parent)]);
}

function stop() {
  const gen = currentGen();
  if (gen) gen.abort.abort();
}

function busyToast() { toast(t('busyHint'), 'info'); }

async function handleAction(name, id, extra) {
  const loaded = state.cur;
  if (!loaded) return;
  const { conv, nodes } = loaded;
  const node = nodes.get(id);
  if (!node) return;
  const busy = state.gens.has(conv.id);
  switch (name) {
    case 'branch':
      if (switchSibling(nodes, conv, id, extra)) {
        await persist(loaded, []);
        renderThread();
      }
      break;
    case 'startEdit':
    case 'cancelEdit':
      renderThread();
      break;
    case 'saveEdit': {
      if (busy) { busyToast(); break; }
      ui.list.stopEditing();
      const text = (extra || '').trim();
      if (node.role === 'user') {
        if (!text && !node.attachments.length) { renderThread(); break; }
        if (text === node.content) { renderThread(); break; }
        const sib = addChild(nodes, node.parent, newNode('user', text, node.parent, { attachments: node.attachments }));
        select(nodes, conv, sib.id);
        state.stick = true;
        await generate(loaded, sib.id, [nodes.get(sib.parent)]);
      } else {
        node.content = extra || '';
        touch(loaded);
        await persist(loaded, [node]);
        renderThread();
      }
      break;
    }
    case 'regenerate':
      if (busy) { busyToast(); break; }
      state.stick = true;
      await generate(loaded, node.parent);
      break;
    case 'delete': {
      if (busy) { busyToast(); break; }
      const ok = await confirmDialog(t, node.role === 'user' ? t('confirmDeleteTurn') : t('confirmDeleteReply'), { danger: true, okLabel: t('delete') });
      if (!ok) break;
      const parent = nodes.get(node.parent);
      const removed = removeSubtree(nodes, conv, id);
      await state.store.deleteNodes(removed);
      touch(loaded);
      await persist(loaded, [parent]);
      renderThread();
      refreshSidebar();
      break;
    }
    default:
  }
}

// ------------------------------------------------------------------ health ----

async function checkHealth() {
  const ok = await ping();
  const changed = ok !== state.online;
  state.online = ok;
  if (ok && (changed || !state.info.n_ctx)) {
    try {
      state.info = await fetchServerInfo();
      if (!state.cur) renderEmpty();
      updateContext();
    } catch { /* keep old info */ }
  }
  if (ui.status) updateHeader();
}

// ----------------------------------------------------------------- sidebar ----

const narrow = () => matchMedia('(max-width: 820px)').matches;

function toggleSidebar() {
  if (narrow()) {
    document.body.classList.toggle('sidebar-open');
  } else {
    const collapsed = document.body.classList.toggle('sidebar-collapsed');
    try { localStorage.setItem(SIDEBAR_KEY, collapsed ? '0' : '1'); } catch { /* ignore */ }
  }
}

function closeSidebarOnMobile() { document.body.classList.remove('sidebar-open'); }

// ------------------------------------------------------------------- mount ----

function cycleTheme() {
  const dark = document.documentElement.dataset.theme === 'dark';
  state.settings.theme = dark ? 'light' : 'dark';
  saveSettingsNow();
  applyAppearance();
}

function settingsActions() {
  let modal = null;
  const actions = {
    relang() {
      modal.close();
      saveDraft();
      applyLang();
      mount();
      const d = state.drafts.get(state.cur ? state.cur.conv.id : '');
      if (d) ui.composer.setDraft(d.text, d.attachments);
      showSettings();
    },
    retheme: applyAppearance,
    rerender: () => renderThread(),
    thinking: () => ui.composer.refresh(),
    exportAll,
    importFiles,
    deleteAll,
    shortcuts: () => openShortcuts(t),
    persistent: () => !!state.store.persistent,
  };
  actions.setModal = (m) => { modal = m; };
  return actions;
}

function showSettings() {
  const actions = settingsActions();
  const m = openSettings({ t, settings: state.settings, info: state.info, save: saveSettingsNow, actions });
  actions.setModal(m);
}

function openHeaderMenu(anchor) {
  const conv = state.cur && state.cur.conv;
  const items = [{ label: t('systemPrompt'), icon: 'sparkle', onclick: editSystemPrompt }];
  if (conv) {
    items.push(
      { label: t('rename'), icon: 'edit', onclick: () => renameConversation(conv.id) },
      { label: conv.pinned ? t('unpin') : t('pin'), icon: 'pin', onclick: () => togglePin(conv.id) },
      { label: t('exportMarkdown'), icon: 'download', onclick: () => exportConversation(conv.id, 'md') },
      { label: t('exportJson'), icon: 'download', onclick: () => exportConversation(conv.id, 'json') },
      { separator: true },
      { label: t('deleteChat'), icon: 'trash', danger: true, onclick: () => deleteConversation(conv.id) });
  }
  menu(anchor, items, { align: 'end' });
}

function mount() {
  const app = document.getElementById('app');
  app.replaceChildren();

  const sidebarEl = h('aside', { class: 'sidebar', id: 'sidebar' });
  const backdrop = h('div', { class: 'backdrop', onclick: () => document.body.classList.remove('sidebar-open') });

  ui.title = h('button', { class: 'conv-title', type: 'button', onclick: () => { if (state.cur) renameConversation(state.cur.conv.id); } });
  ui.status = h('span', { class: 'status-dot' });
  ui.modelName = h('span', { class: 'model-name' });
  ui.banner = h('div', { class: 'banner', role: 'alert', hidden: true }, icon('alert', 16), h('span', { text: t('serverOffline') }));
  const moreBtn = iconButton('more', t('more'), (e) => openHeaderMenu(e.currentTarget));
  const topbar = h('header', { class: 'topbar' },
    iconButton('sidebar', t('toggleSidebar'), toggleSidebar, 'sidebar-toggle'),
    ui.title, h('span', { class: 'grow' }),
    h('span', { class: 'model-badge' }, ui.status, ui.modelName),
    iconButton('sparkle', t('systemPrompt'), editSystemPrompt),
    moreBtn);

  ui.messages = h('div', { class: 'messages' });
  ui.empty = h('div', { class: 'empty' });
  ui.scroll = h('div', { class: 'scroll', tabindex: -1 }, h('div', { class: 'thread' }, ui.empty, ui.messages));
  ui.toBottom = h('button', { class: 'to-bottom', type: 'button', title: t('scrollToBottom'), 'aria-label': t('scrollToBottom'), hidden: true,
    onclick: () => { state.stick = true; stickToBottom(true); } }, icon('arrowDown', 18));
  const composerRoot = h('div', { class: 'composer' });
  const composerWrap = h('footer', { class: 'composer-wrap' }, composerRoot);
  const main = h('main', { class: 'main' }, topbar, ui.banner, h('div', { class: 'scroll-wrap' }, ui.scroll, ui.toBottom), composerWrap);
  app.append(sidebarEl, backdrop, main);

  ui.scroll.addEventListener('scroll', () => {
    const gap = ui.scroll.scrollHeight - ui.scroll.scrollTop - ui.scroll.clientHeight;
    state.stick = gap < 60;
    ui.toBottom.hidden = state.stick;
  });

  ui.sidebar = createSidebar(sidebarEl, {
    t,
    onNew: newChat,
    onSelect: (id) => showConversation(id),
    onSearch: refreshSidebar,
    onSettings: showSettings,
    onShortcuts: () => openShortcuts(t),
    onTheme: cycleTheme,
    onAction: (name, id) => {
      if (name === 'rename') renameConversation(id);
      else if (name === 'pin') togglePin(id);
      else if (name === 'delete') deleteConversation(id);
      else if (name === 'exportMd') exportConversation(id, 'md');
      else if (name === 'exportJson') exportConversation(id, 'json');
    },
  });

  ui.list = createMessageList(ui.messages, {
    t,
    settings: () => state.settings,
    action: handleAction,
    preview: (code, lang) => openPreview(t, code, lang),
    openImage: openImageViewer,
    openText: openTextViewer,
  });

  ui.composer = createComposer(composerRoot, {
    t,
    settings: () => state.settings,
    vision: () => state.info.vision !== false,
    onSubmit: send,
    onStop: stop,
    onThinking: (m) => { state.settings.thinking = m; saveSettingsNow(); ui.composer.refresh(); },
    onDraft: () => {},
  });

  renderEmpty();
  renderThread();
  refreshSidebar();
}

// ---------------------------------------------------------------- shortcuts ----

function installShortcuts() {
  document.addEventListener('keydown', (e) => {
    const mod = e.ctrlKey || e.metaKey;
    const key = e.key.toLowerCase();
    if (mod && key === 'k' && !e.shiftKey) { e.preventDefault(); newChat(); return; }
    if (mod && e.shiftKey && key === 'o') { e.preventDefault(); newChat(); return; }
    if (mod && key === 'b') { e.preventDefault(); toggleSidebar(); return; }
    if (mod && key === '/') { e.preventDefault(); ui.composer.focus(); return; }
    if (mod && key === ',') { e.preventDefault(); showSettings(); return; }
    if (mod && e.shiftKey && key === 'f') {
      e.preventDefault();
      document.body.classList.add('sidebar-open');
      document.body.classList.remove('sidebar-collapsed');
      ui.sidebar.focusSearch();
      return;
    }
    if (e.key === 'Escape' && !hasOpenModal()) {
      closeMenu();
      if (currentGen() && !ui.list.isEditing()) { e.preventDefault(); stop(); }
      return;
    }
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement && document.activeElement.tagName);
    if (e.key === '?' && !typing && !mod && !hasOpenModal()) { e.preventDefault(); openShortcuts(t); }
  });
}

// -------------------------------------------------------------------- boot ----

async function boot() {
  applyAppearance();
  applyLang();
  try { if (localStorage.getItem(SIDEBAR_KEY) === '0') document.body.classList.add('sidebar-collapsed'); } catch { /* ignore */ }
  state.store = await openStore();
  for (const c of await state.store.listConversations()) state.convs.set(c.id, c);
  mount();
  installShortcuts();
  window.addEventListener('beforeunload', (e) => {
    if (state.gens.size) { e.preventDefault(); e.returnValue = ''; }
  });
  window.addEventListener('popstate', () => {
    const id = routeId();
    if ((state.cur ? state.cur.conv.id : null) !== id) showConversation(id, { push: false });
  });
  const id = routeId();
  if (id) await showConversation(id, { push: false });
  await checkHealth();
  setInterval(checkHealth, 10000);
  setInterval(() => { if (state.gens.size) ui.list.tick(); }, 500);
  if (!state.store.persistent) toast(t('storageMemoryWarn'), 'error', 6000);
  window.__q36 = { state };
}

boot().catch((e) => {
  console.error(e);
  document.getElementById('app').textContent = 'Failed to start: ' + e.message;
});
