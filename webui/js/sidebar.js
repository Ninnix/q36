// Sidebar: new chat, search, and the grouped conversation list.

import { h, icon, iconButton, debounce } from './util.js';
import { menu } from './dialogs.js';

function groupOf(conv, now) {
  if (conv.pinned) return 'pinned';
  const d = new Date(conv.updatedAt);
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  const day = 86400000;
  if (d >= start) return 'today';
  if (d >= start - day) return 'yesterday';
  if (d >= start - 7 * day) return 'week';
  if (d >= start - 30 * day) return 'month';
  return 'older';
}

const ORDER = ['pinned', 'today', 'yesterday', 'week', 'month', 'older'];

export function createSidebar(root, ctx) {
  const { t } = ctx;
  let query = '';

  const search = h('input', { type: 'search', class: 'search-input', placeholder: t('searchChats'), 'aria-label': t('searchChats') });
  const list = h('nav', { class: 'chat-list', 'aria-label': t('chats') });
  const newBtn = h('button', { class: 'new-chat', type: 'button', onclick: () => ctx.onNew() }, icon('plus', 18), h('span', { text: t('newChat') }));
  const footer = h('div', { class: 'sidebar-foot' },
    h('button', { class: 'foot-btn', type: 'button', onclick: () => ctx.onSettings() }, icon('settings', 18), h('span', { text: t('settings') })),
    iconButton('keyboard', t('shortcuts'), () => ctx.onShortcuts()),
    iconButton('moon', t('toggleTheme'), () => ctx.onTheme(), 'theme-toggle'));
  root.replaceChildren(
    h('div', { class: 'sidebar-top' }, newBtn, h('div', { class: 'search-wrap' }, icon('search', 16), search)),
    list, footer);

  search.addEventListener('input', debounce(() => { query = search.value.trim().toLowerCase(); ctx.onSearch(); }, 120));

  function openItemMenu(anchor, conv) {
    menu(anchor, [
      { label: t('rename'), icon: 'edit', onclick: () => ctx.onAction('rename', conv.id) },
      { label: conv.pinned ? t('unpin') : t('pin'), icon: 'pin', onclick: () => ctx.onAction('pin', conv.id) },
      { label: t('exportMarkdown'), icon: 'download', onclick: () => ctx.onAction('exportMd', conv.id) },
      { label: t('exportJson'), icon: 'download', onclick: () => ctx.onAction('exportJson', conv.id) },
      { separator: true },
      { label: t('delete'), icon: 'trash', danger: true, onclick: () => ctx.onAction('delete', conv.id) },
    ], { align: 'end' });
  }

  function render(convs, activeId, busyIds) {
    const now = Date.now();
    const items = convs
      .filter((c) => !query || (c.title || '').toLowerCase().includes(query) || (c.text || '').includes(query))
      .sort((a, b) => b.updatedAt - a.updatedAt);
    list.replaceChildren();
    if (!items.length) {
      list.append(h('div', { class: 'empty-list', text: query ? t('noResults') : t('noChats') }));
      return;
    }
    const groups = new Map(ORDER.map((g) => [g, []]));
    for (const c of items) groups.get(query ? 'today' : groupOf(c, now)).push(c);
    for (const g of ORDER) {
      const arr = groups.get(g);
      if (!arr.length) continue;
      list.append(h('div', { class: 'group-title', text: query ? t('searchResults') : t('group_' + g) }));
      for (const c of arr) {
        const more = iconButton('more', t('more'), (e) => { e.stopPropagation(); openItemMenu(e.currentTarget, c); }, 'item-more tiny');
        const item = h('div', {
          class: 'chat-item' + (c.id === activeId ? ' active' : ''), role: 'link', tabindex: 0, dataset: { id: c.id },
          title: c.title || t('untitled'),
        },
        busyIds.has(c.id) ? h('span', { class: 'busy-dot', title: t('generating') }) : null,
        h('span', { class: 'chat-title', text: c.title || t('untitled') }),
        more);
        item.addEventListener('click', () => ctx.onSelect(c.id));
        item.addEventListener('keydown', (e) => { if (e.key === 'Enter') ctx.onSelect(c.id); });
        item.addEventListener('contextmenu', (e) => { e.preventDefault(); openItemMenu(item, c); });
        list.append(item);
      }
    }
  }

  return { render, focusSearch: () => { search.focus(); search.select(); } };
}
