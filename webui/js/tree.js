// Conversation tree.  Every message is a node; editing a prompt or regenerating
// a reply adds a sibling instead of overwriting, so no branch is ever lost.
// A conversation is a Map of nodes keyed by id plus a `root` node id.  The
// visible chat is the path from the root to `conv.currNode`.

import { uid } from './util.js';

export function newNode(role, content, parent, extra = {}) {
  return {
    id: uid(),
    parent,
    role,
    content,
    reasoning: '',
    attachments: [],
    children: [],
    cur: null, // last visited child, so switching branches restores the old tail
    ts: Date.now(),
    ...extra,
  };
}

export function createConversation(system = '') {
  const root = newNode('root', '', null);
  const conv = {
    id: uid(),
    title: '',
    system,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    pinned: false,
    root: root.id,
    currNode: root.id,
    text: '',
  };
  return { conv, nodes: new Map([[root.id, root]]) };
}

export function addChild(nodes, parentId, node) {
  node.parent = parentId;
  const parent = nodes.get(parentId);
  parent.children.push(node.id);
  parent.cur = node.id;
  nodes.set(node.id, node);
  return node;
}

// Messages from the first reply to `leafId` (the root itself is not a message).
export function pathTo(nodes, leafId) {
  const path = [];
  for (let n = nodes.get(leafId); n && n.role !== 'root'; n = nodes.get(n.parent)) path.push(n);
  return path.reverse();
}

// Follow remembered choices down from `id` to the leaf to show.
export function leafFrom(nodes, id) {
  let n = nodes.get(id);
  while (n && n.children.length) {
    const next = n.cur && n.children.includes(n.cur) ? n.cur : n.children[n.children.length - 1];
    n = nodes.get(next);
  }
  return n ? n.id : id;
}

// Make `leafId` the current leaf and remember the path so that later branch
// switches come back to the same place.
export function select(nodes, conv, leafId) {
  let child = nodes.get(leafId);
  while (child && child.parent) {
    const parent = nodes.get(child.parent);
    parent.cur = child.id;
    child = parent;
  }
  conv.currNode = leafId;
}

export function siblingsOf(nodes, id) {
  const n = nodes.get(id);
  if (!n || !n.parent) return [id];
  return nodes.get(n.parent).children;
}

export function switchSibling(nodes, conv, id, delta) {
  const sibs = siblingsOf(nodes, id);
  const idx = sibs.indexOf(id) + delta;
  if (idx < 0 || idx >= sibs.length) return false;
  select(nodes, conv, leafFrom(nodes, sibs[idx]));
  return true;
}

// Removes `id` and everything below it; returns the removed ids.
export function removeSubtree(nodes, conv, id) {
  const node = nodes.get(id);
  if (!node || node.role === 'root') return [];
  const removed = [];
  const stack = [id];
  while (stack.length) {
    const nid = stack.pop();
    const n = nodes.get(nid);
    if (!n) continue;
    removed.push(nid);
    stack.push(...n.children);
    nodes.delete(nid);
  }
  const parent = nodes.get(node.parent);
  parent.children = parent.children.filter((c) => c !== id);
  if (parent.cur === id) parent.cur = parent.children[parent.children.length - 1] || null;
  if (!nodes.has(conv.currNode)) select(nodes, conv, leafFrom(nodes, parent.id));
  return removed;
}

export function messageText(node) {
  return node.content || '';
}

// Flat text used for sidebar search; capped so one huge chat stays cheap.
export function searchText(nodes) {
  let out = '';
  for (const n of nodes.values()) {
    if (n.role === 'root') continue;
    out += n.content + '\n';
    if (out.length > 40000) break;
  }
  return out.slice(0, 40000).toLowerCase();
}

export function deriveTitle(text) {
  const line = (text || '').split('\n').map((l) => l.trim()).find(Boolean) || '';
  const clean = line.replace(/[#*_`>]+/g, '').replace(/\s+/g, ' ').trim();
  return clean.length > 48 ? clean.slice(0, 47).trimEnd() + '…' : clean;
}

export function serialize(conv, nodes) {
  return { version: 1, conv: { ...conv }, nodes: [...nodes.values()] };
}

const str = (v) => (typeof v === 'string' ? v : '');
const IMAGE_URI = /^data:image\/(png|jpe?g|webp|gif);base64,[A-Za-z0-9+/=]+$/;

function cleanAttachment(a) {
  if (!a || typeof a !== 'object') return null;
  if (a.kind === 'image' && typeof a.data === 'string' && IMAGE_URI.test(a.data)) return { id: uid(), kind: 'image', name: str(a.name) || 'image', data: a.data, w: +a.w || 0, h: +a.h || 0 };
  if (a.kind === 'text' && typeof a.data === 'string') return { id: uid(), kind: 'text', name: str(a.name) || 'file.txt', data: a.data };
  return null;
}

// Imported files are untrusted: validate every field and give everything fresh
// ids so an import can never overwrite an existing chat's messages.
export function deserialize(data) {
  if (!data || typeof data !== 'object' || !data.conv || !Array.isArray(data.nodes)) throw new Error('Not a chat export');
  const raw = new Map();
  for (const n of data.nodes) if (n && typeof n.id === 'string') raw.set(n.id, n);
  if (!raw.has(data.conv.root)) throw new Error('Chat export has no root');

  const ids = new Map([...raw.keys()].map((id) => [id, uid()]));
  const nodes = new Map();
  for (const [oldId, n] of raw) {
    const role = ['root', 'user', 'assistant'].includes(n.role) ? n.role : 'user';
    const node = newNode(role, str(n.content), ids.get(n.parent) ?? null, {
      reasoning: str(n.reasoning),
      attachments: (Array.isArray(n.attachments) ? n.attachments : []).map(cleanAttachment).filter(Boolean),
      children: (Array.isArray(n.children) ? n.children : []).filter((c) => ids.has(c)).map((c) => ids.get(c)),
      ts: Number.isFinite(n.ts) ? n.ts : Date.now(),
    });
    node.id = ids.get(oldId);
    node.cur = ids.get(n.cur) ?? null;
    if (n.stats && typeof n.stats === 'object') node.stats = n.stats;
    if (Number.isFinite(n.reasoningMs)) node.reasoningMs = n.reasoningMs;
    if (typeof n.finish === 'string') node.finish = n.finish;
    if (n.aborted) node.aborted = true;
    if (typeof n.error === 'string') node.error = n.error;
    nodes.set(node.id, node);
  }
  const c = data.conv;
  const root = ids.get(c.root);
  const conv = {
    id: uid(), title: str(c.title), system: str(c.system),
    createdAt: Number.isFinite(c.createdAt) ? c.createdAt : Date.now(),
    updatedAt: Number.isFinite(c.updatedAt) ? c.updatedAt : Date.now(),
    pinned: !!c.pinned, root, currNode: ids.get(c.currNode) ?? root, text: '',
  };
  conv.text = searchText(nodes);
  return { conv, nodes };
}

export function toMarkdown(conv, nodes, labels) {
  const lines = [`# ${conv.title || labels.untitled}`, ''];
  for (const n of pathTo(nodes, conv.currNode)) {
    lines.push(`## ${n.role === 'user' ? labels.user : labels.assistant}`, '');
    if (n.reasoning) lines.push('<details><summary>' + labels.thinking + '</summary>', '', n.reasoning, '', '</details>', '');
    for (const a of n.attachments || []) lines.push(`_${labels.attachment}: ${a.name}_`, '');
    lines.push(n.content, '');
  }
  return lines.join('\n');
}
