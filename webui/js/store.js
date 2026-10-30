// Persistence.  Conversations (metadata) and messages (nodes) live in separate
// IndexedDB stores so that saving one streamed reply never rewrites the whole
// chat, which matters when messages carry base64 images.  MemoryStore has the
// same interface for tests and for browsers that block IndexedDB.

const DB_NAME = 'q36-webui';
const DB_VERSION = 1;

function req(r) {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function txDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

class IdbStore {
  constructor(db) { this.db = db; this.persistent = true; }

  static open() {
    return new Promise((resolve, reject) => {
      const r = indexedDB.open(DB_NAME, DB_VERSION);
      r.onupgradeneeded = () => {
        const db = r.result;
        db.createObjectStore('convs', { keyPath: 'id' });
        const nodes = db.createObjectStore('nodes', { keyPath: 'id' });
        nodes.createIndex('conv', 'conv');
      };
      r.onsuccess = () => resolve(new IdbStore(r.result));
      r.onerror = () => reject(r.error);
      r.onblocked = () => reject(new Error('IndexedDB blocked'));
    });
  }

  async listConversations() {
    return req(this.db.transaction('convs').objectStore('convs').getAll());
  }

  async loadConversation(id) {
    const conv = await req(this.db.transaction('convs').objectStore('convs').get(id));
    if (!conv) return null;
    const rows = await req(this.db.transaction('nodes').objectStore('nodes').index('conv').getAll(id));
    return { conv, nodes: new Map(rows.map((n) => { const { conv: _c, ...node } = n; return [node.id, node]; })) };
  }

  async saveConversation(conv) {
    const tx = this.db.transaction('convs', 'readwrite');
    tx.objectStore('convs').put(conv);
    await txDone(tx);
  }

  async saveNodes(convId, nodes) {
    if (!nodes.length) return;
    const tx = this.db.transaction('nodes', 'readwrite');
    const store = tx.objectStore('nodes');
    for (const n of nodes) store.put({ ...n, conv: convId });
    await txDone(tx);
  }

  async deleteNodes(ids) {
    if (!ids.length) return;
    const tx = this.db.transaction('nodes', 'readwrite');
    for (const id of ids) tx.objectStore('nodes').delete(id);
    await txDone(tx);
  }

  async deleteConversation(id) {
    const tx = this.db.transaction(['convs', 'nodes'], 'readwrite');
    tx.objectStore('convs').delete(id);
    const idx = tx.objectStore('nodes').index('conv');
    const cur = idx.openKeyCursor(IDBKeyRange.only(id));
    cur.onsuccess = () => {
      const c = cur.result;
      if (c) { tx.objectStore('nodes').delete(c.primaryKey); c.continue(); }
    };
    await txDone(tx);
  }

  async clear() {
    const tx = this.db.transaction(['convs', 'nodes'], 'readwrite');
    tx.objectStore('convs').clear();
    tx.objectStore('nodes').clear();
    await txDone(tx);
  }
}

export class MemoryStore {
  constructor() { this.convs = new Map(); this.nodes = new Map(); this.persistent = false; }
  async listConversations() { return [...this.convs.values()].map((c) => ({ ...c })); }
  async loadConversation(id) {
    const conv = this.convs.get(id);
    if (!conv) return null;
    const nodes = new Map();
    for (const n of this.nodes.values()) if (n.conv === id) { const { conv: _c, ...node } = n; nodes.set(node.id, structuredClone(node)); }
    return { conv: { ...conv }, nodes };
  }
  async saveConversation(conv) { this.convs.set(conv.id, { ...conv }); }
  async saveNodes(convId, nodes) { for (const n of nodes) this.nodes.set(n.id, structuredClone({ ...n, conv: convId })); }
  async deleteNodes(ids) { for (const id of ids) this.nodes.delete(id); }
  async deleteConversation(id) {
    this.convs.delete(id);
    for (const [k, n] of this.nodes) if (n.conv === id) this.nodes.delete(k);
  }
  async clear() { this.convs.clear(); this.nodes.clear(); }
}

export async function openStore() {
  try {
    if (typeof indexedDB === 'undefined') throw new Error('no IndexedDB');
    return await IdbStore.open();
  } catch (e) {
    console.warn('IndexedDB unavailable, chats will not be saved:', e);
    return new MemoryStore();
  }
}
