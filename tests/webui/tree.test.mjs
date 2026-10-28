import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createConversation, addChild, newNode, pathTo, select, switchSibling, siblingsOf,
  removeSubtree, leafFrom, serialize, deserialize, toMarkdown,
} from '../../webui/js/tree.js';

function chat() {
  const { conv, nodes } = createConversation('sys');
  const u1 = addChild(nodes, conv.root, newNode('user', 'hi', conv.root));
  const a1 = addChild(nodes, u1.id, newNode('assistant', 'hello', u1.id, { reasoning: 'think' }));
  select(nodes, conv, a1.id);
  return { conv, nodes, u1, a1 };
}

test('path follows currNode from the root', () => {
  const { conv, nodes, u1, a1 } = chat();
  assert.deepEqual(pathTo(nodes, conv.currNode).map((n) => n.id), [u1.id, a1.id]);
  assert.deepEqual(pathTo(nodes, conv.currNode).map((n) => n.role).filter((r) => r !== 'root'), ['user', 'assistant']);
});

test('regenerate adds a sibling and switching restores the branch tail', () => {
  const { conv, nodes, u1, a1 } = chat();
  const a2 = addChild(nodes, u1.id, newNode('assistant', 'second', u1.id));
  select(nodes, conv, a2.id);
  assert.deepEqual(siblingsOf(nodes, a2.id), [a1.id, a2.id]);
  assert.ok(switchSibling(nodes, conv, a2.id, -1));
  assert.equal(conv.currNode, a1.id);
  assert.ok(!switchSibling(nodes, conv, a1.id, -1));
  assert.equal(leafFrom(nodes, u1.id), a1.id);
});

test('removeSubtree detaches and returns removed ids', () => {
  const { conv, nodes, u1, a1 } = chat();
  const removed = removeSubtree(nodes, conv, u1.id);
  assert.deepEqual(new Set(removed), new Set([u1.id, a1.id]));
  assert.equal(conv.currNode, conv.root);
  assert.deepEqual(nodes.get(conv.root).children, []);
  assert.deepEqual(removeSubtree(nodes, conv, conv.root), []);
});

test('serialize / deserialize round-trips with fresh ids', () => {
  const { conv, nodes } = chat();
  const copy = deserialize(JSON.parse(JSON.stringify(serialize(conv, nodes))));
  assert.notEqual(copy.conv.id, conv.id);
  for (const id of copy.nodes.keys()) assert.ok(!nodes.has(id), 'ids must not collide');
  const p = pathTo(copy.nodes, copy.conv.currNode).filter((n) => n.role !== 'root');
  assert.deepEqual(p.map((n) => n.content), ['hi', 'hello']);
  assert.equal(p[1].reasoning, 'think');
  assert.match(copy.conv.text, /hello/);
});

test('deserialize rejects garbage and sanitizes attachments', () => {
  assert.throws(() => deserialize(null));
  assert.throws(() => deserialize({ conv: { root: 'x' }, nodes: [] }));
  const { conv, nodes, u1 } = chat();
  u1.attachments = [
    { kind: 'image', name: 'ok.png', data: 'data:image/png;base64,AAAA' },
    { kind: 'image', name: 'bad', data: 'data:image/svg+xml;base64,AAAA' },
    { kind: 'image', name: 'js', data: 'javascript:alert(1)' },
    { kind: 'text', name: 'a.txt', data: 'body' },
    { kind: 'weird' },
    null,
  ];
  const copy = deserialize(JSON.parse(JSON.stringify(serialize(conv, nodes))));
  const user = [...copy.nodes.values()].find((n) => n.role === 'user');
  assert.deepEqual(user.attachments.map((a) => a.name), ['ok.png', 'a.txt']);
});

test('deserialize drops dangling child links and unknown roles', () => {
  const { conv, nodes, a1 } = chat();
  const data = JSON.parse(JSON.stringify(serialize(conv, nodes)));
  data.nodes.find((n) => n.id === a1.id).children = ['nope'];
  data.nodes.find((n) => n.id === a1.id).role = 'system';
  const copy = deserialize(data);
  for (const n of copy.nodes.values()) for (const c of n.children) assert.ok(copy.nodes.has(c));
  assert.ok([...copy.nodes.values()].every((n) => ['root', 'user', 'assistant'].includes(n.role)));
});

test('markdown export lists the visible path', () => {
  const { conv, nodes } = chat();
  const md = toMarkdown(conv, nodes, { untitled: 'Untitled', user: 'User', assistant: 'Assistant', thinking: 'Thinking', attachment: 'File' });
  assert.match(md, /## User\n\nhi/);
  assert.match(md, /## Assistant/);
  assert.match(md, /<details><summary>Thinking/);
});
