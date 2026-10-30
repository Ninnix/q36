import test from 'node:test';
import assert from 'node:assert/strict';
import { renderMarkdown, renderInline } from '../../webui/js/markdown.js';
import { texToHtml } from '../../webui/js/math.js';

const md = renderMarkdown;

test('escapes raw HTML', () => {
  const html = md('<script>alert(1)</script> <img src=x onerror=alert(1)>');
  assert.ok(!html.includes('<script'));
  assert.ok(!html.includes('<img'));
  assert.match(html, /&lt;script&gt;/);
});

test('rejects javascript: links and does not load images', () => {
  assert.ok(!md('[x](javascript:alert(1))').includes('href='));
  const img = md('![alt](https://example.com/a.png)');
  assert.ok(!img.includes('<img'));
  assert.match(img, /<a href="https:\/\/example.com\/a.png"/);
});

test('headings, emphasis, inline code', () => {
  assert.equal(md('# Title'), '<h1>Title</h1>');
  assert.equal(md('**b** *i* ~~s~~ `c<d>`'), '<p><strong>b</strong> <em>i</em> <del>s</del> <code>c&lt;d&gt;</code></p>');
  assert.equal(md('snake_case_name and 2 * 3 * 4'), '<p>snake_case_name and 2 * 3 * 4</p>');
});

test('fenced code with language, unclosed fence while streaming', () => {
  const closed = md('```js\nconst a = 1;\n```');
  assert.match(closed, /class="codeblock"/);
  assert.match(closed, /tok-keyword">const</);
  const open = md('```py\ndef f():\n    return 1');
  assert.match(open, /codeblock streaming/);
  assert.match(open, /return/);
});

test('code block content is never interpreted as markdown', () => {
  const html = md('```\n**not bold** <b>x</b>\n```');
  assert.ok(!html.includes('<strong>'));
  assert.match(html, /&lt;b&gt;x&lt;\/b&gt;/);
});

test('lists: nested, ordered start, tasks, tight vs loose', () => {
  assert.equal(md('- a\n- b'), '<ul><li>a</li><li>b</li></ul>');
  assert.equal(md('3. a\n4. b'), '<ol start="3"><li>a</li><li>b</li></ol>');
  assert.match(md('- a\n  - b\n  - c\n- d'), /<li>a<ul><li>b<\/li><li>c<\/li><\/ul><\/li><li>d<\/li>/);
  assert.match(md('1. a\n  - b'), /<ol><li>a<ul><li>b<\/li><\/ul><\/li><\/ol>/);
  assert.match(md('- [x] done\n- [ ] todo'), /<li class="task"><input type="checkbox" disabled checked> done<\/li>/);
  assert.match(md('- a\n\n- b'), /<li><p>a<\/p><\/li><li><p>b<\/p><\/li>/);
});

test('code fence inside a list item', () => {
  const html = md('1. Run:\n   ```sh\n   make\n   ```\n2. Done');
  assert.match(html, /<ol><li>Run:<div class="codeblock"/);
  assert.match(html, /<li>Done<\/li>/);
});

test('blockquote and hr', () => {
  assert.equal(md('> quote\n> more'), '<blockquote><p>quote<br>more</p></blockquote>');
  assert.equal(md('---'), '<hr>');
});

test('tables with alignment and escaped pipes', () => {
  const html = md('| a | b |\n|:--|--:|\n| 1 | x\\|y |');
  assert.match(html, /<th style="text-align:left">a<\/th><th style="text-align:right">b<\/th>/);
  assert.match(html, /<td style="text-align:right">x\|y<\/td>/);
});

test('table directly after a paragraph line', () => {
  const html = md('Here:\n| a | b |\n|---|---|\n| 1 | 2 |');
  assert.match(html, /<p>Here:<\/p><div class="table-wrap">/);
});

test('br tag allowed in table cells only as <br>', () => {
  assert.match(renderInline('a<br>b'), /a<br>b/);
  assert.ok(!renderInline('a<br onclick=x>b').includes('<br onclick'));
});

test('math: inline, display, currency is left alone', () => {
  assert.match(md('Let $x^2$ be'), /<sup><i>2<\/i>|<sup>2<\/sup>/);
  assert.match(md('$$\n\\frac{a}{b}\n$$'), /math-block.*m-frac/s);
  assert.match(md('\\[ \\sqrt{x} \\]'), /m-sqrt/);
  assert.equal(md('costs $5 and $10 total'), '<p>costs $5 and $10 total</p>');
});

test('latex converter handles common constructs', () => {
  assert.match(texToHtml('\\alpha + \\beta'), /α.*β/s);
  assert.match(texToHtml('\\sum_{i=1}^{n} i'), /m-bigop/);
  assert.match(texToHtml('\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}'), /m-matrix.*m-row.*m-row/s);
  assert.match(texToHtml('\\unknowncmd'), /m-unknown/);
  // unbalanced braces must terminate
  assert.ok(texToHtml('a } b { c').length > 0);
});

test('setext heading and plain paragraphs keep line breaks', () => {
  assert.equal(md('Title\n=====').startsWith('<h1>Title</h1>'), true);
  assert.equal(md('a\nb'), '<p>a<br>b</p>');
});

test('autolinks trim trailing punctuation', () => {
  const html = md('see https://example.com/a(b), ok.');
  assert.match(html, /href="https:\/\/example.com\/a\(b\)"/);
});
