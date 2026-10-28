// Markdown renderer for chat output.  It escapes all HTML in the source (model
// output is untrusted) and builds markup only from the constructs it parses.
// renderBlocks() returns one HTML string per top-level block so the message
// view can update only the blocks that changed while a reply is streaming.

import { highlight, isPreviewable } from './highlight.js';
import { texToHtml } from './math.js';

export function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const labels = { copy: 'Copy', preview: 'Preview', wrap: 'Wrap' };
export function setMarkdownLabels(l) { Object.assign(labels, l); }

// ---------------------------------------------------------------- inline ----

const PH_OPEN = '\uE000';
const PH_CLOSE = '\uE001';
const PH_RE = /\uE000(\d+)\uE001/g;

function safeUrl(url) {
  const u = url.trim();
  if (/^(https?:|mailto:|#|\/(?!\/))/i.test(u)) return u;
  return null;
}

function linkHtml(href, inner) {
  const external = /^https?:/i.test(href);
  return `<a href="${href}"${external ? ' target="_blank" rel="noopener noreferrer"' : ''}>${inner}</a>`;
}

function stash(ctx, html) {
  ctx.items.push(html);
  return PH_OPEN + (ctx.items.length - 1) + PH_CLOSE;
}

function restore(s, ctx) {
  let prev;
  do {
    prev = s;
    s = s.replace(PH_RE, (_, i) => ctx.items[+i]);
  } while (s !== prev && s.includes(PH_OPEN));
  return s;
}

// Runs before HTML escaping: pulls out spans whose content must stay verbatim.
function protectSpans(s, ctx) {
  s = s.replace(/(?<!`)(`+)(?!`)([\s\S]+?)(?<!`)\1(?!`)/g, (_, _t, code) => {
    const c = code.length > 2 && code.startsWith(' ') && code.endsWith(' ') && code.trim() ? code.slice(1, -1) : code;
    return stash(ctx, `<code>${escapeHtml(c)}</code>`);
  });
  s = s.replace(/\$\$([^\n]+?)\$\$/g, (_, tex) => stash(ctx, texToHtml(tex, true)));
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => stash(ctx, texToHtml(tex)));
  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => stash(ctx, texToHtml(tex, true)));
  // $...$ must not look like currency: no space inside the delimiters and no
  // digit right after the closing dollar.
  s = s.replace(/(?<![\\$\w])\$(?![\s$])((?:\\.|[^$\\\n])+?)(?<![\s\\])\$(?![\d\w$])/g,
    (_, tex) => stash(ctx, texToHtml(tex)));
  s = s.replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])/g, (_, c) => stash(ctx, escapeHtml(c)));
  return s;
}

function format(s, ctx) {
  // Images are not loaded (the page CSP forbids remote images and a model
  // should not be able to beacon out); show them as links instead.
  s = s.replace(/(!?)\[((?:[^[\]]|\[[^[\]]*\])*)\]\(\s*(&lt;[^\s]*?&gt;|[^\s()]*(?:\([^\s()]*\)[^\s()]*)*)(?:\s+"([^"]*)")?\s*\)/g,
    (m, bang, label, url, title) => {
      url = url.replace(/^&lt;(.*)&gt;$/, '$1').replace(/&amp;/g, '&');
      const href = safeUrl(url);
      const text = format(label, ctx) || escapeHtml(url);
      if (!href) return stash(ctx, text);
      const safeHref = escapeHtml(href);
      const inner = bang ? `🖼 ${text}` : text;
      const a = linkHtml(safeHref, inner);
      return stash(ctx, title ? a.replace('<a ', `<a title="${escapeHtml(title)}" `) : a);
    });
  s = s.replace(/&lt;((?:https?:\/\/|mailto:)[^\s<>]+?)&gt;/g,
    (_, url) => stash(ctx, linkHtml(url, url)));
  s = s.replace(/\bhttps?:\/\/[^\s<>]+/g, (url) => {
    const trail = /[.,;:!?'")\]}]+$/.exec(url);
    let end = trail ? trail[0] : '';
    let body = url.slice(0, url.length - end.length);
    // Keep a balancing paren that is part of the URL (wikipedia style).
    if (end.startsWith(')') && (body.match(/\(/g) || []).length > (body.match(/\)/g) || []).length) {
      body += ')';
      end = end.slice(1);
    }
    return stash(ctx, linkHtml(body, body)) + end;
  });

  s = s.replace(/(\*\*\*)(?=\S)([\s\S]*?\S)\1/g, '<strong><em>$2</em></strong>');
  s = s.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(?<![\w_])__(?=\S)([\s\S]*?\S)__(?![\w_])/g, '<strong>$1</strong>');
  s = s.replace(/(?<!\*)\*(?![\s*])([^*\n]+?)(?<![\s*])\*(?!\*)/g, '<em>$1</em>');
  s = s.replace(/(?<![\w_])_(?![\s_])([^_\n]+?)(?<![\s_])_(?![\w_])/g, '<em>$1</em>');
  s = s.replace(/~~(?=\S)([\s\S]*?\S)~~/g, '<del>$1</del>');
  return s;
}

export function renderInline(text) {
  const ctx = { items: [] };
  let s = protectSpans(text, ctx);
  s = escapeHtml(s).replace(/&quot;/g, '"');
  // Models often emit <br> inside table cells; allow exactly that tag.
  s = s.replace(/&lt;br\s*\/?&gt;/gi, '<br>');
  s = s.replace(/&amp;(#\d{1,6}|#x[0-9a-fA-F]{1,5}|[a-zA-Z]{2,8});/g, '&$1;');
  s = format(s, ctx);
  s = s.replace(/ {2,}\n|\\\n/g, '<br>').replace(/\n/g, '<br>');
  return restore(s, ctx);
}

// ----------------------------------------------------------------- blocks ---

const reFence = /^( {0,3})(`{3,}|~{3,})[ \t]*([^\s`]*)[^`]*$/;
const reHeading = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
const reHr = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const reBullet = /^( {0,3})([-*+])([ \t]+|$)/;
const reOrdered = /^( {0,3})(\d{1,9})([.)])([ \t]+|$)/;
const reQuote = /^ {0,3}>/;
const reTableSep = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
const reSetext = /^ {0,3}(=+|-+)[ \t]*$/;

const isBlank = (l) => !l.trim();
const indentOf = (l) => /^ */.exec(l)[0].length;

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
  const cells = [];
  let cur = '';
  let inCode = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '\\' && s[i + 1] === '|') { cur += '|'; i++; continue; }
    if (c === '`') inCode = !inCode;
    if (c === '|' && !inCode) { cells.push(cur.trim()); cur = ''; continue; }
    cur += c;
  }
  cells.push(cur.trim());
  return cells;
}

function isTableStart(lines, i) {
  return i + 1 < lines.length && lines[i].includes('|') && reTableSep.test(lines[i + 1]) &&
    lines[i + 1].includes('-') && (lines[i + 1].includes('|') || splitRow(lines[i]).length === 1) &&
    splitRow(lines[i]).length === splitRow(lines[i + 1]).length;
}

function mathStart(line) {
  const t = line.trim();
  if (t.startsWith('$$')) return '$$';
  if (t.startsWith('\\[')) return '\\[';
  return null;
}

function startsBlock(lines, i) {
  const l = lines[i];
  if (reFence.test(l) || reHeading.test(l) || reHr.test(l) || reQuote.test(l)) return true;
  const b = reBullet.exec(l);
  if (b && l.slice(b[0].length).trim()) return true;
  const o = reOrdered.exec(l);
  if (o && o[2] === '1' && l.slice(o[0].length).trim()) return true;
  if (mathStart(l)) return true;
  return isTableStart(lines, i);
}

function codeBlockHtml(code, lang, open) {
  const label = lang || 'text';
  const preview = !open && isPreviewable(lang) ?
    `<button type="button" class="code-btn" data-action="preview">${escapeHtml(labels.preview)}</button>` : '';
  return `<div class="codeblock${open ? ' streaming' : ''}" data-lang="${escapeHtml(label)}">` +
    `<div class="code-head"><span class="code-lang">${escapeHtml(label)}</span><span class="grow"></span>` +
    `${preview}<button type="button" class="code-btn" data-action="wrap">${escapeHtml(labels.wrap)}</button>` +
    `<button type="button" class="code-btn" data-action="copy">${escapeHtml(labels.copy)}</button></div>` +
    `<pre><code>${highlight(code, lang)}</code></pre></div>`;
}

function parseTable(lines, i) {
  const header = splitRow(lines[i]);
  const aligns = splitRow(lines[i + 1]).map((c) => {
    const l = c.startsWith(':');
    const r = c.endsWith(':');
    return l && r ? 'center' : r ? 'right' : l ? 'left' : '';
  });
  const cell = (tag, text, idx) => {
    const a = aligns[idx] ? ` style="text-align:${aligns[idx]}"` : '';
    return `<${tag}${a}>${renderInline(text)}</${tag}>`;
  };
  let html = '<div class="table-wrap"><table><thead><tr>' +
    header.map((c, k) => cell('th', c, k)).join('') + '</tr></thead><tbody>';
  i += 2;
  while (i < lines.length && !isBlank(lines[i]) && lines[i].includes('|') && !reFence.test(lines[i])) {
    const cells = splitRow(lines[i]);
    html += '<tr>' + header.map((_, k) => cell('td', cells[k] ?? '', k)).join('') + '</tr>';
    i++;
  }
  return { html: html + '</tbody></table></div>', next: i };
}

function matchMarker(line) {
  if (reHr.test(line)) return null;
  let m = reBullet.exec(line);
  if (m) return { ordered: false, indent: m[1].length, width: m[1].length + 1, spaces: m[3].length, len: m[0].length, line };
  m = reOrdered.exec(line);
  if (m) {
    return {
      ordered: true, indent: m[1].length, width: m[1].length + m[2].length + 1, spaces: m[4].length,
      len: m[0].length, start: parseInt(m[2], 10), line,
    };
  }
  return null;
}

function parseList(lines, i) {
  const first = matchMarker(lines[i]);
  const items = [];
  let loose = false;
  while (i < lines.length) {
    const m = matchMarker(lines[i]);
    if (!m || m.ordered !== first.ordered) break;
    const blankStart = !lines[i].slice(m.len).trim();
    const contentIndent = m.width + (blankStart || m.spaces > 4 ? 1 : m.spaces);
    const itemLines = [blankStart ? '' : lines[i].slice(m.len)];
    let blankInside = false;
    i++;
    while (i < lines.length) {
      const line = lines[i];
      if (isBlank(line)) {
        let j = i;
        while (j < lines.length && isBlank(lines[j])) j++;
        if (j >= lines.length || indentOf(lines[j]) < contentIndent) break;
        for (; i < j; i++) itemLines.push('');
        blankInside = true;
        continue;
      }
      if (indentOf(line) >= contentIndent) {
        itemLines.push(line.slice(contentIndent));
        i++;
        continue;
      }
      const nested = matchMarker(line);
      if (nested && nested.indent >= m.indent + 2) {
        // Models often indent a nested list by 2 spaces under "1. ".
        itemLines.push(line.slice(Math.min(indentOf(line), contentIndent)));
        i++;
        continue;
      }
      const prev = itemLines[itemLines.length - 1];
      if (prev && prev.trim() && !startsBlock(lines, i) && !matchMarker(line)) {
        itemLines.push(line.trim());
        i++;
        continue;
      }
      break;
    }
    // Only treat blank lines between paragraphs of one item as "loose"; a blank
    // line before a nested block does not change how the item itself renders.
    const looseInside = blankInside && itemLines.some((l, k) => k > 0 && isBlank(l) &&
      itemLines.slice(k + 1).some((x) => x.trim() && !matchMarker(x) && indentOf(x) < 2));
    items.push({ lines: itemLines, looseInside });
    if (i < lines.length && isBlank(lines[i])) {
      let j = i;
      while (j < lines.length && isBlank(lines[j])) j++;
      const next = j < lines.length ? matchMarker(lines[j]) : null;
      if (next && next.ordered === first.ordered && next.indent < 4) { loose = true; i = j; } else break;
    }
  }
  const isLoose = loose || items.some((it) => it.looseInside);
  const body = items.map((it) => {
    let task = '';
    const tm = /^\[([ xX])\][ \t]+/.exec(it.lines[0]);
    if (tm) {
      task = `<input type="checkbox" disabled${tm[1] !== ' ' ? ' checked' : ''}> `;
      it.lines[0] = it.lines[0].slice(tm[0].length);
    }
    const inner = renderLines(it.lines, !isLoose).join('');
    return `<li${tm ? ' class="task"' : ''}>${task}${inner}</li>`;
  }).join('');
  const tag = first.ordered ? 'ol' : 'ul';
  const start = first.ordered && first.start !== 1 ? ` start="${first.start}"` : '';
  return { html: `<${tag}${start}>${body}</${tag}>`, next: i };
}

function parseMath(lines, i, open) {
  const close = open === '$$' ? '$$' : '\\]';
  let first = lines[i].trim().slice(open.length);
  const buf = [];
  const sameLine = first.indexOf(close);
  if (sameLine >= 0) {
    return { html: `<div class="math-block">${texToHtml(first.slice(0, sameLine), true)}</div>`, next: i + 1 };
  }
  if (first.trim()) buf.push(first);
  i++;
  while (i < lines.length) {
    const k = lines[i].indexOf(close);
    if (k >= 0) {
      buf.push(lines[i].slice(0, k));
      i++;
      break;
    }
    buf.push(lines[i]);
    i++;
  }
  return { html: `<div class="math-block">${texToHtml(buf.join('\n'), true)}</div>`, next: i };
}

function renderLines(lines, tight = false) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) { i++; continue; }

    let m = reFence.exec(line);
    if (m) {
      const indent = m[1].length;
      const fence = m[2];
      const code = [];
      i++;
      let closed = false;
      const closeRe = new RegExp(`^ {0,3}${fence[0] === '`' ? '`' : '~'}{${fence.length},}[ \\t]*$`);
      while (i < lines.length) {
        if (closeRe.test(lines[i])) { closed = true; i++; break; }
        code.push(indent ? lines[i].replace(new RegExp(`^ {0,${indent}}`), '') : lines[i]);
        i++;
      }
      out.push(codeBlockHtml(code.join('\n'), m[3], !closed));
      continue;
    }

    m = reHeading.exec(line);
    if (m) {
      out.push(`<h${m[1].length}>${renderInline(m[2] || '')}</h${m[1].length}>`);
      i++;
      continue;
    }
    if (reHr.test(line)) { out.push('<hr>'); i++; continue; }

    const ms = mathStart(line);
    if (ms) {
      const r = parseMath(lines, i, ms);
      out.push(r.html);
      i = r.next;
      continue;
    }

    if (reQuote.test(line)) {
      const inner = [];
      while (i < lines.length) {
        if (reQuote.test(lines[i])) inner.push(lines[i].replace(/^ {0,3}> ?/, ''));
        else if (!isBlank(lines[i]) && inner.length && !isBlank(inner[inner.length - 1]) && !startsBlock(lines, i)) inner.push(lines[i]);
        else break;
        i++;
      }
      out.push(`<blockquote>${renderLines(inner).join('')}</blockquote>`);
      continue;
    }

    if (isTableStart(lines, i)) {
      const r = parseTable(lines, i);
      out.push(r.html);
      i = r.next;
      continue;
    }

    if (matchMarker(line)) {
      const r = parseList(lines, i);
      out.push(r.html);
      i = r.next;
      continue;
    }

    // Paragraph (possibly a setext heading).
    const para = [line];
    i++;
    while (i < lines.length && !isBlank(lines[i])) {
      if (reSetext.test(lines[i]) && para.length) {
        const level = lines[i].trim()[0] === '=' ? 1 : 2;
        out.push(`<h${level}>${renderInline(para.join('\n'))}</h${level}>`);
        para.length = 0;
        i++;
        break;
      }
      if (startsBlock(lines, i)) break;
      para.push(lines[i]);
      i++;
    }
    if (para.length) {
      const html = renderInline(para.join('\n').replace(/^[ \t]+/gm, ''));
      out.push(tight ? html : `<p>${html}</p>`);
    }
  }
  return out;
}

export function renderBlocks(text) {
  return renderLines(text.replace(/\r\n?/g, '\n').replace(/\t/g, '    ').split('\n'));
}

export function renderMarkdown(text) {
  return renderBlocks(text).join('');
}
