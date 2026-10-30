// Tiny regex-based syntax highlighter. It is deliberately approximate: one
// master regex per language family picks out comments, strings, numbers and
// identifiers; everything else is only HTML-escaped.

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const words = (s) => new Set(s.split(/\s+/).filter(Boolean));

const KW = {
  c: words(`auto break case char const continue default do double else enum extern float for goto if inline int long register
    restrict return short signed sizeof static struct switch typedef union unsigned void volatile while bool true false NULL
    class namespace template typename public private protected virtual new delete this nullptr using try catch throw operator
    constexpr final override noexcept explicit mutable`),
  js: words(`async await break case catch class const continue debugger default delete do else export extends finally for from
    function if import in instanceof let new of return super switch this throw try typeof var void while with yield static get
    set true false null undefined interface type enum implements private public protected readonly abstract as declare namespace
    keyof satisfies`),
  go: words(`break case chan const continue default defer else fallthrough for func go goto if import interface map package range
    return select struct switch type var true false nil iota`),
  rust: words(`as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub
    ref return self Self static struct super trait true type unsafe use where while`),
  java: words(`abstract assert boolean break byte case catch char class const continue default do double else enum extends final
    finally float for goto if implements import instanceof int interface long native new package private protected public return
    short static strictfp super switch synchronized this throw throws transient try void volatile while var record true false null
    fun val when object companion data sealed override suspend inline lateinit`),
  py: words(`and as assert async await break class continue def del elif else except finally for from global if import in is lambda
    nonlocal not or pass raise return try while with yield True False None self cls match case`),
  sh: words(`if then else elif fi for while until do done case esac function in select time return exit break continue local export
    readonly unset shift source alias set declare echo cd`),
  sql: words(`select from where insert into values update set delete create table drop alter add column index primary key foreign
    references join inner left right outer full cross on group by order having limit offset union all distinct as and or not null
    is in like between exists case when then else end asc desc count sum avg min max with begin commit rollback default unique
    check constraint view trigger returning`),
  rb: words(`alias and begin break case class def defined do else elsif end ensure false for if in module next nil not or redo
    rescue retry return self super then true undef unless until when while yield puts require attr_accessor`),
  php: words(`abstract and array as break case catch class clone const continue declare default do echo else elseif empty endfor
    endforeach endif endswitch endwhile extends final finally fn for foreach function global if implements include instanceof
    interface isset list namespace new or print private protected public require return static switch throw trait try unset use
    var while yield true false null`),
  yaml: words('true false null yes no on off'),
  json: words('true false null'),
  css: words('important'),
  lua: words(`and break do else elseif end false for function goto if in local nil not or repeat return then true until while`),
};

const BUILTIN = {
  py: words('print len range int str float list dict set tuple bool type isinstance open map filter sum min max abs sorted enumerate zip super'),
  js: words('console window document Math JSON Object Array String Number Promise Map Set Date RegExp Error fetch require module process'),
  sh: words('ls cat grep sed awk find git make sudo apt pip npm curl wget mkdir rm cp mv chmod tar'),
};

// language alias -> [family, keyword set, comment style]
const LANGS = {};
function reg(names, family, kw, comment) {
  for (const n of names.split(' ')) LANGS[n] = { family, kw, comment };
}
reg('c h cpp c++ cc hpp cxx glsl cuda cu objc cs csharp', 'code', 'c', 'slash');
reg('js javascript jsx mjs cjs ts typescript tsx node', 'code', 'js', 'slash');
reg('go golang', 'code', 'go', 'slash');
reg('rust rs', 'code', 'rust', 'slash');
reg('java kotlin kt scala swift dart groovy', 'code', 'java', 'slash');
reg('php', 'code', 'php', 'both');
reg('py python python3 ipython', 'code', 'py', 'hash');
reg('sh bash shell zsh console fish', 'code', 'sh', 'hash');
reg('rb ruby', 'code', 'rb', 'hash');
reg('yaml yml toml ini conf properties dockerfile makefile make cmake', 'code', 'yaml', 'hash');
reg('sql mysql postgres postgresql sqlite psql', 'code', 'sql', 'dash');
reg('lua', 'code', 'lua', 'dash');
reg('json jsonc json5', 'json', 'json', 'slash');
reg('css scss less', 'css', 'css', 'slash');
reg('html xml svg xhtml vue', 'markup', null, null);
reg('diff patch', 'diff', null, null);

const COMMENT = {
  slash: String.raw`\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$)`,
  hash: String.raw`#[^\n]*`,
  dash: String.raw`--[^\n]*|\/\*[\s\S]*?(?:\*\/|$)`,
  both: String.raw`\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?(?:\*\/|$)`,
};
const STRING = String.raw`"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\[\s\S]|[^"\\\n])*(?:"|$)|'(?:\\[\s\S]|[^'\\\n])*(?:'|$)|\`(?:\\[\s\S]|[^\`\\])*(?:\`|$)`;
const NUMBER = String.raw`\b(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?[a-zA-Z]*)\b`;
const IDENT = String.raw`[A-Za-z_$][\w$]*`;

const cache = new Map();
function codeRegex(comment) {
  const key = 'code:' + comment;
  if (!cache.has(key)) {
    cache.set(key, new RegExp(`(${COMMENT[comment]})|(${STRING})|(${NUMBER})|(${IDENT})`, 'g'));
  }
  return cache.get(key);
}

const span = (cls, text) => `<span class="tok-${cls}">${esc(text)}</span>`;

function highlightCode(code, lang) {
  const kw = KW[lang.kw];
  const builtin = BUILTIN[lang.kw];
  const re = codeRegex(lang.comment);
  re.lastIndex = 0;
  let out = '';
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    last = m.index + m[0].length;
    if (m[1] !== undefined) out += span('comment', m[0]);
    else if (m[2] !== undefined) out += span('string', m[0]);
    else if (m[3] !== undefined) out += span('number', m[0]);
    else {
      const id = m[0];
      const next = code[last];
      if (kw.has(id) || (lang.kw === 'sql' && kw.has(id.toLowerCase()))) out += span('keyword', id);
      else if (builtin && builtin.has(id)) out += span('builtin', id);
      else if (next === '(') out += span('function', id);
      else if (/^[A-Z][a-z]/.test(id) && lang.kw !== 'sh') out += span('type', id);
      else out += esc(id);
    }
    if (m[0].length === 0) re.lastIndex++;
  }
  return out + esc(code.slice(last));
}

function highlightJson(code) {
  const re = /("(?:\\.|[^"\\\n])*")(\s*:)?|(-?\b\d[\d.eE+-]*\b)|\b(true|false|null)\b|(\/\/[^\n]*|\/\*[\s\S]*?\*\/)/g;
  let out = '';
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    last = m.index + m[0].length;
    if (m[1] !== undefined) out += span(m[2] ? 'property' : 'string', m[1]) + esc(m[2] || '');
    else if (m[3] !== undefined) out += span('number', m[3]);
    else if (m[4] !== undefined) out += span('keyword', m[4]);
    else out += span('comment', m[5]);
  }
  return out + esc(code.slice(last));
}

function highlightCss(code) {
  const re = /(\/\*[\s\S]*?(?:\*\/|$))|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(#[0-9a-fA-F]{3,8}\b|-?\b\d[\d.]*(?:px|em|rem|%|vh|vw|s|ms|deg|fr)?\b)|(@[\w-]+)|([\w-]+)(?=\s*:)|([.#][\w-]+)/g;
  let out = '';
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    last = m.index + m[0].length;
    if (m[1] !== undefined) out += span('comment', m[0]);
    else if (m[2] !== undefined) out += span('string', m[0]);
    else if (m[3] !== undefined) out += span('number', m[0]);
    else if (m[4] !== undefined) out += span('keyword', m[0]);
    else if (m[5] !== undefined) out += span('property', m[0]);
    else out += span('type', m[0]);
  }
  return out + esc(code.slice(last));
}

function highlightMarkup(code) {
  const re = /(<!--[\s\S]*?(?:-->|$))|(<\/?)([A-Za-z][\w:-]*)((?:"[^"]*"|'[^']*'|[^>"'])*)(\/?>)?/g;
  let out = '';
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    last = m.index + m[0].length;
    if (m[1] !== undefined) { out += span('comment', m[0]); continue; }
    out += esc(m[2]) + span('keyword', m[3]);
    out += esc(m[4]).replace(/([\w:@.-]+)(=)?(&quot;[^]*?&quot;|"[^"]*"|'[^']*')?/g, (all, name, eq, val) => {
      return `<span class="tok-property">${name}</span>${eq || ''}${val ? `<span class="tok-string">${val}</span>` : ''}`;
    });
    out += esc(m[5] || '');
  }
  return out + esc(code.slice(last));
}

function highlightDiff(code) {
  return code.split('\n').map((line) => {
    if (/^(\+\+\+|---|diff |index )/.test(line)) return span('keyword', line);
    if (line.startsWith('@@')) return span('type', line);
    if (line.startsWith('+')) return `<span class="tok-add">${esc(line)}</span>`;
    if (line.startsWith('-')) return `<span class="tok-del">${esc(line)}</span>`;
    return esc(line);
  }).join('\n');
}

// Very long blocks (logs, data dumps) are not worth tokenising on every
// streaming update.
const MAX_HIGHLIGHT = 60000;

export function highlight(code, langName) {
  const name = (langName || '').toLowerCase();
  const lang = LANGS[name];
  if (!lang || code.length > MAX_HIGHLIGHT) return esc(code);
  switch (lang.family) {
    case 'json': return highlightJson(code);
    case 'css': return highlightCss(code);
    case 'markup': return highlightMarkup(code);
    case 'diff': return highlightDiff(code);
    default: return highlightCode(code, lang);
  }
}

export function isPreviewable(lang) {
  return ['html', 'htm', 'svg', 'xhtml'].includes((lang || '').toLowerCase());
}
