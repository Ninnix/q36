// Best-effort LaTeX to HTML for chat output. There is no layout engine here:
// fractions, roots, scripts, accents, matrices and the common symbol set are
// covered; anything unknown is shown as its raw command so nothing is lost.

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const SYMBOLS = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ϵ', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ', vartheta: 'ϑ',
  iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', varpi: 'ϖ', rho: 'ρ', varrho: 'ϱ', sigma: 'σ',
  varsigma: 'ς', tau: 'τ', upsilon: 'υ', phi: 'ϕ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
  Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
  times: '×', cdot: '·', div: '÷', pm: '±', mp: '∓', ast: '∗', star: '⋆', circ: '∘', bullet: '•', oplus: '⊕', otimes: '⊗',
  leq: '≤', le: '≤', geq: '≥', ge: '≥', neq: '≠', ne: '≠', approx: '≈', equiv: '≡', sim: '∼', simeq: '≃', cong: '≅', propto: '∝',
  ll: '≪', gg: '≫', infty: '∞', partial: '∂', nabla: '∇', sum: '∑', prod: '∏', coprod: '∐', int: '∫', iint: '∬', iiint: '∭', oint: '∮',
  to: '→', rightarrow: '→', leftarrow: '←', gets: '←', uparrow: '↑', downarrow: '↓', leftrightarrow: '↔', Rightarrow: '⇒', Leftarrow: '⇐',
  Leftrightarrow: '⇔', iff: '⇔', implies: '⇒', mapsto: '↦', longrightarrow: '⟶', longleftarrow: '⟵', hookrightarrow: '↪',
  in: '∈', notin: '∉', ni: '∋', subset: '⊂', subseteq: '⊆', supset: '⊃', supseteq: '⊇', cup: '∪', cap: '∩', setminus: '∖',
  emptyset: '∅', varnothing: '∅', forall: '∀', exists: '∃', nexists: '∄', neg: '¬', lnot: '¬', land: '∧', wedge: '∧', lor: '∨', vee: '∨',
  ldots: '…', dots: '…', cdots: '⋯', vdots: '⋮', ddots: '⋱', angle: '∠', degree: '°', perp: '⊥', parallel: '∥', hbar: 'ℏ', ell: 'ℓ',
  prime: '′', Re: 'ℜ', Im: 'ℑ', aleph: 'ℵ', top: '⊤', bot: '⊥', vdash: '⊢', models: '⊨', therefore: '∴', because: '∵', square: '□',
  triangle: '△', langle: '⟨', rangle: '⟩', lfloor: '⌊', rfloor: '⌋', lceil: '⌈', rceil: '⌉', lbrace: '{', rbrace: '}', vert: '|', Vert: '‖',
  mid: '∣', backslash: '\\', checkmark: '✓', dagger: '†', ddagger: '‡', S: '§', P: '¶', copyright: '©', pounds: '£', euro: '€', textdegree: '°',
  // Spacing.
  quad: ' ', qquad: '  ', ',': ' ', ';': ' ', ':': ' ', '!': '', ' ': ' ', '&': '&', '%': '%', '$': '$', '#': '#', _: '_',
  '{': '{', '}': '}', '|': '‖',
};
const FUNCTIONS = new Set(`sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh coth log ln lg exp lim limsup liminf max min sup inf det dim
  gcd lcm deg ker hom arg mod Pr`.split(/\s+/));
const BIGOPS = new Set(['sum', 'prod', 'coprod', 'int', 'iint', 'iiint', 'oint', 'lim', 'limsup', 'liminf', 'max', 'min', 'sup', 'inf', 'bigcup', 'bigcap']);
SYMBOLS.bigcup = '⋃';
SYMBOLS.bigcap = '⋂';
const BLACKBOARD = { R: 'ℝ', N: 'ℕ', Z: 'ℤ', Q: 'ℚ', C: 'ℂ', P: 'ℙ', H: 'ℍ', E: '𝔼' };
const ACCENTS = { hat: '̂', widehat: '̂', bar: '̄', overline: '̅', tilde: '̃', widetilde: '̃', dot: '̇', ddot: '̈', vec: '⃗', check: '̌', acute: '́', grave: '̀', breve: '̆' };
const DELIMS = { '(': '(', ')': ')', '[': '[', ']': ']', '.': '', '|': '|', '/': '/', '<': '⟨', '>': '⟩' };
const ENV_DELIMS = {
  pmatrix: ['(', ')'], bmatrix: ['[', ']'], Bmatrix: ['{', '}'], vmatrix: ['|', '|'], Vmatrix: ['‖', '‖'], matrix: ['', ''],
  smallmatrix: ['', ''], cases: ['{', ''], array: ['', ''], aligned: ['', ''], align: ['', ''], gather: ['', ''], split: ['', ''],
  'align*': ['', ''], 'gather*': ['', ''], equation: ['', ''], 'equation*': ['', ''],
};

function tokenize(src) {
  const toks = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === '\\') {
      const m = /^\\([A-Za-z]+\*?|.)/s.exec(src.slice(i));
      if (!m) { i++; continue; }
      toks.push({ t: 'cmd', v: m[1] });
      i += m[0].length;
    } else if ('{}^_&'.includes(c)) {
      toks.push({ t: c });
      i++;
    } else if (/\s/.test(c)) {
      i++;
    } else {
      toks.push({ t: 'ch', v: c });
      i++;
    }
  }
  return toks;
}

class Parser {
  constructor(src) { this.toks = tokenize(src); this.i = 0; }
  peek() { return this.toks[this.i]; }

  // Parse a sequence of atoms until a terminator ('}' , '&', row break or end).
  parseSeq(stop) {
    let html = '';
    while (this.i < this.toks.length) {
      const t = this.peek();
      if (t.t === '}' || (stop && stop(t))) break;
      html += this.parseAtomWithScripts();
    }
    return html;
  }

  parseAtomWithScripts() {
    const base = this.parseAtom();
    let sub = null;
    let sup = null;
    while (this.peek() && (this.peek().t === '^' || this.peek().t === '_')) {
      const kind = this.toks[this.i++].t;
      const arg = this.parseArg();
      if (kind === '^') sup = arg; else sub = arg;
    }
    if (sub === null && sup === null) return base.html;
    if (base.bigop && (sub !== null || sup !== null)) {
      return `<span class="m-bigop">${sup !== null ? `<span class="m-over">${sup}</span>` : ''}<span class="m-base">${base.html}</span>${sub !== null ? `<span class="m-under">${sub}</span>` : ''}</span>`;
    }
    if (sub !== null && sup !== null) {
      return `${base.html}<span class="m-scripts"><sup>${sup}</sup><sub>${sub}</sub></span>`;
    }
    return base.html + (sup !== null ? `<sup>${sup}</sup>` : `<sub>${sub}</sub>`);
  }

  // One script argument or command argument: a {group} or a single token.
  parseArg() {
    const t = this.peek();
    if (!t) return '';
    if (t.t === '{') {
      this.i++;
      const html = this.parseSeq();
      if (this.peek() && this.peek().t === '}') this.i++;
      return html;
    }
    const a = this.parseAtom();
    return a.html;
  }

  parseRawGroup() {
    const t = this.peek();
    if (!t || t.t !== '{') return '';
    this.i++;
    let depth = 1;
    let out = '';
    while (this.i < this.toks.length) {
      const k = this.toks[this.i++];
      if (k.t === '{') depth++;
      if (k.t === '}' && --depth === 0) break;
      out += k.t === 'cmd' ? (SYMBOLS[k.v] ?? (k.v.length === 1 ? k.v : ' ')) : (k.v ?? k.t);
    }
    return out;
  }

  parseAtom() {
    const t = this.toks[this.i++];
    if (!t) return { html: '' };
    switch (t.t) {
      case '{': {
        const html = this.parseSeq();
        if (this.peek() && this.peek().t === '}') this.i++;
        return { html: `<span class="m-group">${html}</span>` };
      }
      case '&': return { html: '' };
      case '^': case '_': return { html: '' };
      case 'ch': return { html: this.char(t.v) };
      default: return this.command(t.v);
    }
  }

  char(c) {
    if (/[A-Za-z]/.test(c)) return `<i>${c}</i>`;
    if (c === '-') return '−';
    if (c === '*') return '∗';
    if (c === "'") return '′';
    if (c === '~') return ' ';
    return esc(c);
  }

  command(name) {
    switch (name) {
      case 'frac': case 'dfrac': case 'tfrac': case 'cfrac': {
        const a = this.parseArg();
        const b = this.parseArg();
        return { html: `<span class="m-frac"><span class="m-num">${a}</span><span class="m-den">${b}</span></span>` };
      }
      case 'binom': case 'dbinom': case 'tbinom': {
        const a = this.parseArg();
        const b = this.parseArg();
        return { html: `<span class="m-delim">(</span><span class="m-frac m-binom"><span class="m-num">${a}</span><span class="m-den">${b}</span></span><span class="m-delim">)</span>` };
      }
      case 'sqrt': {
        let idx = '';
        if (this.peek() && this.peek().t === 'ch' && this.peek().v === '[') {
          this.i++;
          while (this.peek() && !(this.peek().t === 'ch' && this.peek().v === ']')) idx += this.parseAtom().html;
          this.i++;
        }
        const body = this.parseArg();
        return { html: `${idx ? `<sup class="m-rootidx">${idx}</sup>` : ''}<span class="m-sqrt"><span class="m-radical">√</span><span class="m-radicand">${body}</span></span>` };
      }
      case 'text': case 'mathrm': case 'textrm': case 'operatorname': case 'mbox': case 'textit': case 'mathit': case 'textbf': case 'mathbf': case 'boldsymbol': case 'bm': case 'mathsf': case 'mathtt': case 'mathcal': case 'mathscr': case 'mathfrak': {
        const raw = name.startsWith('text') || name === 'mbox' || name === 'operatorname';
        let cls = 'm-rm';
        if (name === 'textbf' || name === 'mathbf' || name === 'boldsymbol' || name === 'bm') cls = 'm-bf';
        else if (name === 'textit' || name === 'mathit') cls = 'm-it';
        else if (name === 'mathcal' || name === 'mathscr') cls = 'm-cal';
        if (raw) return { html: `<span class="${cls}">${esc(this.parseRawGroup() || '')}</span>` };
        return { html: `<span class="${cls}">${this.parseArg()}</span>` };
      }
      case 'mathbb': {
        const raw = this.parseRawGroup();
        return { html: [...raw].map((c) => BLACKBOARD[c] || c).join('') };
      }
      case 'left': case 'right': case 'big': case 'Big': case 'bigg': case 'Bigg': case 'bigl': case 'bigr': case 'Bigl': case 'Bigr': {
        const t = this.toks[this.i++];
        if (!t) return { html: '' };
        let d = '';
        if (t.t === 'ch') d = DELIMS[t.v] ?? t.v;
        else if (t.t === 'cmd') d = SYMBOLS[t.v] ?? DELIMS[t.v] ?? '';
        else if (t.t === '{') d = '{';
        else if (t.t === '}') d = '}';
        return { html: d ? `<span class="m-delim">${esc(d)}</span>` : '' };
      }
      case 'overline': case 'underline': case 'bar': {
        const body = this.parseArg();
        return { html: `<span class="${name === 'underline' ? 'm-under-line' : 'm-over-line'}">${body}</span>` };
      }
      case 'overbrace': case 'underbrace': case 'cancel': case 'boxed': case 'phantom': case 'displaystyle': case 'textstyle': case 'scriptstyle': case 'limits': case 'nolimits': case 'mathop': case 'mathbin': case 'mathrel': {
        if (['displaystyle', 'textstyle', 'scriptstyle', 'limits', 'nolimits', 'phantom'].includes(name)) return { html: '' };
        return { html: this.parseArg() };
      }
      case 'begin': return { html: this.environment() };
      case 'end': this.parseRawGroup(); return { html: '' };
      case '\\': case 'newline': return { html: '<br>' };
      case 'not': {
        const next = this.parseAtom();
        return { html: next.html + '̸' };
      }
      case 'label': case 'tag': case 'nonumber': case 'notag': case 'hline': case 'hfill': this.parseRawGroup(); return { html: '' };
      default:
    }
    if (ACCENTS[name]) {
      const body = this.parseArg();
      return { html: `<span class="m-accent">${body}${ACCENTS[name]}</span>` };
    }
    if (FUNCTIONS.has(name)) return { html: `<span class="m-fn">${name}</span>`, bigop: BIGOPS.has(name) };
    if (name in SYMBOLS) return { html: esc(SYMBOLS[name]), bigop: BIGOPS.has(name) };
    return { html: `<span class="m-unknown">\\${esc(name)}</span>` };
  }

  environment() {
    const env = this.parseRawGroup();
    const [open, close] = ENV_DELIMS[env] || ['', ''];
    if (env === 'array' && this.peek() && this.peek().t === '{') this.parseRawGroup();
    const rows = [];
    let row = [];
    let cell = '';
    const flushCell = () => { row.push(cell); cell = ''; };
    const isEnd = (t) => t.t === 'cmd' && t.v === 'end';
    while (this.i < this.toks.length) {
      const t = this.peek();
      if (isEnd(t)) { this.i++; this.parseRawGroup(); break; }
      if (t.t === '&') { this.i++; flushCell(); continue; }
      if (t.t === 'cmd' && t.v === '\\') { this.i++; flushCell(); rows.push(row); row = []; continue; }
      if (t.t === '}') { this.i++; continue; }
      cell += this.parseAtomWithScripts();
    }
    flushCell();
    if (row.some((c) => c.trim())) rows.push(row);
    const aligned = /align|split|cases|gather|array/.test(env);
    const table = `<span class="m-matrix${aligned ? ' m-aligned' : ''}">${rows.map((r) => `<span class="m-row">${r.map((c) => `<span class="m-cell">${c}</span>`).join('')}</span>`).join('')}</span>`;
    return (open ? `<span class="m-delim m-tall">${open}</span>` : '') + table + (close ? `<span class="m-delim m-tall">${close}</span>` : '');
  }
}

export function texToHtml(tex, display = false) {
  let body;
  try {
    const p = new Parser(tex);
    body = '';
    while (p.i < p.toks.length) {
      body += p.parseSeq();
      // A stray closing brace stops parseSeq without being consumed.
      if (p.i < p.toks.length) p.i++;
    }
  } catch {
    return `<code class="math-raw">${esc(tex)}</code>`;
  }
  return `<span class="${display ? 'math math-display' : 'math'}">${body}</span>`;
}
