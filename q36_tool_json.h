#ifndef Q36_TOOL_JSON_H
#define Q36_TOOL_JSON_H

/* Shared frontend JSON reader and tool framing. The including frontend
 * supplies its checked allocation helpers. */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

static inline void buf_reserve(buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) abort();
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) {
            cap = need;
            break;
        }
        cap *= 2;
    }
    b->ptr = xrealloc(b->ptr, cap);
    b->cap = cap;
}

static inline void buf_append(buf *b, const void *p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static inline void buf_putc(buf *b, char c) {
    buf_append(b, &c, 1);
}

static inline void buf_puts(buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}

static inline void buf_printf(buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) abort();
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static inline char *buf_take(buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static inline void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static inline void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static inline bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static inline int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static inline void utf8_put(buf *b, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp <= 0x7f) {
        buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static inline bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static inline bool json_string(const char **p, char **out) {
    *out = NULL;
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            buf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                const char *low_start = *p;
                if (json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                } else {
                    *p = low_start;
                    cp = 0xfffd;
                }
            }
            utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static inline bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p || !isfinite(v)) return false;
    *p = end;
    *out = v;
    return true;
}

/* The request parser only understands the API fields we use and skips the
 * rest.  Skipping is recursive because JSON values nest, so keep an explicit
 * ceiling: without it, a useless ignored field like {"x":[[[...]]]} can spend
 * the whole C stack before the request is rejected. */
#define JSON_MAX_NESTING 256

static inline bool json_skip_value_depth(const char **p, int depth);

static inline bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') {
        (*p)++;
        return true;
    }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static inline bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static inline bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

static inline bool json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

static inline bool json_raw_value(const char **p, char **out) {
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    size_t n = (size_t)(*p - start);
    char *s = xmalloc(n + 1);
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}

static inline char *json_minify_raw_value(const char *json) {
    const char *p = json ? json : "null";
    json_ws(&p);
    const char *start = p;
    if (!json_skip_value(&p)) return xstrdup(json ? json : "null");
    const char *end = p;

    buf b = {0};
    bool in_string = false;
    bool escape = false;
    for (const char *s = start; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (in_string) {
            buf_putc(&b, (char)c);
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
        } else if (c == '"') {
            in_string = true;
            buf_putc(&b, (char)c);
        } else if (!isspace(c)) {
            buf_putc(&b, (char)c);
        }
    }
    return buf_take(&b);
}

static inline bool json_raw_value_is_complete(const char *json) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (!json_skip_value(&p)) return false;
    json_ws(&p);
    return *p == '\0';
}

typedef struct {
    char *key;
    char *value;
    bool is_string;
    bool used;
} json_arg;

typedef struct {
    json_arg *v;
    int len;
    int cap;
} json_args;

static inline void json_args_free(json_args *args) {
    for (int i = 0; i < args->len; i++) {
        free(args->v[i].key);
        free(args->v[i].value);
    }
    free(args->v);
    memset(args, 0, sizeof(*args));
}

static inline void json_args_push(json_args *args, json_arg arg) {
    if (args->len == args->cap) {
        args->cap = args->cap ? args->cap * 2 : 8;
        args->v = xrealloc(args->v, (size_t)args->cap * sizeof(args->v[0]));
    }
    args->v[args->len++] = arg;
}

static inline int json_args_find_unused(json_args *args, const char *key) {
    if (!key) return -1;
    for (int i = 0; i < args->len; i++) {
        if (!args->v[i].used && args->v[i].key && !strcmp(args->v[i].key, key)) return i;
    }
    return -1;
}

static inline bool json_args_parse(const char *json, json_args *args) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        bool is_string = false;
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') goto bad;
        p++;
        json_ws(&p);
        if (*p == '"') {
            is_string = true;
            if (!json_string(&p, &value)) goto bad;
        } else {
            char *raw = NULL;
            if (!json_raw_value(&p, &raw)) goto bad;
            value = json_minify_raw_value(raw);
            free(raw);
        }

        json_arg arg = {.key = key, .value = value, .is_string = is_string};
        json_args_push(args, arg);
        key = value = NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
        continue;
bad:
        free(key);
        free(value);
        json_args_free(args);
        return false;
    }
    if (*p != '}') {
        json_args_free(args);
        return false;
    }
    return true;
}

static inline void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

static inline void append_qwen_parameter(buf *b, const json_arg *arg) {
    buf_puts(b, "<parameter=");
    buf_puts(b, arg->key ? arg->key : "");
    buf_puts(b, ">\n");
    for (const char *p = arg->value ? arg->value : ""; *p; p++) {
        if (q36_tool_text_needs_escape(p, "</parameter>"))
            buf_puts(b, *p == '<' ? "&lt;" : "&amp;");
        else buf_putc(b, *p);
    }
    buf_puts(b, "\n</parameter>\n");
}

/* The agent and HTTP APIs consume the same tool events. Translate only a
 * completed native invocation, keeping the sampled tokens intact for replay. */
static inline char *inkling_call_xml(const char *header, const char *json) {
    json_args call = {0}, args = {0};
    buf out = {0};
    if (!header || !*header || strpbrk(header, "<>= \t\r\n") ||
        !json_raw_value_is_complete(json) || !json_args_parse(json, &call)) goto done;
    int name = json_args_find_unused(&call, "name"), params = json_args_find_unused(&call, "args");
    if (name < 0 || params < 0 || call.len != 2 || !call.v[name].is_string ||
        strcmp(header, call.v[name].value) || call.v[params].is_string ||
        !json_args_parse(call.v[params].value, &args)) goto done;
    for (int i = 0; i < args.len; i++)
        if (!*args.v[i].key || strpbrk(args.v[i].key, "<>= \t\r\n")) goto done;
    buf_printf(&out, "<tool_call>\n<function=%s>\n", header);
    for (int i = 0; i < args.len; i++) append_qwen_parameter(&out, &args.v[i]);
    buf_puts(&out, "</function>\n</tool_call>");
done:
    json_args_free(&call);
    json_args_free(&args);
    return out.ptr;
}

/* Read the existing function definitions from either frontend. */
static inline void inkling_append_tools(q36_engine *e, q36_tokens *tokens, const char *schemas) {
    if (!schemas || !*schemas) return;
    buf list = {0};
    const char *p = schemas;
    json_ws(&p);
    while (*p == '{') {
        char *raw = NULL;
        json_args args = {0};
        if (!json_raw_value(&p, &raw)) break;
        json_args_parse(raw, &args);
        free(raw);
        int fn = json_args_find_unused(&args, "function");
        if (fn >= 0) {
            raw = xstrdup(args.v[fn].value);
            json_args_free(&args);
            json_args_parse(raw, &args);
            free(raw);
        }
        int name = json_args_find_unused(&args, "name");
        int desc = json_args_find_unused(&args, "description");
        int params = json_args_find_unused(&args, "parameters");
        if (name >= 0 && args.v[name].is_string) {
            if (list.len) buf_putc(&list, ',');
            buf_puts(&list, "{\"description\":");
            json_escape(&list, desc >= 0 ? args.v[desc].value : "");
            buf_puts(&list, ",\"name\":");
            json_escape(&list, args.v[name].value);
            buf_puts(&list, ",\"parameters\":");
            buf_puts(&list, params >= 0 ? args.v[params].value : "{}");
            buf_puts(&list, ",\"type\":\"function\"}");
        }
        json_args_free(&args);
        json_ws(&p);
    }
    if (list.len) {
        q36_tokenize_rendered_chat(e, "<|message_system|>tool_declare<|content_xml|>[", tokens);
        q36_tokenize_text(e, list.ptr, tokens);
        q36_tokenize_rendered_chat(e, "]<|end_message|>", tokens);
    }
    buf_free(&list);
}

static inline void inkling_append_tool_header(q36_engine *e, q36_tokens *tokens, const char *name) {
    q36_tokenize_rendered_chat(e, "<|message_tool|>", tokens);
    q36_tokenize_text(e, name, tokens);
}

static inline void inkling_append_result(q36_engine *e, q36_tokens *tokens,
                                         const char *name, const char *content) {
    inkling_append_tool_header(e, tokens, name);
    q36_tokenize_rendered_chat(e, "<|content_text|>", tokens);
    q36_tokenize_text(e, content, tokens);
    q36_tokenize_rendered_chat(e, "<|end_message|>", tokens);
}

typedef struct inkling_tool_stream {
    buf header, body;
    bool in_header, in_body, thinking;
} inkling_tool_stream;

static inline void inkling_tool_reset(inkling_tool_stream *s) {
    buf_free(&s->header);
    buf_free(&s->body);
    s->in_header = s->in_body = false;
}

static inline char *inkling_tool_piece(inkling_tool_stream *s, q36_engine *e, int token,
                                       bool show_thinking, size_t *len) {
    char *raw = q36_token_text(e, token, len);
    char *call = NULL;
    if (!strcmp(raw, "<|message_model|>")) {
        inkling_tool_reset(s);
        s->in_header = !s->thinking;
    } else if (s->in_header && !strcmp(raw, "<|content_invoke_tool_json|>")) {
        s->in_header = false;
        s->in_body = true;
    } else if (!strcmp(raw, "<|end_message|>") && s->in_body) {
        call = inkling_call_xml(s->header.ptr, s->body.ptr);
        inkling_tool_reset(s);
        /* A malformed call goes through the existing tool-error path. */
        if (!call) call = xstrdup("<tool_call>\n</tool_call>");
    } else if (s->in_header && strncmp(raw, "<|", 2)) {
        buf_append(&s->header, raw, *len);
    } else if (s->in_body) {
        buf_append(&s->body, raw, *len);
    } else {
        s->in_header = false;
        free(raw);
        return q36_chat_token_text(e, token, show_thinking, &s->thinking, len);
    }
    free(raw);
    if (!call) call = xstrdup("");
    *len = strlen(call);
    return call;
}
#endif
