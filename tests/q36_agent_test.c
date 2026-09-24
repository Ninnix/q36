#define Q36_AGENT_TEST
#define Q36_AGENT_TEST_NO_MAIN
#include "../q36_agent.c"
#include <sys/resource.h>

static void test_tool_arg(agent_tool_call *call, const char *name, const char *value) {
    agent_tool_call_add_arg(call, name, value, strlen(value));
}

static int test_write_file(const char *path, const char *data, size_t len,
                           char *err, size_t errlen) {
    return agent_replace_file(path, data, len, NULL, 0, err, errlen);
}

static void test_atomic_file_tools(void) {
    char dir[] = "/tmp/q36-agent-files-XXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(dir) != NULL);
    char path[PATH_MAX], linkpath[PATH_MAX], err[256];
    snprintf(path, sizeof(path), "%s/file", dir);
    snprintf(linkpath, sizeof(linkpath), "%s/link", dir);
    char original[4096];
    memset(original, 'x', sizeof(original));
    AGENT_TEST_ASSERT(test_write_file(path, original, sizeof(original), err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(chmod(path, 0751) == 0);
#ifdef __APPLE__
    AGENT_TEST_ASSERT(setxattr(path, "com.q36.agent-test", "keep", 4, 0, 0) == 0);
#elif defined(__linux__)
    AGENT_TEST_ASSERT(setxattr(path, "user.q36-agent-test", "keep", 4, 0) == 0);
#endif

    pid_t child = fork();
    AGENT_TEST_ASSERT(child >= 0);
    if (child == 0) {
        signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit = {64, 64};
        if (setrlimit(RLIMIT_FSIZE, &limit)) _exit(2);
        int rc = agent_replace_file(path, original, sizeof(original),
                                     original, sizeof(original), err, sizeof(err));
        _exit(rc == -1 ? 0 : 3);
    }
    int status = 0;
    if (child > 0) waitpid(child, &status, 0);
    AGENT_TEST_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    char *data = NULL;
    size_t len = 0;
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(len == sizeof(original) && !memcmp(data, original, len));
    free(data);
    AGENT_TEST_ASSERT(agent_replace_file(path, "bad", 3, "stale", 5, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "changed") != NULL);

    AGENT_TEST_ASSERT(symlink("file", linkpath) == 0);
    AGENT_TEST_ASSERT(test_write_file(linkpath, "new", 3, err, sizeof(err)) == 0);
    struct stat st;
    AGENT_TEST_ASSERT(lstat(linkpath, &st) == 0 && S_ISLNK(st.st_mode));
    AGENT_TEST_ASSERT(stat(path, &st) == 0 && (st.st_mode & 0777) == 0751);
    AGENT_TEST_ASSERT(st.st_uid == getuid());
#ifdef __APPLE__
    char attribute[16];
    AGENT_TEST_ASSERT(getxattr(path, "com.q36.agent-test", attribute, sizeof(attribute), 0, 0) == 4);
    AGENT_TEST_ASSERT(!memcmp(attribute, "keep", 4));
#elif defined(__linux__)
    char attribute[16];
    AGENT_TEST_ASSERT(getxattr(path, "user.q36-agent-test", attribute, sizeof(attribute)) == 4);
    AGENT_TEST_ASSERT(!memcmp(attribute, "keep", 4));
#endif
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(len == 3 && !memcmp(data, "new", 3));
    free(data);
    unlink(linkpath);
    AGENT_TEST_ASSERT(link(path, linkpath) == 0);
    AGENT_TEST_ASSERT(test_write_file(path, "bad", 3, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "hard-linked") != NULL);
    unlink(linkpath);
    unlink(path);
    /* Failed replacements must not leave temporary files behind. */
    AGENT_TEST_ASSERT(rmdir(dir) == 0);

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = true;
    AGENT_TEST_ASSERT(agent_edit_find_old_span("literal [upto] text", 19,
                         "[upto]", false, &match, &match_len, &anchored, err, sizeof(err)));
    AGENT_TEST_ASSERT(!anchored && match_len == 6 && !memcmp(match, "[upto]", 6));
}

static void test_streaming_file_tools(void) {
    char path[] = "/tmp/q36-agent-read-XXXXXX";
    int fd = mkstemp(path);
    AGENT_TEST_ASSERT(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    /* Larger than the old whole-file cap, with matches at both ends. */
    fputs("needle first\r\n", fp);
    for (int i = 0; i < 1024 * 1024; i++) fputs("0123456789abcdef\n", fp);
    fputs("needle last\r", fp);
    fclose(fp);
    agent_worker w = {0};
    char *text = agent_read_range(&w, path, 1, 1, false, false, true);
    AGENT_TEST_ASSERT(strstr(text, "needle first") && w.more_valid);
    AGENT_TEST_ASSERT(w.more_next_line == 2 && w.more_byte_offset == 14);
    free(text);
    agent_tool_call more = {0};
    test_tool_arg(&more, "count", "1");
    text = agent_tool_more(&w, &more);
    AGENT_TEST_ASSERT(strstr(text, "2 0123456789abcdef"));
    free(text);
    agent_tool_call_free(&more);

    agent_tool_call call = {0};
    test_tool_arg(&call, "path", path);
    test_tool_arg(&call, "query", "needle");
    char linkpath[PATH_MAX];
    snprintf(linkpath, sizeof(linkpath), "%s-link", path);
    AGENT_TEST_ASSERT(symlink(path, linkpath) == 0);
    agent_tool_call linked = {0};
    test_tool_arg(&linked, "path", linkpath);
    test_tool_arg(&linked, "query", "needle");
    text = agent_tool_search(&w, &linked);
    AGENT_TEST_ASSERT(strstr(text, "2 matches") && strstr(text, "needle last"));
    free(text);
    agent_tool_call_free(&linked);
    unlink(linkpath);
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "2 matches") && strstr(text, "needle last"));
    free(text);
    test_tool_arg(&call, "mode", "regexp");
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:"));
    free(text);
    agent_tool_call_free(&call);

    fp = fopen(path, "wb");
    for (int i = 0; i < 256 * 1024; i++) fputc('x', fp);
    fclose(fp);
    size_t total = 0;
    text = agent_read_range(&w, path, 1, 1, false, true, true);
    do {
        size_t n = strspn(text, "x");
        total += n;
        AGENT_TEST_ASSERT(n > 0 && n < AGENT_TOOL_MAX_BYTES);
        if (w.more_valid) AGENT_TEST_ASSERT(strstr(text, "Read truncated") != NULL);
        free(text);
        if (!w.more_valid) break;
        text = agent_tool_more(&w, &more);
    } while (total < 512 * 1024);
    AGENT_TEST_ASSERT(total == 256 * 1024 && !w.more_valid);
    text = agent_read_range(&w, path, 1, INT_MAX, true, true, true);
    AGENT_TEST_ASSERT(strstr(text, "Tool error: whole read") && !w.more_valid);
    free(text);
    fp = fopen(path, "wb");
    for (int i = 0; i < 256 * 1024; i++) fputc(0x80, fp);
    fclose(fp);
    text = agent_read_range(&w, path, 1, 1, false, true, true);
    AGENT_TEST_ASSERT(strlen(text) < AGENT_TOOL_MAX_BYTES && w.more_valid);
    free(text);
    unlink(path);
    AGENT_TEST_ASSERT(mkfifo(path, 0600) == 0);
    double started = now_sec();
    text = agent_read_range(&w, path, 1, 1, false, false, true);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:") && now_sec() - started < 0.5);
    free(text);
    unlink(path);
    test_tool_arg(&call, "path", path);
    test_tool_arg(&call, "query", "needle");
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:"));
    free(text);
    agent_tool_call_free(&call);

    agent_buf b = {0};
    char *large = xmalloc(200000);
    memset(large, 'q', 200000);
    agent_buf_append(&b, large, 200000);
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(strlen(text) == 200000);
    free(text);
    b.limit = 100;
    agent_buf_append(&b, large, 200000);
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(strstr(text, "Output truncated") != NULL);
    free(large);
    free(text);
    b.limit = 3;
    agent_buf_puts(&b, "a\xe4\xb8\xad" "b");
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(!strncmp(text, "a\n[Output truncated", 19));
    free(text);
}

static void test_background_jobs(void) {
    agent_worker w = {0};
    pthread_mutex_init(&w.mu, NULL);
    w.wake_fd[0] = w.wake_fd[1] = -1;
    char err[256], marker[] = "/tmp/q36-agent-deadline-XXXXXX";
    int fd = mkstemp(marker);
    close(fd);
    unlink(marker);
    char cmd[PATH_MAX + 128];
    snprintf(cmd, sizeof(cmd), "sleep 2; printf late > %s", marker);
    agent_bash_job *job = agent_bash_start(&w, cmd, 1, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    /* Deliberately no status polling: this stands in for model generation. */
    usleep(2400000);
    AGENT_TEST_ASSERT(access(marker, F_OK) != 0);
    bool finished = false;
    char *obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "timed_out=1"));
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "(sleep 0.1; printf descendant-output) &", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    usleep(350000);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "descendant-output"));
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "head -c 2097152 /dev/zero | tr '\\000' x", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    usleep(900000);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "exit_status=0"));
    AGENT_TEST_ASSERT(job->bytes == 2097152);
    free(obs);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(strlen(obs) < AGENT_BASH_TAIL_BYTES + 2048);
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "sleep 0.3; printf completed", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    char output_path[PATH_MAX];
    snprintf(output_path, sizeof(output_path), "%s", job->path);
    agent_tool_call call = {.name = xstrdup("bash_status")};
    char id[32];
    snprintf(id, sizeof(id), "%d", job->id);
    test_tool_arg(&call, "job", id);
    test_tool_arg(&call, "refresh_sec", "1");
    double start = now_sec();
    obs = agent_execute_tool_call(&w, &call);
    AGENT_TEST_ASSERT(now_sec() - start >= 0.2);
    AGENT_TEST_ASSERT(strstr(obs, "status=done") && strstr(obs, "completed"));
    AGENT_TEST_ASSERT(w.bash_jobs == NULL);
    free(obs);
    agent_tool_call_free(&call);
    unlink(output_path);

    /* Two monitors must make progress together, and stopping one must neither
     * block shutdown nor kill the other job's process group. */
    job = agent_bash_start(&w, "sleep 30", 60, err, sizeof(err));
    agent_bash_job *other = agent_bash_start(&w, "sleep 0.2; printf independent", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job && other);
    if (!job || !other) goto done;
    start = now_sec();
    agent_bash_signal(job, SIGKILL);
    unlink(job->path);
    agent_bash_remove_job(&w, job);
    AGENT_TEST_ASSERT(now_sec() - start < 2);
    while (agent_bash_is_running(other) && now_sec() - start < 3) usleep(10000);
    obs = agent_bash_observation(other, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "exit_status=0") && strstr(obs, "independent"));
    free(obs);
    unlink(other->path);
    agent_bash_remove_job(&w, other);
done:
    agent_bash_jobs_free(&w);
    unlink(marker);
    free(w.out);
    pthread_mutex_destroy(&w.mu);
}

static void test_shell_terminal_controls(void) {
    const char malicious[] = "before\x1b[2Jafter\x1b[H!\x1b]52;c;secret\a"
                             "\x1bPdata\x1b\\\x1b[31mred\x1b[0m\b\n";
    char *safe = agent_terminal_safe_text(malicious, sizeof(malicious) - 1);
    AGENT_TEST_ASSERT(!strcmp(safe, "beforeafter!\x1b[31mred\x1b[0m\\x08\n"));
    free(safe);
    const char c1[] = "\xe4\xb8\xad\xc2\x9b" "2J\x9b" "2J";
    safe = agent_terminal_safe_text(c1, sizeof(c1) - 1);
    AGENT_TEST_ASSERT(!strcmp(safe, "\xe4\xb8\xad\\xc2\\x9b2J\\x9b2J"));
    free(safe);
    for (size_t i = 0; i < sizeof(malicious); i++) {
        safe = agent_terminal_safe_text(malicious, i);
        AGENT_TEST_ASSERT(!strstr(safe, "\x1b[2J") && !strstr(safe, "\x1b]52"));
        free(safe);
    }
}

static const char *test_output_dir;

static void test_fixture(const char *name, const char *data, size_t len) {
    if (!test_output_dir) return;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", test_output_dir, name);
    FILE *fp = fopen(path, "wb");
    AGENT_TEST_ASSERT(fp != NULL);
    if (!fp) return;
    AGENT_TEST_ASSERT(fwrite(data, 1, len, fp) == len);
    AGENT_TEST_ASSERT(fclose(fp) == 0);
}

static void test_completion(const char *text, linenoiseCompletions *completions) {
    (void)text;
    linenoiseAddCompletion(completions, "example");
}

static void test_fragmented_terminal_input(void) {
    int input[2];
    AGENT_TEST_ASSERT(pipe(input) == 0);
    fcntl(input[0], F_SETFL, O_NONBLOCK);
    FILE *sink = tmpfile();
    AGENT_TEST_ASSERT(sink != NULL);
    if (!sink) { close(input[0]); close(input[1]); return; }
    setenv("LINENOISE_ASSUME_TTY", "1", 1);
    struct linenoiseState l = {0};
    char buffer[1024] = "";
    l.ifd = input[0]; l.ofd = fileno(sink);
    l.buf = buffer; l.buflen = sizeof(buffer) - 1;
    l.cols = 80; l.prompt = "";
    const char *samples[] = {"\xc3\xa9", "\xe4\xb8\xad", "\xf0\x9f\x98\x80"};
    for (size_t s = 0; s < sizeof(samples)/sizeof(samples[0]); s++) {
        const char *sample = samples[s];
        for (size_t split = 1; split < strlen(sample); split++) {
            linenoiseEditClear(&l);
            for (size_t i = 0; i < split; i++)
                AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sample[i]) == linenoiseEditMore);
            AGENT_TEST_ASSERT(l.len == 0);
            AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
            AGENT_TEST_ASSERT(l.len == 0);
            for (size_t i = split; i < strlen(sample); i++)
                AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sample[i]) == linenoiseEditMore);
            AGENT_TEST_ASSERT(!strcmp(l.buf, sample));
        }
    }
    linenoiseEditClear(&l);
    const char sequence[] = "ab\x1b[DZ\x1b[3~\x1b[H!";
    for (size_t i = 0; i < sizeof(sequence) - 1; i++) {
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sequence[i]) == linenoiseEditMore);
        AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
    }
    AGENT_TEST_ASSERT(!strcmp(l.buf, "!aZ"));
    linenoiseEditClear(&l);
    const char paste[] = "\x1b[200~one\r\n\xe4\xb8\xad\nthree\x1b[201~";
    for (size_t i = 0; i < sizeof(paste) - 1; i++) {
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, paste[i]) == linenoiseEditMore);
        AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
        if (i < sizeof(paste) - 2) AGENT_TEST_ASSERT(l.len == 0);
    }
    AGENT_TEST_ASSERT(!strcmp(l.buf, "one\n\xe4\xb8\xad\nthree"));
    linenoiseEditClear(&l);
    linenoiseEditFeedByte(&l, '\xe4');
    linenoiseEditFeedByte(&l, 'X');
    AGENT_TEST_ASSERT(!strcmp(l.buf, "\xef\xbf\xbdX"));
    linenoiseEditClear(&l);
    const char invalid_paste[] = "\x1b[200~bad\xe4\x1b[201~";
    for (size_t i = 0; i < sizeof(invalid_paste) - 1; i++)
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, invalid_paste[i]) == linenoiseEditMore);
    AGENT_TEST_ASSERT(l.len == 0 && !l.paste_active);
    linenoiseSetCompletionCallback(test_completion);
    linenoiseEditFeedByte(&l, 'e');
    linenoiseEditFeedByte(&l, '\t');
    AGENT_TEST_ASSERT(l.in_completion);
    linenoiseEditFeedByte(&l, '\x1b');
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "e"));
    linenoiseEditFeedByte(&l, '[');
    linenoiseEditFeedByte(&l, 'D');
    AGENT_TEST_ASSERT(l.pos == 0);
    linenoiseEditClear(&l);
    linenoiseEditFeedByte(&l, '\t');
    const char unicode[] = "\xe4\xb8\xad";
    for (size_t i = 0; i < sizeof(unicode) - 1; i++) linenoiseEditFeedByte(&l, unicode[i]);
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "example\xe4\xb8\xad"));
    linenoiseEditClear(&l);
    l.buflen = 3;
    linenoiseEditFeedByte(&l, 'e');
    linenoiseEditFeedByte(&l, '\t');
    linenoiseEditFeedByte(&l, ' ');
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "e") && l.len == 1);
    l.buflen = sizeof(buffer) - 1;
    linenoiseSetCompletionCallback(NULL);
    free(l.queued_input);
    free(l.paste_buf);
    close(input[0]); close(input[1]); fclose(sink);
    unsetenv("LINENOISE_ASSUME_TTY");
}

static void test_markdown_literals(void) {
    const char *input[] = {"Use *.c files.", "The literal is \\*.", "An unmatched `tick",
                          "**bold** and *italic* and `code`.", "``a ` b``", "*unclosed",
                          "* list item\n", "trailing \\", "**unclosed", "`a``", "\\`literal\\`",
                          "> **Hint:** Check `errno`.\n"};
    const char *expected[] = {"Use *.c files.", "The literal is *.", "An unmatched `tick",
                             "bold and italic and code.", "a ` b", "*unclosed",
                             "* list item\n", "trailing \\", "**unclosed", "`a``", "`literal`",
                             "> Hint: Check errno.\n"};
    for (size_t i = 0; i < sizeof(input)/sizeof(input[0]); i++) {
        agent_tail_capture capture = {.cap = 16384};
        agent_token_renderer r = {.capture = &capture, .format_markdown = true};
        for (size_t j = 0; j < strlen(input[i]); j++) renderer_markdown_feed(&r, input[i][j]);
        renderer_markdown_finish(&r);
        renderer_flush_utf8(&r);
        size_t len;
        char *out = agent_tail_capture_take(&capture, &len);
        AGENT_TEST_ASSERT(!strcmp(out, expected[i]));
        if (strcmp(out, expected[i])) fprintf(stderr, "markdown: %s => %s\n", input[i], out);
        free(out);
    }
    agent_tail_capture capture = {.cap = 20000};
    agent_token_renderer r = {.capture = &capture, .format_markdown = true};
    renderer_markdown_feed(&r, '*');
    for (int i = 0; i < 8192; i++) renderer_markdown_feed(&r, 'x');
    renderer_markdown_finish(&r);
    size_t len;
    char *out = agent_tail_capture_take(&capture, &len);
    AGENT_TEST_ASSERT(len == 8193 && out[0] == '*');
    free(out);
}

static void test_unicode_output_and_footer(void) {
    agent_editor ed = {0};
    ed.edit.cols = 80;
    char ascii[78];
    memset(ascii, 'a', sizeof(ascii));
    editor_note_output(&ed, ascii, sizeof(ascii));
    editor_note_output(&ed, "\xe4", 1);
    AGENT_TEST_ASSERT(ed.output_col == 78 && ed.output_utf8_len == 1);
    editor_note_output(&ed, "\xb8\xad", 2);
    AGENT_TEST_ASSERT(ed.output_col == 0 && ed.output_pending_wrap);
    editor_note_output(&ed, "\xcc\x81", 2);
    AGENT_TEST_ASSERT(ed.output_col == 0 && ed.output_pending_wrap);
    editor_note_output(&ed, "x", 1);
    AGENT_TEST_ASSERT(ed.output_col == 1 && !ed.output_pending_wrap);
    editor_note_output(&ed, "\r", 1);
    AGENT_TEST_ASSERT(ed.output_col == 0 && !ed.output_pending_wrap);
    const char family[] = "\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb";
    for (size_t i = 0; i < sizeof(family) - 1; i++) editor_note_output(&ed, family + i, 1);
    AGENT_TEST_ASSERT(ed.output_col == 2);
    int width;
    AGENT_TEST_ASSERT(linenoiseNextGrapheme(family, sizeof(family) - 1, &width) == sizeof(family) - 1 && width == 2);
    const char flag[] = "\xf0\x9f\x87\xae\xf0\x9f\x87\xb9";
    AGENT_TEST_ASSERT(linenoiseNextGrapheme(flag, sizeof(flag) - 1, &width) == sizeof(flag) - 1 && width == 2);
    editor_note_output(&ed, flag, sizeof(flag) - 1);
    AGENT_TEST_ASSERT(ed.output_col == 4);
    editor_note_output(&ed, "\x1b[", 2);
    AGENT_TEST_ASSERT(ed.output_escape == 2);
    editor_note_output(&ed, "31mX", 4);
    AGENT_TEST_ASSERT(ed.output_col == 5 && !ed.output_escape);

    agent_prompt_queue q = {0};
    char queued[181];
    for (int i = 0; i < 60; i++) memcpy(queued + i * 3, "\xe4\xb8\xad", 3);
    queued[180] = 0;
    agent_prompt_queue_push(&q, queued);
    agent_status st = {0};
    char footer[4096];
    build_footer_text(&st, &q, 40, footer, sizeof(footer));
    test_fixture("queue-footer.txt", footer, strlen(footer));
    for (size_t pos = 0; pos < strlen(footer);) {
        uint32_t cp;
        size_t n = linenoiseUtf8Decode(footer + pos, strlen(footer) - pos, &cp);
        AGENT_TEST_ASSERT(n && cp != 0xfffd);
        if (!n) break;
        pos += n;
    }
    agent_prompt_queue_free(&q);
}

static void test_footer_only_updates(void) {
    FILE *sink = tmpfile();
    AGENT_TEST_ASSERT(sink != NULL);
    if (!sink) return;
    int saved = dup(STDOUT_FILENO);
    dup2(fileno(sink), STDOUT_FILENO);
    agent_editor ed = {.active = true, .scroll_region = true, .term_rows = 24,
                       .term_cols = 80, .output_bottom = 22, .prompt_row = 23};
    snprintf(ed.prompt, sizeof(ed.prompt), "q36-agent> ");
    snprintf(ed.status, sizeof(ed.status), "generation 0");
    char buffer[] = "draft";
    ed.edit = (struct linenoiseState){.ifd = -1, .ofd = STDOUT_FILENO,
        .buf = buffer, .buflen = sizeof(buffer), .len = 5, .pos = 5, .oldpos = 5,
        .prompt = ed.prompt, .plen = strlen(ed.prompt), .cols = 80,
        .oldrows = 1, .oldstatusrows = 1, .oldrpos = 1,
        .screen_cursor_row = 23, .screen_cursor_col = 17};
    linenoiseEditSetStatus(&ed.edit, ed.status, "", "");
    const char initial[] = "\x1b[23;1Hq36-agent> draft\r\ngeneration 0\x1b[23;17H";
    write_all(STDOUT_FILENO, initial, sizeof(initial) - 1);
    editor_set_prompt_status(&ed, ed.prompt, "generation 1");
    off_t first = lseek(STDOUT_FILENO, 0, SEEK_CUR);
    editor_set_prompt_status(&ed, ed.prompt, "generation 2");
    AGENT_TEST_ASSERT(lseek(STDOUT_FILENO, 0, SEEK_CUR) == first && ed.status_dirty);
    ed.last_prompt_redraw_time -= 1;
    editor_set_prompt_status(&ed, ed.prompt, "generation 2");
    AGENT_TEST_ASSERT(!ed.status_dirty);
    editor_set_prompt_status(&ed, ed.prompt, "done");
    editor_flush_prompt_status(&ed, true);
    AGENT_TEST_ASSERT(!ed.status_dirty && !strcmp(buffer, "draft"));
    dup2(saved, STDOUT_FILENO);
    close(saved);
    fseek(sink, 0, SEEK_END);
    size_t len = (size_t)ftell(sink);
    rewind(sink);
    char *text = xmalloc(len + 1);
    AGENT_TEST_ASSERT(fread(text, 1, len, sink) == len);
    text[len] = 0;
    AGENT_TEST_ASSERT(strstr(text, "\x1b[?2026h") && !strstr(text, "\x1b[0K"));
    test_fixture("status.ansi", text, len);
    free(text);
    free(ed.edit.status); free(ed.edit.status_start); free(ed.edit.status_end);
    fclose(sink);
}

static int test_terminal_driver(void) {
    agent_editor ed = {0};
    linenoiseSetMultiLine(1);
    if (editor_start(&ed, "q36-agent> ", "ready", NULL)) return 2;
    double start = now_sec(), next = start;
    char *answer = NULL;
    unsigned tick = 0;
    while (now_sec() - start < 10 && !answer) {
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        poll(&pfd, 1, 10);
        if (pfd.revents & POLLIN) editor_read_stdin(&ed);
        while (linenoiseEditQueuedInput(&ed.edit)) {
            char *line = linenoiseEditFeed(&ed.edit);
            if (line == linenoiseEditMore) continue;
            if (line) answer = line;
            else answer = xstrdup("<input error>");
            break;
        }
        if (now_sec() >= next) {
            char status[80];
            snprintf(status, sizeof(status), "generation %u", ++tick);
            if (tick % 4 == 0) {
                const char output[] = "model output \xe4\xb8\xad\n";
                editor_write_async(&ed, output, sizeof(output) - 1, "q36-agent> ", status, false);
            } else editor_set_prompt_status(&ed, "q36-agent> ", status);
            next = now_sec() + 0.05;
        }
    }
    editor_stop(&ed);
    editor_restore_terminal_layout(&ed);
    if (!answer) return 3;
    printf("\nRESULT:");
    for (size_t i = 0; i < strlen(answer); i++) printf("%02x", (unsigned char)answer[i]);
    puts("");
    free(answer);
    return 0;
}

static char *test_hint_capture(const char *text, size_t split, bool markdown,
                               bool color, int *calls) {
    agent_tail_capture capture = {.cap = 32768};
    agent_token_renderer renderer = {.capture = &capture, .format_thinking = true,
        .format_markdown = markdown, .use_color = color, .last_output_newline = true};
    agent_qwen_tool_parser parser = {.state = AGENT_QWEN_TOOL_SEARCH};
    agent_stream_renderer stream = {.renderer = &renderer, .parser = &parser};
    size_t n = strlen(text);
    if (split <= n) {
        agent_stream_text(&stream, text, split, false);
        /* A prompt redraw resets terminal colors between generated fragments. */
        if (color && renderer.wrote_visible_output) renderer_write(&renderer, "\x1b[0m", 4);
        agent_stream_text(&stream, text + split, n - split, false);
    } else {
        for (size_t i = 0; i < n; i++) {
            agent_stream_text(&stream, text + i, 1, false);
            if (color && renderer.wrote_visible_output) renderer_write(&renderer, "\x1b[0m", 4);
        }
    }
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    AGENT_TEST_ASSERT(!renderer.md_hint && !renderer.md_hint_prefix_len && !renderer.color_open);
    if (calls) {
        *calls = (int)parser.calls.len;
        AGENT_TEST_ASSERT(parser.calls.len == 1);
        if (parser.calls.len == 1) {
            AGENT_TEST_ASSERT(!strcmp(parser.calls.v[0].name, "bash"));
            AGENT_TEST_ASSERT(parser.calls.v[0].argc == 1);
            if (parser.calls.v[0].argc == 1)
                AGENT_TEST_ASSERT(!strcmp(parser.calls.v[0].args[0].value, "printf HINT_OK"));
        }
    } else AGENT_TEST_ASSERT(parser.calls.len == 0);
    agent_qwen_tool_parser_free(&parser);
    return agent_tail_capture_take(&capture, NULL);
}

static void test_hint_rendering(void) {
    const char *badge = "\x1b[1;97;48;5;23m Hint \x1b[0m";
    const char *sample =
        "The error path is fixed.\n\n"
        "> **Hint:** Save `errno` before **cleanup**; calls can overwrite it.\n"
        "> Keep the original error for reporting.\n"
        "> Unicode stays intact: caf\xc3\xa9, \xe4\xb8\xad.\n"
        "Normal prose resumes here.\n\n"
        "```text\n> **Hint:** This is literal code.\n```\n"
        "Another normal line.\n";
    for (size_t split = 0; split <= strlen(sample) + 1; split++) {
        char *out = test_hint_capture(sample, split, true, true, NULL);
        const char *label = strstr(out, badge);
        AGENT_TEST_ASSERT(label && !strstr(label + strlen(badge), badge));
        AGENT_TEST_ASSERT(strstr(out, "\x1b[1mcleanup") || split > strlen(sample));
        if (split == strlen(sample)) test_fixture("hints.ansi", out, strlen(out));
        if (split > strlen(sample)) test_fixture("hints-fragmented.ansi", out, strlen(out));
        free(out);
    }
    const char *literal[] = {"> quoted text", "> **Hinting:** not a hint",
        "Inline > **Hint:** not an aside", "`> **Hint:** literal`",
        "\\> **Hint:** escaped", "<think>> **Hint:** hidden reasoning</think>Normal."};
    for (size_t i = 0; i < sizeof(literal) / sizeof(literal[0]); i++) {
        char *out = test_hint_capture(literal[i], strlen(literal[i]), true, true,
                                      NULL);
        AGENT_TEST_ASSERT(!strstr(out, badge));
        free(out);
    }
    const char marker[] = "> **Hint:**";
    for (size_t i = 0; i < sizeof(marker) - 1; i++) {
        char partial[sizeof(marker)];
        memcpy(partial, marker, i);
        partial[i] = 0;
        char *out = test_hint_capture(partial, i, true, true, NULL);
        AGENT_TEST_ASSERT(!strstr(out, badge) && !strncmp(out, partial, i));
        free(out);
    }
    char *out = test_hint_capture(sample, strlen(sample), false, false, NULL);
    AGENT_TEST_ASSERT(!strncmp(out, sample, strlen(sample)) && !strchr(out, '\x1b'));
    free(out);
    out = test_hint_capture("> **Hint:** Check `errno`.\n", 0, true, false, NULL);
    AGENT_TEST_ASSERT(!strcmp(out, "> Hint: Check errno.\n\n"));
    free(out);
    const char *tool = "> **Hint:** Keep shell checks reproducible.\n"
        "<tool_call><function=bash><parameter=command>printf HINT_OK"
        "</parameter></function></tool_call>";
    for (size_t split = 0; split <= strlen(tool); split++) {
        int calls = 0;
        out = test_hint_capture(tool, split, true, true, &calls);
        AGENT_TEST_ASSERT(calls == 1 && strstr(out, badge));
        free(out);
    }
}

/* Behave like sudo's terminal password reader without using real credentials
 * or requiring privilege. stdout/stderr still belong to the bash tool. */
static int password_child(void) {
    int fd = open("/dev/tty", O_RDWR);
    if (fd < 0) { puts("NO_TERMINAL"); return 1; }
    struct termios saved, mode;
    if (tcgetattr(fd, &saved) != 0) return 2;
    mode = saved;
    mode.c_lflag &= ~(ECHO | ECHONL);
    for (int attempt = 0; attempt < 3; attempt++) {
        if (tcsetattr(fd, TCSAFLUSH, &mode) != 0) return 3;
        write_all(fd, "[sudo] password for test: ", 26);
        char buf[1024] = {0};
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        tcsetattr(fd, TCSANOW, &saved);
        if (n > 0 && !strcmp(buf, "q36-test-secret\n")) {
            close(fd);
            puts("PASSWORD_OK");
            return 0;
        }
        write_all(fd, "\nSorry, try again.\n", 19);
    }
    close(fd);
    return 1;
}

static void test_worker_init(agent_worker *w, agent_config *cfg) {
    memset(w, 0, sizeof(*w));
    w->cfg = cfg;
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    if (pipe(w->wake_fd) != 0) abort();
    set_nonblock(w->wake_fd[0], true, NULL);
    set_nonblock(w->wake_fd[1], true, NULL);
}

static void test_worker_free(agent_worker *w) {
    agent_bash_jobs_free(w);
    free(w->out);
    close(w->wake_fd[0]);
    close(w->wake_fd[1]);
    pthread_mutex_destroy(&w->mu);
    pthread_cond_destroy(&w->cond);
}

static void test_worker_ownership(void) {
    agent_config cfg = {0};
    agent_worker w;
    test_worker_init(&w, &cfg);
    w.initialized = true;
    AGENT_TEST_ASSERT(worker_is_idle(&w));
    w.active = true;
    AGENT_TEST_ASSERT(!worker_is_idle(&w));
    AGENT_TEST_ASSERT(!worker_submit(&w, "next turn"));
    w.active = false;
    w.save_requested = true;
    AGENT_TEST_ASSERT(!worker_is_idle(&w));
    AGENT_TEST_ASSERT(!worker_submit(&w, "next turn"));
    worker_pause(&w);
    AGENT_TEST_ASSERT(worker_is_idle(&w));
    AGENT_TEST_ASSERT(!worker_take_save_requested(&w) && w.save_requested);
    AGENT_TEST_ASSERT(!worker_submit(&w, "next turn"));
    worker_resume(&w);
    AGENT_TEST_ASSERT(worker_take_save_requested(&w));
    w.interrupt = true;
    AGENT_TEST_ASSERT(worker_submit(&w, "next turn"));
    AGENT_TEST_ASSERT(!w.interrupt && !worker_is_idle(&w));
    free(w.cmd_text);
    test_worker_free(&w);
}

static void *test_worker_ui_wait(void *arg) {
    agent_worker *w = arg;
    if (w->cfg->gen.seed == 0) {
        free(worker_request_queued_user_drain(w));
    } else if (w->cfg->gen.seed == 1) {
        agent_password_request request = {0};
        agent_request_password(w, &request);
    } else {
        char err[160];
        agent_web_confirm(w, "Allow?", err, sizeof(err));
    }
    worker_clear_interrupt(w);
    agent_set_status(w, AGENT_WORKER_IDLE);
    while (worker_wait_for_work(w)) {
        pthread_mutex_lock(&w->mu);
        free(w->cmd_text);
        w->cmd_text = NULL;
        w->status.state = AGENT_WORKER_IDLE;
        pthread_mutex_unlock(&w->mu);
    }
    return NULL;
}

static void test_worker_pause_ui_waits(void) {
    for (int request = 0; request < 3; request++) {
        agent_config cfg = {0};
        cfg.gen.seed = request;
        agent_worker w;
        test_worker_init(&w, &cfg);
        w.initialized = true;
        w.active = true;
        w.status.state = AGENT_WORKER_GENERATING;
        pthread_t thread;
        AGENT_TEST_ASSERT(pthread_create(&thread, NULL, test_worker_ui_wait, &w) == 0);
        struct pollfd pfd = {.fd = w.wake_fd[0], .events = POLLIN};
        AGENT_TEST_ASSERT(poll(&pfd, 1, 2000) == 1);
        worker_pause(&w);
        AGENT_TEST_ASSERT(worker_is_idle(&w) && !w.interrupt);
        worker_resume(&w);
        AGENT_TEST_ASSERT(worker_submit(&w, "continue after cancelled exit"));
        worker_pause(&w);
        AGENT_TEST_ASSERT(worker_is_idle(&w));
        worker_stop(&w);
        pthread_join(thread, NULL);
        test_worker_free(&w);
    }
}

static void *password_job_thread(void *arg) {
    agent_worker *w = arg;
    agent_bash_refresh_for(w, w->bash_jobs, 10);
    agent_set_status(w, AGENT_WORKER_IDLE);
    return NULL;
}

static int password_driver(const char *cmd, int timeout, bool noninteractive) {
    agent_config cfg = {.non_interactive = noninteractive};
    agent_worker w;
    test_worker_init(&w, &cfg);
    agent_editor editor = {0};
    const char *prompt = "q36-agent> ";
    if (editor_start(&editor, prompt, "test", "unfinished draft") != 0) return 1;
    char err[160];
    agent_bash_job *job = agent_bash_start(&w, cmd, timeout, err, sizeof(err));
    if (!job) { fprintf(stderr, "%s\n", err); return 2; }
    w.status.state = AGENT_WORKER_GENERATING;
    pthread_t thread;
    if (pthread_create(&thread, NULL, password_job_thread, &w) != 0) return 3;
    for (;;) {
        struct pollfd pfd = {.fd = w.wake_fd[0], .events = POLLIN};
        poll(&pfd, 1, 100);
        drain_wake_fd(w.wake_fd[0]);
        agent_password_request request;
        if (worker_take_password_request(&w, &request)) {
            if (editor_prompt_password(&editor, &w, &request, prompt, "test") != 0)
                abort();
        }
        pthread_mutex_lock(&w.mu);
        bool done = w.status.state == AGENT_WORKER_IDLE;
        w.wake_pending = false;
        pthread_mutex_unlock(&w.mu);
        if (done) break;
    }
    pthread_join(thread, NULL);
    bool draft_ok = !strcmp(editor.edit.buf, "unfinished draft");
    editor_stop(&editor);
    editor_restore_terminal_layout(&editor);
    char *obs = agent_bash_observation(job, false, NULL);
    printf("\nDRAFT_OK=%d\n%s", draft_ok, obs);
    free(obs);
    unlink(job->path);
    test_worker_free(&w);
    return draft_ok ? 0 : 4;
}

static void test_bash_noninteractive(void) {
    agent_config cfg = {.non_interactive = true};
    agent_worker w;
    test_worker_init(&w, &cfg);
    char err[160];
    agent_bash_job *job = agent_bash_start(&w,
        "printf stdout; printf stderr >&2; read value; test $? -ne 0", 5,
        err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (job) {
        agent_bash_refresh_for(&w, job, 5);
        AGENT_TEST_ASSERT(!job->running && job->exit_status == 0);
        char *obs = agent_bash_observation(job, false, NULL);
        AGENT_TEST_ASSERT(strstr(obs, "stdoutstderr") != NULL);
        free(obs);
        unlink(job->path);
    }
    test_worker_free(&w);
}

static void test_compaction_boundaries(void) {
    agent_qwen_tool_parser parser = {.state = AGENT_QWEN_TOOL_SEARCH};
    agent_stream_renderer stream = {.parser = &parser};
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    stream.pending_len = 1;
    AGENT_TEST_ASSERT(agent_stream_compaction_needs_lookahead(&stream));
    stream.pending_len = 0;
    stream.qwen_tool_start_len = 1;
    AGENT_TEST_ASSERT(agent_stream_compaction_needs_lookahead(&stream));
    stream.qwen_tool_active = true;
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    stream.qwen_tool_active = false;
    parser.state = AGENT_QWEN_TOOL_PARAM_VALUE;
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    int data[1000] = {0}, role[] = {42, 43, 44};
    q36_tokens tokens = {.v = data, .len = 1000};
    q36_tokens user = {.v = role, .len = 3};
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, &user) == 900);
    memcpy(data + 850, role, sizeof(role));
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, &user) == 850);
    memcpy(data + 950, role, sizeof(role));
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, &user) == 950);
    data[951] = 0;
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, &user) == 850);
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 150, 100, 100, &user) == 100);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(4096) == 512);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(100000) == 4096);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(1024) == 256);
}

static void test_observation_error_is_not_context_exhaustion(void) {
    agent_worker worker = {0};
    q36_tokens_push(&worker.transcript, 42);
    agent_tool_observation observation;
    agent_tool_observation_init(&observation);
    agent_tool_observation_puts(&observation, "image result");
    q36_vision_embedding invalid = {0};
    agent_tool_observation_add_image(&observation, &invalid);
    char err[160] = {0};
    int count = -1;
    AGENT_TEST_ASSERT(agent_tool_observation_fits(&worker, &observation, 16,
                                                 &count, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "invalid image observation") != NULL);
    AGENT_TEST_ASSERT(count == -1);
    AGENT_TEST_ASSERT(worker.transcript.len == 1 && worker.transcript.v[0] == 42);
    agent_tool_observation_free(&observation);
    q36_tokens_free(&worker.transcript);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--password-child")) return password_child();
    if (argc == 4 && !strcmp(argv[1], "--password-driver"))
        return password_driver(argv[2], atoi(argv[3]), false);
    if (argc == 4 && !strcmp(argv[1], "--noninteractive-driver"))
        return password_driver(argv[2], atoi(argv[3]), true);
    if (argc == 2 && !strcmp(argv[1], "--terminal-driver")) return test_terminal_driver();
    if (argc == 3 && !strcmp(argv[1], "--terminal-fixtures")) test_output_dir = argv[2];
    q36_agent_unit_tests_run();
    test_worker_ownership();
    test_worker_pause_ui_waits();
    test_compaction_boundaries();
    test_observation_error_is_not_context_exhaustion();
    test_atomic_file_tools();
    test_streaming_file_tools();
    test_background_jobs();
    test_shell_terminal_controls();
    test_fragmented_terminal_input();
    test_markdown_literals();
    test_hint_rendering();
    test_unicode_output_and_footer();
    test_footer_only_updates();
    test_bash_noninteractive();
    if (agent_test_failures) {
        fprintf(stderr, "q36-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("q36-agent tests: ok");
    return 0;
}
