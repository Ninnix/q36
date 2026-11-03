#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef Q36_REVISION
#define Q36_REVISION "unknown"
#endif

int q36_cli_main(int argc, char **argv);
int sceSystemServiceLoadExec(const char *, const char *const *);
int sceSystemServiceHideSplashScreen(void);

static _Noreturn void leave(int status) {
    fprintf(stderr, "\nq36-ps5: finished status=%d\n", status);
    fflush(NULL);
    /* A native title leaves through LoadExec, including engine error exits. */
    sceSystemServiceLoadExec("exit", NULL);
    for (;;) sleep(1);
}

_Noreturn void __wrap_exit(int status) {
    leave(status);
}

static int data_writable(void) {
    mkdir("/data/q36", 0777);
    FILE *fp = fopen("/data/q36/.probe", "w");
    if (!fp) return 0;
    fclose(fp);
    unlink("/data/q36/.probe");
    return 1;
}

static void open_data(void) {
    if (data_writable()) return;
    /* OnionHEN handles this request for titles on its jailbreak allowlist. */
    FILE *fp = fopen("/download0/etahen_jailbreak.tmp", "w");
    if (!fp) return;
    fprintf(fp, "{\"PID\":%ld}\n", (long)getpid());
    fclose(fp);
    if (rename("/download0/etahen_jailbreak.tmp", "/download0/etahen_jailbreak") != 0) return;
    for (int attempt = 0; attempt < 40; attempt++) {
        usleep(250000);
        if (data_writable()) return;
    }
}

static int shader_directory(char *storage, size_t size) {
    snprintf(storage, size, "/download0/q36");
    if (chdir("/app0") == 0) return 1;
    /* A HEN may move the process outside its title sandbox. */
    for (int slot = 0; slot <= 50; slot++) {
        char app[128];
        snprintf(app, sizeof(app), "/mnt/sandbox/PPSA99576_%03d/app0", slot);
        if (chdir(app) != 0) continue;
        snprintf(storage, size, "/mnt/sandbox/PPSA99576_%03d/download0/q36", slot);
        return 1;
    }
    return 0;
}

int main(void) {
    open_data();
    char storage[256];
    int shaders_found = shader_directory(storage, sizeof(storage));
    const char *directories[] = {"/data/q36", storage};
    const char *directory = NULL;
    char path[1024];
    FILE *log = NULL;
    for (size_t i = 0; i < sizeof(directories) / sizeof(directories[0]); i++) {
        mkdir(directories[i], 0777);
        snprintf(path, sizeof(path), "%s/smoke.log", directories[i]);
        log = fopen(path, "w");
        if (log) {
            directory = directories[i];
            break;
        }
    }
    if (!log) leave(1);
    fclose(log);
    if (!freopen(path, "a", stdout)) leave(1);
    if (!freopen(path, "a", stderr)) leave(1);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    fprintf(stderr, "q36-ps5: revision %s, log %s\n", Q36_REVISION, path);
    sceSystemServiceHideSplashScreen();
    setenv("radv_enable_unified_heap_on_apu", "true", 1);
    setenv("Q36_THREADS", "2", 1);
    if (!shaders_found) {
        perror("q36-ps5: /app0 (shader directory)");
        leave(1);
    }

    char model[1024] = "/data/PS5LM/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf";
    snprintf(path, sizeof(path), "%s/model.txt", directory);
    FILE *fp = fopen(path, "r");
    if (fp) {
        if (!fgets(model, sizeof(model), fp) || (!strchr(model, '\n') && !feof(fp))) {
            fprintf(stderr, "q36-ps5: invalid model.txt\n");
            fclose(fp);
            leave(1);
        }
        fclose(fp);
        model[strcspn(model, "\r\n")] = '\0';
    }
    if (model[0] != '/') {
        fprintf(stderr, "q36-ps5: model path must be absolute\n");
        leave(1);
    }
    char *argv[] = {
        "q36", "--vulkan", "-m", model, "--ctx", "2048", "-t", "2",
        "--temp", "0", "-n", "64", "-p", "Explain what a radix tree is.", NULL
    };
    for (size_t i = 0; argv[i]; i++) fprintf(stderr, "q36-ps5: arg %s\n", argv[i]);
    int rc = q36_cli_main((int)(sizeof(argv) / sizeof(argv[0])) - 1, argv);
    fprintf(stderr, "\nq36-ps5: one-prompt returned %d\n", rc);
    leave(rc);
}
