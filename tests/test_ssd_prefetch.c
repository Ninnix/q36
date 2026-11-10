#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

#ifdef POSIX_FADV_WILLNEED
static int record_advice(int fd, off_t offset, off_t bytes, int advice);
#define posix_fadvise record_advice
#include "../q36_ssd.c"
#undef posix_fadvise

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready = PTHREAD_COND_INITIALIZER;
static uint32_t calls;
static off_t offsets[4];
static ino_t inode;
static bool release_first;

static int record_advice(int fd, off_t offset, off_t bytes, int advice) {
    struct stat st;
    assert(fstat(fd,&st) == 0 && st.st_ino == inode);
    assert(bytes == 4096 && advice == POSIX_FADV_WILLNEED);
    pthread_mutex_lock(&mu);
    assert(calls < 4);
    offsets[calls++] = offset;
    pthread_cond_signal(&ready);
    while (calls == 1 && !release_first) pthread_cond_wait(&ready,&mu);
    pthread_mutex_unlock(&mu);
    return 0;
}

static void wait_calls(uint32_t count) {
    struct timespec deadline;
    assert(clock_gettime(CLOCK_REALTIME,&deadline) == 0);
    deadline.tv_sec += 5;
    while (calls < count) assert(pthread_cond_timedwait(&ready,&mu,&deadline) == 0);
}

int main(void) {
    FILE *file = tmpfile();
    struct stat st;
    assert(file && fstat(fileno(file),&st) == 0);
    inode = st.st_ino;
    q36_ssd_prefetch *p = q36_ssd_prefetch_create(fileno(file));
    assert(p && (fcntl(p->fd,F_GETFD) & FD_CLOEXEC));
    fclose(file);
    q36_ssd_range first[] = {{1,4096},{2,4096}}, next = {9,4096}, latest = {10,4096};
    assert(q36_ssd_prefetch_submit(p,first,2) == 8192);
    pthread_mutex_lock(&mu);
    wait_calls(1);
    assert(q36_ssd_prefetch_submit(p,&next,1) == 4096);
    assert(q36_ssd_prefetch_submit(p,NULL,0) == 0);
    assert(q36_ssd_prefetch_submit(p,&latest,1) == 4096);
    release_first = true;
    pthread_cond_signal(&ready);
    wait_calls(2);
    pthread_mutex_unlock(&mu);
    int owned_fd = p->fd;
    q36_ssd_prefetch_free(p);
    assert(fcntl(owned_fd,F_GETFD) == -1 && errno == EBADF);
    assert(calls == 2 && offsets[0] == 1 && offsets[1] == 10);
    q36_ssd_prefetch_free(NULL);
    assert(q36_ssd_prefetch_create(-1) == NULL);
    puts("SSD read-ahead cancellation, descriptor ownership and shutdown: OK");
    return 0;
}
#else
int main(void) { puts("SSD read-ahead: unavailable on this platform"); return 0; }
#endif
