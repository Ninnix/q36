#include <sys/types.h>
#include <errno.h>

int isatty(int fd) {
    (void)fd;
    errno = ENOTTY;
    return 0;
}

/* These calls are cache hints; no title module exports posix_fadvise. */
int posix_fadvise(int fd, off_t offset, off_t length, int advice) {
    (void)fd;
    (void)offset;
    (void)length;
    (void)advice;
    return 0;
}
