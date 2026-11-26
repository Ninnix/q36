#ifndef Q36_MTP_H
#define Q36_MTP_H

#include <string.h>

/* Below this top-two logit gap, verify one proposal before drafting further. */
#define Q36_MTP_PROBE_MARGIN 1.0f

/* Reuse a continuation only when the longest matching suffix is unambiguous.
 * The known first token participates in the match but is not in history yet.
 * Bound the scan so CPU work stays small as the context grows. */
static inline int q36_mtp_lookup(const int *history, int len, int first,
                                  int *draft, int cap) {
    if (!history || !draft || cap < 2 || len < 13 || cap > len - 11) return 0;
    int best = -1, longest = 11, ambiguous = 0;
    int start = len > 65536 ? len - 65536 : 11;
    for (int pos = start; pos <= len - cap; pos++) {
        if (history[pos] != first) continue;
        int n = 1;
        while (n < 16 && n <= pos && n <= len &&
               history[pos - n] == history[len - n]) n++;
        if (n < 12 || n < longest) continue;
        if (n > longest) {
            best = pos;
            longest = n;
            ambiguous = 0;
        } else if (best >= 0 && memcmp(history + best + 1, history + pos + 1,
                                       (size_t)(cap - 1) * sizeof(int))) {
            ambiguous = 1;
        }
    }
    if (best < 0 || ambiguous) return 0;
    draft[0] = first;
    memcpy(draft + 1, history + best + 1, (size_t)(cap - 1) * sizeof(int));
    return cap;
}

/* The first token came from the target. Each following token must be the
 * best allowed token of the preceding verified row. */
static inline int q36_mtp_accepted_prefix(const int *tokens,const int *tops,
                                          int count,int eos,int excluded) {
    if (!count) return 0;
    int n=1;
    while (n<count && tokens[n-1]!=eos) {
        int top=tops[2*(n-1)];
        if (top==excluded) top=tops[2*(n-1)+1];
        if (top!=tokens[n]) break;
        n++;
    }
    return n;
}
#endif
