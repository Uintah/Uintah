
#include <stdlib.h>

struct OSHunk {
    static OSHunk* alloc(size_t size);
    static void free(OSHunk*);
    void* data;
    OSHunk* next;

    int ninuse;
    size_t spaceleft;
    void* curr;
    size_t len;
};
