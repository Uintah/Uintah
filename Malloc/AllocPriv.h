

#include <stdlib.h>
struct OSHunk;

struct Sentinel {
    unsigned long first_word;
    unsigned long second_word;
};

struct Tag {
    Allocator* allocator;
    size_t size;
    char* tag;
    Tag* next;
    OSHunk* hunk;
    size_t reqsize;
};

struct AllocBin {
    Tag* free;
    Tag* inuse;
    size_t maxsize;
    size_t minsize;
    int ninuse;
    int ntotal;
    size_t nalloc;
    size_t nfree;
};

struct Allocator {
    unsigned long the_lock;
    void initlock();
    inline void lock();
    void longlock();
    inline void unlock();

    void* alloc_big(size_t size, char* tag);
    
    void* alloc(size_t size, char* tag);
    void free(void*);
    void* realloc(void* p, size_t size);

    int strict;
    int lazy;
    OSHunk* hunks;

    AllocBin* small_bins;
    AllocBin* medium_bins;
    AllocBin big_bin;

    inline AllocBin* get_bin(size_t size);
    void fill_bin(AllocBin*);
    void get_hunk(size_t, OSHunk*&, void*&);

    void init_bin(AllocBin*, size_t maxsize, size_t minsize);

    void audit(Tag*, int);

    // Statistics...
    size_t nalloc;
    size_t nfree;
    size_t sizealloc;
    size_t sizefree;
    size_t nlonglocks;
    size_t nnaps;

    size_t nfillbin;
    size_t nmmap;
    size_t sizemmap;
    size_t nmunmap;
    size_t sizemunmap;

    size_t highwater_alloc;
    size_t highwater_mmap;

    size_t mysize;
};

void AllocError(char*);
