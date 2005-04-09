
#ifndef Lib_New_h
#define Lib_New_h 1

#ifdef linux
#define __MALLOC_0_RETURNS_NULL
#endif

#include <stdlib.h>

struct MemoryManager {
    static unsigned long nnew;
    static unsigned long nfillbin;
    static unsigned long snew;
    static unsigned long ndelete;
    static unsigned long sdelete;
    static unsigned long nsbrk;
    static unsigned long ssbrk;
    struct BinList {
	struct BinList* next;
	unsigned long realsize;
    };
    struct Bin {
	int ssize;
	int lsize;
	int n_inuse;
	int n_inlist;
	int n_reqd;
	int n_deld;
	BinList* first;
    };
    static Bin* bins;
    static int bin;
    static int nbins;
    enum {MAGIC = 0x52};
    struct MemOverhead {
	unsigned short magic;
	unsigned char bin;
	unsigned char slop;
	unsigned long size;
    };
    static int initialized;
    static void initialize();
    static void fillbin(int);
    static void error(char*);
    static void* lockedptr;
    static int lockedbin;
    static void* malloc(size_t);
    static void free(void*);
    static void* last_freed;
    static void lock_last_freed();
    static size_t last_freed_size;
    static void unlock_last_freed();
    static size_t blocksize(void*);
    friend void* operator new(size_t);
    friend void operator delete(void*);
    friend void* malloc(size_t);
    friend void free(void*);
    friend void* calloc(size_t, size_t);
    friend void* realloc(void*, size_t);
    static void print_stats();
    static int get_nbins();
    static void get_binstats(int bin, int& ssize, int& lsize,
			     int& n_reqd, int& n_deld,
			     int& n_inlist);
    static void get_global_stats(long& nnew_, long& snew_,
				 long& nfillbin_,
				 long& ndelete_, long& sdelete_,
				 long& nsbrk_, long& ssbrk_);
    static void set_locker(void (*)(), void (*)());
    static void audit(void*);
};

#endif
