
//
//
// Note for the future
// Try to speed this up by using blocks > 2k
//
//

#include "New.h"
#include "NewInstr.h"
#include <stdio.h>
#include <unistd.h>
#include <Multitask/ITC.h>

#if defined(SYSV) || defined(SYSTYPE_SVR4)
#include <string.h>
#else
extern "C" void bcopy(const void *src, void *dst, int length);
extern "C" void bzero(void *b, int length);
#endif

int print_memory_stats=0;

unsigned long MemoryManager::nnew=0;
unsigned long MemoryManager::nfillbin=0;
unsigned long MemoryManager::snew=0;
unsigned long MemoryManager::ndelete=0;
unsigned long MemoryManager::sdelete=0;
unsigned long MemoryManager::nsbrk=0;
unsigned long MemoryManager::ssbrk=0;
size_t MemoryManager::last_freed_size=0;
void* MemoryManager::last_freed=0;
void* MemoryManager::lockedptr=0;
int MemoryManager::lockedbin=0;
int MemoryManager::initialized=0;
int MemoryManager::bin=0;
int MemoryManager::nbins=0;
static unsigned long largest_bin;
MemoryManager::Bin* MemoryManager::bins=0;
#ifdef SCI_LOGNEW
static FILE* log_fp;
static int logging;
static int lognew=1;
#endif

static int have_locks=0;
static void (*locker)();
static void (*unlocker)();

void MemoryManager::set_locker(void (*l)(), void (*u)())
{
    locker=l;
    unlocker=u;
    have_locks=1;
}

static inline void lock()
{
    if(have_locks)
	(*locker)();
}

static inline void unlock()
{
    if(have_locks)
	(*unlocker)();
}

#define NBITS 28

void MemoryManager::initialize()
{
    if(sizeof(MemOverhead) != 8)error("Overhead is wrong size");
    initialized=1;
    int lastsize=0;
    nbins=0;
    int size;
    for(int i=0;i<NBITS;i++){
	for(int j=8;j<=15;j++){
	    size=j<<i;
	    if(size >= lastsize+8){
		nbins++;
		lastsize=size;
	    }
	}
    }
    
    //equiv to: bins=new Bin[nbins];
    void* memend=sbrk(0);
    if(memend==(void*)-1)error("No more memory");
    // Soak up some memory to align to 1k boundary
    if((int)memend &0x3ff)sbrk(1024-(int)memend & 0x3ff);
    int allocsize=1024;
    int asize=sizeof(Bin)*nbins;
    while(asize > allocsize)allocsize+=1024;
    bins=(Bin*)sbrk(allocsize);
    if(bins==(Bin*)-1)error("No more memory");
    
    int n=0;
    lastsize=0;
    for(i=0;i<NBITS;i++){
	for(int j=8;j<=15;j++){
	    size=j<<i;
	    if(size >= lastsize+8){
		bins[n].first=0;
		bins[n].ssize=lastsize+1;
		bins[n].lsize=size;
		bins[n].n_inuse=0;
		bins[n].n_inlist=0;
		bins[n].n_reqd=0;
		bins[n].n_deld=0;
		n++;
		lastsize=size;
	    }
	}
    }
    if(n != nbins)error("Bad mismatch!");
    largest_bin=bins[nbins-1].lsize;
#ifdef SCI_LOGNEW
    if(lognew){
	log_fp=fopen("new.log", "w");
	if(!log_fp)error("Can't open log(new.log)");
	logging=0;
    }
#endif
}

void* MemoryManager::malloc(size_t size)
{
    if(!initialized)initialize();
    int osize=size+sizeof(MemOverhead);
    int l=0;
    int r=nbins;
    int m=bin;
    if(osize > largest_bin)
	error("Request for allocation too large");
    while(1){
	Bin* b=&bins[m];
	if(osize > b->lsize){
	    // too small
	    l=m+1;
	    m=(l+r)/2;
	} else if(osize < b->ssize){
	    // too big
	    r=m-1;
	    m=(l+r)/2;
	} else {
	    // fit...
	    break;
	}
    }
    bin=m; // Store in global in case the next one is the same size...
    
    // Now lock it up...
    lock();
    nnew++;
    snew+=size;
    last_freed=0;

    if(!bins[m].first)fillbin(m);
    // Get one from the bin
    BinList* ptr=bins[m].first;
    unsigned long realsize=ptr->realsize;
    bins[m].first=ptr->next;
    bins[m].n_inuse++;
    bins[m].n_inlist--;
    bins[m].n_reqd++;
    MemOverhead* ovr=(MemOverhead*)ptr;
#ifdef SCI_LOGNEW
    if(lognew && !logging){
	logging=1;
	fprintf(log_fp, "N %d %08x\n", size, ovr);
	logging=0;
    }
#endif
    // Safe to unlock now...
    unlock();

    ovr->magic=MAGIC;
    ovr->size=size;
    ovr->bin=m;
    unsigned long slop=realsize-size;
    if(slop > 248)slop=248;	// Might lose some memory here
    ovr->slop=(unsigned char)slop;
    ovr++;		// Skip past overhead
    return (void*)ovr;
}

void MemoryManager::free(void* ptr)
{
    if(!ptr)return;
#if 0
    if(!ptr)error("Freeing null ptr");
#endif
    MemOverhead* ovr=(MemOverhead*)ptr;
    ovr--;		// Go back to beginning of overhead
    if(ovr->magic != MAGIC)error("Bad Magic number on free");

    // Lock it up...
    lock();
    last_freed=ptr;
    ndelete++;
#ifdef SCI_LOGNEW
    if(lognew && !logging){
	logging=1;
	fprintf(log_fp, "D %d %08x\n", ovr->size, ptr);
	logging=0;
    }
#endif
    sdelete+=ovr->size;
    unsigned long realsize=ovr->size+ovr->slop;
    last_freed_size=(size_t)realsize;
    int b=ovr->bin;
    BinList* binptr=(BinList*)ovr;
    binptr->next=bins[b].first;
    binptr->realsize=realsize;
    bins[b].first=binptr;
    bins[b].n_inlist++;
    bins[b].n_inuse--;
    bins[b].n_deld++;
    unlock();
}

void* malloc(size_t size)
{
    return MemoryManager::malloc(size);
}

void* calloc(size_t nelem, size_t elsize)
{
    size_t tsize=nelem*elsize;
    void* ptr=MemoryManager::malloc(tsize);
#if defined(SYSV) || defined(SYSTYPE_SVR4)
    memset(ptr, (char)0, tsize);
#else
    bzero(ptr, tsize);
#endif
    return ptr;
}

void free(void* ptr)
{
    MemoryManager::free(ptr);
}

void* realloc(void* ptr, size_t newsize)
{
    void* newptr;
    if(ptr==MemoryManager::last_freed){
	// It has been freed
	MemoryManager::lock_last_freed();
	newptr=MemoryManager::malloc(newsize);
	size_t copysize=newsize;
	size_t oldsize=MemoryManager::last_freed_size;
	if(oldsize < copysize)oldsize=copysize;
#if defined(SYSV) || defined(SYSTYPE_SVR4)
	memcpy(newptr, ptr, copysize);
#else
	bcopy(newptr, ptr, copysize);
#endif
	MemoryManager::unlock_last_freed();
    } else {
	// It is still out there
	newptr=MemoryManager::malloc(newsize);
	int copysize=newsize;
	int oldsize=MemoryManager::blocksize(ptr);
	if(oldsize < copysize)copysize=oldsize;
#if defined(SYSV) || defined(SYSTYPE_SVR4)
	memcpy(newptr, ptr, copysize);
#else
	bcopy(newptr, ptr, copysize);
#endif
	MemoryManager::free(ptr);
    }
    return newptr;
}

void* operator new(size_t size)
{
    return MemoryManager::malloc(size);
}

void operator delete(void* ptr)
{
    MemoryManager::free(ptr);
}

void MemoryManager::fillbin(int b)
{
    nfillbin++;
    BinList* ptr;
    unsigned long spaceleft;
    int ibin;
    int found=0;
    // First look for a bin with 0 reqd, but something in the list
    for(ibin=b+1;ibin<nbins;ibin++){
	if(bins[ibin].first != 0 && bins[ibin].n_inlist > bins[ibin].n_reqd){
	    found=1;
	    break;
	}
    }
    if(!found){
	// Next, find the biggest bin with something in it
	for(ibin=nbins-1;ibin>b;ibin--){
	    if(bins[ibin].first != 0){
		found=1;
		break;
	    }
	}
    }
    if(found){
	// Found one
	ptr=bins[ibin].first;
	spaceleft=ptr->realsize-bins[b].lsize;
	bins[ibin].first=ptr->next;
	bins[ibin].n_inlist--;
    } else {
	// Get more core
	unsigned int sizealloc=2048;
	int reqsize=bins[b].lsize;
	if(reqsize >= 2048){
	    // Round up to 2048 byte boundary
	    sizealloc=((unsigned)reqsize+2047) & ~2047;
	}
	spaceleft=sizealloc-bins[b].lsize;
	ptr=(BinList*)sbrk(sizealloc);
	nsbrk++;
	ssbrk+=sizealloc;
	if(ptr==(void*)-1)error("No more memory");
    }
    
    // Put first part in...
    ptr->next=bins[b].first;
    ptr->realsize=bins[b].lsize;
    bins[b].first=ptr;
    bins[b].n_inlist++;
    ptr=(BinList*)((char*)ptr+bins[b].lsize);
    
    // Put second part in...
    if(spaceleft >=8){
	int l=0;
	int r=nbins;
	int m=bin;
	if(spaceleft > largest_bin)
	    error("spaceleft is too large!");
	while(1){
	    Bin* b=&bins[m];
	    if(spaceleft < b->lsize){
		// too small
		r=m-1;
		m=(l+r)/2;
	    } else if(spaceleft > (b+1)->lsize){
		// Too big
		l=m+1;
		m=(l+r)/2;
	    } else {
		// Fit...
		break;
	    }
	}
	ptr->next=bins[m].first;
	ptr->realsize=spaceleft;
	bins[m].first=ptr;
	bins[m].n_inlist++;
    }
}

void MemoryManager::lock_last_freed()
{
    MemOverhead* ovr=(MemOverhead*)last_freed;
    ovr--;
    BinList* binptr=(BinList*)ovr;
    for(int b=0;b<nbins;b++)if(bins[b].first==binptr)break;
    if(b==nbins)error("lock_last_freed screwed up!\n");
    lockedptr=binptr;
    bins[b].first=binptr->next;
    lockedbin=b;
}

void MemoryManager::unlock_last_freed()
{
    lock();
    BinList* binptr=(BinList*)lockedptr;
    binptr->next=bins[lockedbin].first;
    bins[lockedbin].first=binptr;
    unlock();
}

size_t MemoryManager::blocksize(void* ptr)
{
    if(!ptr)error("Blocksize on null ptr");
    MemOverhead* ovr=(MemOverhead*)ptr;
    ovr--;
    if(ovr->magic != MAGIC)error("Bad Magic number on blocksize");
    unsigned long realsize=ovr->size+ovr->slop;
    return realsize;
}

void MemoryManager::error(char* message)
{
    static int error=0;
    if(error){
	abort();
    }
    error=1;
    fprintf(stderr, "Fatal Memory Manager Error: ");
    fprintf(stderr,  message);
    fprintf(stderr, "\n");
    print_stats();
    abort();
}

void MemoryManager::print_stats()
{
    fprintf(stderr, "Bin Table:\n ssize  lsize        reqd     deld  inlist  inuse\n");
    for(int i=0;i<nbins;i++){
	Bin* ptr=&bins[i];
	if(ptr->n_inuse || ptr->n_inlist || ptr->n_reqd || ptr->n_deld){
	    fprintf(stderr, "%6d-%6d : %9d%9d %7d%7d\n", 
		    ptr->ssize, ptr->lsize,
		    ptr->n_reqd, ptr->n_deld,
		    ptr->n_inlist, ptr->n_inuse);
	}
    }
    fprintf(stderr, "----\n");
    fprintf(stderr, "Calls to operator new: %d (%d bytes)\n",nnew,snew);
    fprintf(stderr, "Calls to fillbin: %d (%.2f%%)\n", nfillbin, 100.*(double)nfillbin/(double)nnew);
    fprintf(stderr, "Calls to operator delete: %d (%d bytes)\n",ndelete,sdelete);
    fprintf(stderr, "Requests from sbrk: %d (%d bytes)\n",nsbrk, ssbrk);
    fprintf(stderr, "Missing allocations: %d (%d bytes)\n", nnew-ndelete, snew-sdelete);
}

class MemoryStats {
public:
    MemoryStats();
    ~MemoryStats();
};

static MemoryStats memprinter;

MemoryStats::MemoryStats()
{
    
}

MemoryStats::~MemoryStats()
{
    if(print_memory_stats)MemoryManager::print_stats();
}

#ifdef SCI_LOGNEW
void newinst_push(char* what, char* file, int line)
{
    if(lognew){
	fprintf(log_fp, "P \"%s\" \"%s\" %d\n", what, file, line);
    }
}

void newinst_pop(int line)
{
    if(lognew){
	fprintf(log_fp, "U %d\n", line);
    }
}
#endif

void MemoryManager::get_global_stats(long& nnew_, long& snew_,
				     long& nfillbin_,
				     long& ndelete_, long& sdelete_,
				     long& nsbrk_, long& ssbrk_)
{
    nnew_=nnew;
    snew_=snew;
    nfillbin_=nfillbin;
    ndelete_=ndelete;
    sdelete_=sdelete;
    nsbrk_=nsbrk;
    ssbrk_=ssbrk;
}

int MemoryManager::get_nbins()
{
    return nbins;
}

void MemoryManager::get_binstats(int bin, int& ssize, int& lsize,
				 int& n_reqd, int& n_deld, int& n_inlist)
{
    ssize=bins[bin].ssize;
    lsize=bins[bin].lsize;
    n_reqd=bins[bin].n_reqd;
    n_deld=bins[bin].n_deld;
    n_inlist=bins[bin].n_inlist;
}

void MemoryManager::audit(void* ptr)
{
    MemOverhead* ovr=(MemOverhead*)ptr;
    ovr--;		// Go back to beginning of overhead
    if(ovr->magic != MAGIC)error("Bad Magic number on audit");
}

