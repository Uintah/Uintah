/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  Allocator.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 */

#ifdef __INTEL_COMPILER
   // Disable the fprintf warning that appears everywhere in this file on icpc.
#  pragma warning( disable : 181 )
#endif

#define LINUX_GETENV_HACK 0
/* TODO:
6) Destroy allocators
*/

#include <sci_defs/bits_defs.h>

#include <Core/Malloc/Allocator.h>

// USE_LENNY_HACK: See Allocator.h for mor information:

#if !defined( DISABLE_SCI_MALLOC )

//#define ALIGN 16
const int ALIGN=16;

#  include <Core/Malloc/AllocPriv.h>
#  include <Core/Malloc/AllocOS.h>

#  if defined(__sun) || defined(_WIN32)
#    include <cstring>
#    define bcopy(src,dest,n) memcpy(dest,src,n)
#  elif defined(__linux) || defined(__digital__) || defined(__sgi) || defined(_AIX) || defined(__APPLE__) || defined(__CYGWIN__)
#    include <cstring>
#  else
#    error "Need bcopy idfdef for this architecture"
#  endif

#  ifndef _WIN32
#    include <sys/param.h>
//   irix64 KCC stuff
#    include <strings.h>
#  endif
#  include <cstdio>

#  ifdef SCI_PTHREAD
#    include <pthread.h>
#  endif

/* we use UCONV to avoid compiler warnings. */
// NOTE(boulos): On Darwin systems, even if it's not a 64-bit build
// the compiler will generate warnings (so we use %lu for that case as
// well)
// SIZET is used for typecasting to ensure the types line up.  This helps work 
// around portability issues with printf and types of size_t.
#  if defined(SCI_64BITS) || defined(__APPLE__)
     typedef unsigned long SIZET;
#    define UCONV "%lu"
#  else
     typedef unsigned int SIZET;
#    define UCONV "%u"
#  endif

namespace SCIRun {

// Dd: For AIX
#  ifdef STATSIZE
#    undef STATSIZE
#  endif

#  define STATSIZE (4096+BUFSIZ)

#  ifndef DISABLE_SCI_MALLOC
     static char trace_buffer[STATSIZE];
#  endif

Allocator* default_allocator=0;

// Granularity of small things - 8 bytes
// Anything smaller than this is considered "small"
#define SMALL_THRESHOLD (512-8)
#define SMALLEST_ALLOCSIZE (8*1024)

#define SMALL_BIN(size) (((size)+7)>>4)
#define NSMALL_BINS ((SMALL_THRESHOLD+8)>>4)
#define SMALL_BINSIZE(bin) (((bin)<<4)+8)

// Granularity of medium things - 2k bytes
#define MEDIUM_THRESHOLD (65536*8)

#define MEDIUM_BIN(size) (((size)-1)>>11)
#define NMEDIUM_BINS ((MEDIUM_THRESHOLD)>>11)
#define MEDIUM_BINSIZE(bin) (((bin)<<11)+2048)

#define OVERHEAD (sizeof(Tag)+sizeof(Sentinel)+sizeof(Sentinel))

#define OBJFREE 1
#define OBJINUSE 2
#define OBJFREEING 3
#define OBJMEMALIGNFREEING 4

#define SENT_VAL_FREE 0xdeadbeef
#define SENT_VAL_INUSE 0xbeefface

#define NORMAL_OS_ALLOC_SIZE (512*1024)

// Objects bigger than this can't be allocated
#define MAX_ALLOCSIZE (1024*1024*1024)

static bool do_shutdown=false;
static int mallocStatsAppendNum = -1;

void AllocatorMallocStatsAppendNumber(int num)
{
  mallocStatsAppendNum = num;
}

inline size_t Allocator::obj_maxsize(Tag* t)
{
    return (t->bin == &big_bin)?(t->hunk->len-OVERHEAD):t->bin->maxsize;
}

static void account_bin(Allocator* a, AllocBin* bin, FILE* out,
                        size_t& bytes_overhead,
                        size_t& bytes_free,
                        size_t& bytes_fragmented,
                        size_t& bytes_inuse)
{
    Tag* p;
    for(p=bin->free;p!=0;p=p->next){
        bytes_overhead+=OVERHEAD;
        bytes_free+=a->obj_maxsize(p);
    }
    for(p=bin->inuse;p!=0;p=p->next){
        bytes_overhead+=OVERHEAD;
        bytes_inuse+=p->reqsize;
        bytes_fragmented+=a->obj_maxsize(p)-p->reqsize;
        if(out){
#  ifdef USE_TAG_LINENUM
            fprintf(out, "%p: "UCONV" bytes (%s:%d)\n",
                    (char*)p+sizeof(Tag)+sizeof(Sentinel),
                    (SIZET)p->reqsize, p->tag, p->linenum);
#  else
            fprintf(out, "%p: "UCONV" bytes (%s)\n",
                    (char*)p+sizeof(Tag)+sizeof(Sentinel),
                    (SIZET)p->reqsize, p->tag);
#  endif
        }
    }
}

#if !defined( USE_LENNY_HACK )
static
#endif
void
shutdown()
{
  static char stat_buffer[STATSIZE];

  Allocator* a=DefaultAllocator();
  if (a->statsfile && !a->stats_out) {
    char filename[256];
    strcpy(filename, a->statsfile);
    if (mallocStatsAppendNum >= 0) {
      strcat(filename, ".");
      sprintf(filename + strlen(filename), "%i", mallocStatsAppendNum);
    }
    a->stats_out=fopen(filename, "w");
    setvbuf(a->stats_out, stat_buffer, _IOFBF, STATSIZE);
    if(!a->stats_out){
      perror("fopen");
      fprintf(stderr, "cannot open stats file: %s, will not print stats\n",
              filename);
      a->stats_out=0;
    }
  }

  if(a->stats_out){
    if(do_shutdown){
      // We have already done this once, but we got called again,
      // so we rewind the file if we can
      rewind(a->stats_out);
    }

    // Just in case...
    a->lock();

    fprintf(a->stats_out, "Unfreed objects:\n");
    // Full accounting - go through each bin...
    size_t bytes_overhead=0, bytes_free=0, bytes_fragmented=0,
      bytes_inuse=0, bytes_inhunks=0;
    int i;
    for(i=0;i<NSMALL_BINS;i++)
      account_bin(a, &a->small_bins[i], a->stats_out, bytes_overhead,
                  bytes_free, bytes_fragmented, bytes_inuse);
    for(i=0;i<NMEDIUM_BINS;i++)
      account_bin(a, &a->medium_bins[i], a->stats_out, bytes_overhead,
                  bytes_free, bytes_fragmented, bytes_inuse);
    account_bin(a, &a->big_bin, a->stats_out, bytes_overhead, bytes_free,
                bytes_fragmented, bytes_inuse);

    // Count hunks...
    for(OSHunk* hunk=a->hunks;hunk!=0;hunk=hunk->next){
      bytes_overhead+=sizeof(OSHunk);
      bytes_inhunks+=hunk->spaceleft;
    }
    // And the ones in the bigbin...
    Tag* p;
    for(p=a->big_bin.free;p!=0;p=p->next)
      bytes_overhead+=sizeof(OSHunk);
    for(p=a->big_bin.inuse;p!=0;p=p->next)
      bytes_overhead+=sizeof(OSHunk);
    bytes_overhead+=a->mysize;
    if(bytes_inuse == 0)
      fprintf(a->stats_out, "None\n");

    fprintf(a->stats_out, "statistics:\n");
    fprintf(a->stats_out, "alloc:\t\t\t"UCONV" calls\n",(SIZET)a->nalloc);
    fprintf(a->stats_out, "alloc:\t\t\t"UCONV" bytes\n", (SIZET)a->sizealloc);
    fprintf(a->stats_out, "free:\t\t\t"UCONV" calls\n", (SIZET)a->nfree);
    fprintf(a->stats_out, "free:\t\t\t"UCONV" bytes\n", (SIZET)a->sizefree);
    fprintf(a->stats_out, "fillbin:\t\t"UCONV" calls\n", (SIZET)a->nfillbin);
    fprintf(a->stats_out, "mmap:\t\t\t"UCONV" calls\n", (SIZET)a->nmmap);
    fprintf(a->stats_out, "mmap:\t\t\t"UCONV" bytes\n", (SIZET)a->sizemmap);
    fprintf(a->stats_out, "munmap:\t\t\t"UCONV" calls\n", (SIZET)a->nmunmap);
    fprintf(a->stats_out, "munmap:\t\t\t"UCONV" bytes\n", (SIZET)a->sizemunmap);
    fprintf(a->stats_out, "highwater alloc:\t"UCONV" bytes\n",
            (SIZET)a->highwater_alloc);
    fprintf(a->stats_out, "highwater mmap:\t\t"UCONV" bytes\n",
            (SIZET)a->highwater_mmap);
    fprintf(a->stats_out, "\n");
    fprintf(a->stats_out, "breakdown of total bytes:\n");
    fprintf(a->stats_out, "in use:\t\t\t"UCONV" bytes\n", (SIZET)bytes_inuse);
    fprintf(a->stats_out, "free:\t\t\t"UCONV" bytes\n", (SIZET)bytes_free);
    fprintf(a->stats_out, "fragmentation:\t\t"UCONV" bytes\n",
            (SIZET)bytes_fragmented);
    fprintf(a->stats_out, "left in mmap hunks:\t"UCONV"\n", (SIZET)bytes_inhunks);
    fprintf(a->stats_out, "per object overhead:\t"UCONV" bytes\n",
            (SIZET)bytes_overhead);
    fprintf(a->stats_out, "\n");
    fprintf(a->stats_out, ""UCONV" bytes missing ("UCONV" memory objects)\n",
            (SIZET)a->sizealloc-(SIZET)a->sizefree, (SIZET)a->nalloc-(SIZET)a->nfree);

    a->unlock();
    do_shutdown=true;
  }
}

inline AllocBin* Allocator::get_bin(size_t size)
{
  if(size <= SMALL_THRESHOLD){
    size_t bin=SMALL_BIN(size);
    return &small_bins[bin];
  } else if(size <= MEDIUM_THRESHOLD){
    size_t bin=MEDIUM_BIN(size);
    return &medium_bins[bin];
  } else {
    return &big_bin;
  }
}

#  if defined(SCI_NOTHREAD) || defined(DISABLE_SCI_MALLOC)

void Allocator::initlock()
{
}


inline void Allocator::lock()
{
}

inline void Allocator::unlock()
{
}


void LockAllocator(Allocator * /*a*/)
{
}

void UnLockAllocator(Allocator * /*a*/)
{
}
#  else
#    ifdef SCI_PTHREAD

// This is code taken from Core/Thread/RecursiveMutex_default.cc
// I'm using this code to make sure that if a thread locks the allocator
// that this thread is the only one who can use the allocator until
// it unlocks the mutex.

// These should be made part of the allocator class when verified that it
// works.

void Allocator::initlock()
{
  // Set this to false.  We don't want to use a recursive mutex unless needed
  use_rlock = false;
  lock_count = 0;
  owner = 0;
  owner_initialized = false;

  static pthread_mutex_t init = PTHREAD_MUTEX_INITIALIZER;
  the_lock=init;
}

inline void Allocator::lock()
{
  if (!use_rlock) {
    // Lock the mutex
    if(pthread_mutex_lock(&the_lock) != 0) {
      perror("Allocator::lock: pthread_mutex_lock");
      exit(-1);
    }
  } else {
    pthread_t me= pthread_self();
    // pthread_equal returns a non zero value when they are equal
    if(owner_initialized && pthread_equal(owner, me)) {
      // Already have exclusive rights, so increment the lock count
      lock_count++;
      return;
    }
    // Lock the mutex
    if(pthread_mutex_lock(&the_lock) != 0) {
      perror("Allocator::lock: pthread_mutex_lock");
      exit(-1);
    }
  }
}

inline void Allocator::rlock()
{
  pthread_t me= pthread_self();
  // pthread_equal returns a non zero value when they are equal
  if(owner_initialized && pthread_equal(owner, me)) {
    // Already have exclusive rights, so increment the lock count
    lock_count++;
    return;
  }
  // Lock the mutex
  if(pthread_mutex_lock(&the_lock) != 0) {
    perror("Allocator::lock: pthread_mutex_lock");
    exit(-1);
  }
  // Set the owner to the calling thread
  owner = me;
  owner_initialized = true;
  use_rlock = true;
  // Start the count at 1.
  lock_count = 1;
}

inline void Allocator::unlock()
{
  if (!use_rlock) {
    if(pthread_mutex_unlock(&the_lock) != 0) {
      perror("pthread_mutex_lock");
      exit(-1);
    }
  } else {
    // If the lock_count is 0, then the thread is done using it.  Unlock it
    // and go.
    if(--lock_count == 0) {
      // Again, I don't know what to initialize owner to, so I'll make sure
      // that I set this flag to make sure we know if it is initialized.
      owner = 0;
      owner_initialized = false;
      // Since this is the last unlock for this thread, we turn off the use
      // of the recursive mutex.  If we want to use it again, we have to
      // explicitly lock the allocator again.
      use_rlock = false;
      if(pthread_mutex_unlock(&the_lock) != 0) {
        perror("pthread_mutex_lock");
        exit(-1);
      }
    }
  }
}

void LockAllocator(Allocator *a)
{
  a->rlock();
}

void UnLockAllocator(Allocator *a)
{
  a->unlock();
}

#    else

#      ifdef __sgi

void Allocator::initlock()
{
  if(init_lock(&the_lock))
    AllocError("Error initializing lock");
}


inline void Allocator::lock()
{
  spin_lock(&the_lock);
}

inline void Allocator::unlock()
{
  if(release_lock(&the_lock) != 0)
    AllocError("Error unlocking lock");
}


void LockAllocator(Allocator *a)
{
  a->lock();
}

void UnLockAllocator(Allocator *a)
{
  a->unlock();
}

#      else  // !__sgi
#        error ERROR: undefined allocator lock mode (Malloc/Allocator.cc).
#      endif // __sgi
#    endif // SCI_PTHREAD
#  endif // SCI_NOTHREAD ||| DISABLE_SCI_MALLOC

void MakeDefaultAllocator()
{
  if(!default_allocator){
    default_allocator=MakeAllocator();
  }
}

void AllocError(const char* msg)
{
  fprintf(stderr, "Allocator error: %s\n", msg);
  abort();
}

Allocator* MakeAllocator()
{
#  ifndef DISABLE_SCI_MALLOC
  // Compute the size of the allocator structures
  size_t size=sizeof(Allocator);
  int nsmall=NSMALL_BINS;
  size+=nsmall*sizeof(AllocBin);
  int nmedium=NMEDIUM_BINS;
  size+=nmedium*sizeof(AllocBin);

  OSHunk* alloc_hunk=OSHunk::alloc(size, false, 0);
  Allocator* a=(Allocator*)alloc_hunk->data;
  alloc_hunk->spaceleft=0;
  alloc_hunk->next=0;
  alloc_hunk->ninuse=1;

  a->hunks=alloc_hunk;
  a->small_bins=(AllocBin*)(a+1);
  a->mysize=size;
  // Fill in the small bin info...
  for(int j=0;j<NSMALL_BINS;j++){
    int minsize=j==0?0:SMALL_BINSIZE(j-1)+1;
    a->init_bin(&a->small_bins[j], SMALL_BINSIZE(j), minsize);
  }
  a->medium_bins=a->small_bins+nsmall;
  for(int i=0;i<NMEDIUM_BINS;i++){
    int minsize=i==0?SMALL_THRESHOLD+1:MEDIUM_BINSIZE(i-1)+1;
    a->init_bin(&a->medium_bins[i], MEDIUM_BINSIZE(i), minsize);
  }
  a->init_bin(&a->big_bin, MAX_ALLOCSIZE, MEDIUM_THRESHOLD+1);


  // See if we are in strict mode
  if(LINUX_GETENV_HACK||getenv("MALLOC_STRICT")){
    a->strict=1;
  } else {
    a->strict=0;
  }

  if(getenv("MALLOC_LAZY")){
    a->lazy=1;
  } else {
    a->lazy=0;
  }

  // Initialize stats...
  a->nmmap=1;
  a->sizemmap=size+sizeof(OSHunk);
  a->highwater_mmap=size;
  a->nalloc=a->nfree=a->sizealloc=a->sizefree=0;
  a->nfillbin=0;
  a->nmunmap=a->sizemunmap=0;
  a->highwater_alloc=0;

  a->pagesize=getpagesize();
  // Setup the lock...
  a->initlock();

  // Must run this block of code before the MALLOC_STATS code
  // because the "alloc" in the MALLOC_STATS block uses the
  // "trace_out" var that is set here.
  bool atexit_added=false;
  if(getenv("MALLOC_TRACE")){
    // Set the default allocator, since the fopen below may
    // call malloc.
    if(!default_allocator)
      default_allocator=a;
    char* file=getenv("MALLOC_TRACE");
    if(!file || strlen(file) == 0){
      a->trace_out=stderr;
    } else {
      char filename[MAXPATHLEN];
      sprintf(filename, file, getpid());
      a->trace_out=fopen(filename, "w");
      setvbuf(a->trace_out, trace_buffer, _IOFBF, STATSIZE);
      if(!a->trace_out){
        perror("fopen");
        fprintf(stderr, "cannot open trace file: %s, not tracing\n",
                file);
        a->trace_out=0;
      }
    }
    if(a->trace_out){
      if(!a->stats_out){
        a->stats_out=a->trace_out;
#if !defined( USE_LENNY_HACK )
        atexit(shutdown);
        atexit_added=true;
#endif
      }
    }
  } else {
    a->trace_out=0;
  }

  a->statsfile = 0;
  char* statsfile = getenv("MALLOC_STATS");
  if(LINUX_GETENV_HACK||statsfile){
    // Set the default allocator, since the fopen below may
    // call malloc.
    if(!default_allocator)
      default_allocator=a;
    if(!statsfile || strlen(statsfile) == 0){
      a->stats_out=stderr;
    } else {
      //char filename[MAXPATHLEN];
      //sprintf(filename, statsfile, getpid());
      a->statsfile = statsfile;

      // open stats_out at the end to make sure mpi processes have split
      a->stats_out = 0;
    }
    if((a->stats_out || statsfile) && !atexit_added){
#if !defined( USE_LENNY_HACK )	
      atexit(shutdown);
#endif
    }
  } else {
    a->stats_out=0;
  }

  a->dieing = false;
  return a;
#  else
  return NULL;
#  endif // DISABLE_SCI_MALLOC
}

void* Allocator::alloc(size_t size, const char* tag, int linenum)
{
  if(size > MEDIUM_THRESHOLD)
    return alloc_big(size, tag, linenum);
  if(size == 0)
    return 0;

  // Find a block that this will fit in...
  AllocBin* obj_bin=get_bin(size);
#  ifndef DEBUG
  if(obj_bin->maxsize < size || size < obj_bin->minsize){
    fprintf(stderr, "maxsize: "UCONV"\n", (SIZET)obj_bin->maxsize);
    fprintf(stderr, "size: "UCONV"\n", (SIZET)size);
    AllocError("Bins messed up...");
  }
#  endif
  lock();

  if(!obj_bin->free)
    fill_bin(obj_bin);
  Tag* obj=obj_bin->free;
  obj_bin->free=obj->next;
  if(obj_bin->free)
    obj_bin->free->prev=0;

  // Tell the hunk that we are using this one...
  obj->hunk->ninuse++;
  obj->tag=tag;
#  ifdef USE_TAG_LINENUM
  obj->linenum = linenum;
#  endif
  obj->next=obj_bin->inuse;
  if(obj_bin->inuse)
    obj_bin->inuse->prev=obj;
  obj->prev=0;
  obj_bin->inuse=obj;
  obj->reqsize=size;
  obj_bin->ninuse++;

  nalloc++;
  sizealloc+=size;
  size_t bytes_inuse=sizealloc-sizefree;
  
  if(sizealloc<sizefree)
    bytes_inuse=0;
  
  if(bytes_inuse > highwater_alloc)
    highwater_alloc=bytes_inuse;
  obj_bin->nalloc++;

  // Safe to unlock now
  unlock();

  // Make sure that it is still cleared out...
  if(!lazy)
    audit(obj, OBJFREE);

  // Setup the new sentinels...
  char* data=(char*)obj;
  data+=sizeof(Tag);
  Sentinel* sent1=(Sentinel*)data;
  data+=sizeof(Sentinel);
  char* d=data;
  data+=obj_maxsize(obj);
  Sentinel* sent2=(Sentinel*)data;

  sent1->first_word=sent1->second_word=
    sent2->first_word=sent2->second_word=SENT_VAL_INUSE;
  // Fill in the region between the end of the allocation and the
  // end of the chunk.
  if(strict){
    unsigned int i = 0xffff5a5a;
    unsigned int start =
      (unsigned int)((obj->reqsize+sizeof(int))/sizeof(int));
    for(unsigned int* p=(unsigned int*)d+start;
        p<(unsigned int*)sent2;p++)
      *p++=i;
  }

  if(trace_out)
#  ifdef USE_TAG_LINENUM
    fprintf(trace_out, "A %p "UCONV" (%s:%d)\n", d, (SIZET)size, tag, linenum);
#  else
  fprintf(trace_out, "A %p "UCONV" (%s)\n", d, (SIZET)size, tag);
#  endif

  if(do_shutdown)
    shutdown();
  return (void*)d;
}

// When USE_TAG_LINENUM is not defined this function could generate
// warnings, because linenum will never be used.  I could have
// #ifdef'ed the function header, but I thought that would be more
// ugly than simply getting a warning.  Since this is in a .cc file
// instead of a header file the warning should only be seen once
// rather than over and over again.
void* Allocator::alloc_big(size_t size, const char* tag, int linenum)
{
  lock();

  Tag* obj=big_bin.free;
  size_t osize=size+OVERHEAD;
  size_t maxsize=osize+(size>>4);
  for(;obj!=0;obj=obj->next){
    // See if this object is within 6.25% of the right size...
    if(obj->hunk->len > osize && obj->hunk->len <= maxsize)
      break;
  }
  if(!obj){
    // First, see if we need to clean out the list.
    int nfree=big_bin.ntotal-big_bin.ninuse;
    if(nfree >= 20){
      // Skip the first half...
      obj=big_bin.free;
      for(int i=0;i<10;i++)
        obj=obj->next;
      Tag* last=obj;
      obj=obj->next;
      // Free these ones...
      while(obj!=0){
        Tag* next=obj->next;
        OSHunk* hunk=obj->hunk;
        nmunmap++;
        sizemunmap+=hunk->len+sizeof(OSHunk);
        OSHunk::free(hunk);
        obj=next;
        big_bin.ntotal--;
      }
      last->next=0;
    }

    // Make a new one...
    size_t tsize=sizeof(OSHunk)+OVERHEAD+size;
    // Round up to nearest page size
    size_t npages=(tsize+pagesize-1)/pagesize;
    tsize=npages*pagesize;
    tsize-=sizeof(OSHunk);
    unsigned long offset = sizeof(OSHunk)%ALIGN;
    if(offset != 0)
      offset = ALIGN-offset;
    tsize -= offset;
    OSHunk* hunk=OSHunk::alloc(tsize, true, this);
    nmmap++;
    sizemmap+=tsize+sizeof(OSHunk);
    size_t diffmmap=sizemmap-sizemunmap;
    if(diffmmap > highwater_mmap)
      highwater_mmap=diffmmap;
    obj=(Tag*)hunk->data;
    obj->bin=&big_bin;
    obj->tag="never used (big object)";
#  ifdef USE_TAG_LINENUM
    obj->linenum=0;
#  endif
    obj->next=big_bin.free;
    if(big_bin.free)
      big_bin.free->prev=obj;
    obj->prev=0;
    big_bin.free=obj;
    obj->hunk=hunk;
    big_bin.ntotal++;

    // Fill in sentinel info...
    char* data=(char*)obj;
    data+=sizeof(Tag);
    Sentinel* sent1=(Sentinel*)data;
    data+=sizeof(Sentinel);
    char* d=(char*)data;
    data+=obj_maxsize(obj);
    Sentinel* sent2=(Sentinel*)data;

    sent1->first_word=sent1->second_word=
      sent2->first_word=sent2->second_word=SENT_VAL_FREE;
    if(strict){
      // Fill in the data region with markers.
      unsigned int i = 0xffff5a5a;
      for(unsigned int* p=(unsigned int*)d;
          p<(unsigned int*)sent2;p++)
        *p++=i;
    }
  }

  // Have obj now...
  if(obj->prev)
    obj->prev->next=obj->next;
  else
    big_bin.free=obj->next;
  if(obj->next)
    obj->next->prev=obj->prev;

  // Tell the hunk that we are using this one...
  obj->hunk->ninuse++;
  obj->tag=tag;
#  ifdef USE_TAG_LINENUM
  obj->linenum = linenum;
#  endif
  obj->next=big_bin.inuse;
  obj->prev=0;
  if(big_bin.inuse)
    big_bin.inuse->prev=obj;
  big_bin.inuse=obj;
  obj->reqsize=size;
  big_bin.ninuse++;

  nalloc++;
  sizealloc+=size;
  size_t bytes_inuse=sizealloc-sizefree;
  if(bytes_inuse > highwater_alloc)
    highwater_alloc=bytes_inuse;
  big_bin.nalloc++;

  // Safe to unlock now
  unlock();

  // Make sure that it is still cleared out...
  if(!lazy)
    audit(obj, OBJFREE);

  // Setup the new sentinels...
  char* data=(char*)obj;
  data+=sizeof(Tag);
  Sentinel* sent1=(Sentinel*)data;
  data+=sizeof(Sentinel);
  char* d=data;
  data+=obj_maxsize(obj);
  Sentinel* sent2=(Sentinel*)data;

  sent1->first_word=sent1->second_word=
    sent2->first_word=sent2->second_word=SENT_VAL_INUSE;
  // Fill in the region between the end of the allocation and the
  // end of the chunk.
  if(strict){
    unsigned int i = 0xffff5a5a;
    unsigned int start =
      (unsigned int)((obj->reqsize+sizeof(int))/sizeof(int));
    for(unsigned int* p=(unsigned int*)d+start;
        p<(unsigned int*)sent2;p++)
      *p++=i;
  }

  if(trace_out)
#  ifdef USE_TAG_LINENUM
    fprintf(trace_out, "A %p "UCONV" (%s:%d)\n",d, (SIZET)size, tag, linenum);
#  else
  fprintf(trace_out, "A %p "UCONV" (%s)\n",d, (SIZET)size, tag);
#  endif

  if(do_shutdown)
    shutdown();
  return (void*)d;
}

void* Allocator::realloc(void* dobj, size_t newsize)
{
  if(!dobj)
    return alloc(newsize, "realloc", 0);
  // NOTE:  Realloc after free is NOT supported, and
  // probably never will be - MP problems
  char* dd=(char*)dobj;
  dd-=sizeof(Sentinel);
  dd-=sizeof(Tag);
  Tag* oldobj=(Tag*)dd;

  // Make sure that it is still intact...
  if(!lazy)
    audit(oldobj, OBJFREEING);

  // Check the simple case first...
  AllocBin* oldbin=get_bin(oldobj->bin->maxsize);
  if(newsize <= obj_maxsize(oldobj) && newsize >= oldbin->minsize){
    size_t oldsize=oldobj->reqsize;
    oldobj->reqsize=newsize;
    // Setup the new sentinels...
    char* data=(char*)oldobj;
    data+=sizeof(Tag);
    data+=sizeof(Sentinel);
    char* d=data;
    data+=obj_maxsize(oldobj);
    Sentinel* sent2=(Sentinel*)data;

    // Fill in the region between the end of the allocation and the
    // end of the chunk.
    if(strict){
      unsigned int i = 0xffff5a5a;
      unsigned int start =
        (unsigned int)((oldobj->reqsize+sizeof(int))/sizeof(int));
      for(unsigned int* p=(unsigned int*)d+start;
          p<(unsigned int*)sent2;p++)
        *p++=i;
    }
    if(trace_out)
#  ifdef USE_TAG_LINENUM
      fprintf(trace_out, "R %p "UCONV" %p "UCONV" (%s:%d)\n", dobj,
              (SIZET)oldsize, dobj, (SIZET)newsize, oldobj->tag, oldobj->linenum);
#  else
    fprintf(trace_out, "R %p "UCONV" %p "UCONV" (%s)\n", dobj,
            (SIZET)oldsize, dobj, (SIZET)newsize, oldobj->tag);
#  endif

    return dobj;
  }

  void* nobj=alloc(newsize, "realloc", 0);
  size_t minsize=newsize;
  size_t oldsize=oldobj->reqsize;
  if(newsize > oldsize)
    minsize=oldsize;
  bcopy(dobj, nobj, minsize);
  free(dobj);
  if(trace_out)
#  ifdef USE_TAG_LINENUM
    fprintf(trace_out, "R %p "UCONV" %p "UCONV" (%s:%d)\n", dobj,
            (SIZET)oldsize, nobj, (SIZET)newsize, oldobj->tag, oldobj->linenum);
#  else
  fprintf(trace_out, "R %p "UCONV" %p "UCONV" (%s)\n", dobj,
          (SIZET)oldsize, nobj, (SIZET)newsize, oldobj->tag);
#  endif

  return nobj;
}

void* Allocator::memalign(size_t alignment, size_t size, const char* ctag)
{
  if(alignment <= 8)
    return alloc(size, ctag, 0);

  size_t asize=size+sizeof(Tag)+sizeof(Sentinel)+alignment-8;
  void* addr=(char*)alloc(asize, ctag, 0);
  char* m=(char*)addr;
  size_t misalign=((size_t)m+sizeof(Tag)+sizeof(Sentinel))%alignment;
  misalign=misalign==0?0:alignment-misalign;
  m+=misalign;
  Tag* tag=(Tag*)m;
  m+=sizeof(Tag);
  tag->bin=0;
  tag->next=tag->prev=(Tag*)addr;
  tag->hunk=0;
  tag->reqsize=size;
  tag->tag=ctag;
#  ifdef USE_TAG_LINENUM
  tag->linenum=0;
#  endif
  Sentinel* sent1=(Sentinel*)m;
  m+=sizeof(Sentinel);
  sent1->first_word=sent1->second_word=SENT_VAL_INUSE;
  return m;
}

void Allocator::free(void* dobj)
{
  //    fprintf(stderr, "Freeing %x\n", dobj);
  if(!dobj)
    return;
  char* dd=(char*)dobj;
  dd-=sizeof(Sentinel);
  dd-=sizeof(Tag);
  Tag* obj=(Tag*)dd;

  if(!obj->bin){
    // This was allocated with memalign...
    if(!lazy)
      audit(obj, OBJMEMALIGNFREEING);
    if(obj->next != obj->prev)
      AllocError("Memalign tag inconsistency, or memory corrupt!\n");
    free((void*)obj->prev);
    if(do_shutdown)
      shutdown();
    return;
  }

  // Make sure that it is still intact...
  if(trace_out)
#  ifdef USE_TAG_LINENUM
    fprintf(trace_out, "F %p "UCONV" (%s:%d)\n", dobj, (SIZET)obj->reqsize, obj->tag, obj->linenum);
#  else
  fprintf(trace_out, "F %p "UCONV" (%s)\n", dobj, (SIZET)obj->reqsize, obj->tag);
#  endif

  if(!lazy)
    audit(obj, OBJFREEING);

  AllocBin* obj_bin=get_bin(obj->bin->maxsize);


  lock();
  nfree++;
  sizefree+=obj->reqsize;
  obj_bin->nfree++;

  // Remove it from the inuse list...
  if(obj->next)
    obj->next->prev=obj->prev;
  if(obj->prev){
    obj->prev->next=obj->next;
  } else {
    obj_bin->inuse=obj->next;
  }
  obj_bin->ninuse--;

  if(obj_bin == &big_bin && obj->reqsize > 50*1024*1024){
    // Go ahead and unmap this segment...
    OSHunk* hunk=obj->hunk;
    nmunmap++;
    sizemunmap+=hunk->len+sizeof(OSHunk);
    OSHunk::free(hunk);
    big_bin.ntotal--;
  } else {
    // Put it in the free list...
    obj->next=obj_bin->free;
    if(obj_bin->free)
      obj_bin->free->prev=obj;
    obj->prev=0;
    obj_bin->free=obj;


    // Setup the new sentinels...
    char* data=(char*)obj;
    data+=sizeof(Tag);
    Sentinel* sent1=(Sentinel*)data;
    data+=sizeof(Sentinel);
    char* d=(char*)data;
    data+=obj_maxsize(obj);
    Sentinel* sent2=(Sentinel*)data;

    sent1->first_word=sent1->second_word=
      sent2->first_word=sent2->second_word=SENT_VAL_FREE;

    if(strict){
      // Fill in the data region with markers.
      unsigned int i = 0xffff5a5a;
      for(unsigned int* p=(unsigned int*)d;
          p<(unsigned int*)sent2;p++)
        *p++=i;
    }
  }
  unlock();
  if(do_shutdown)
    shutdown();
}

void Allocator::fill_bin(AllocBin* bin)
{
  nfillbin++;
  if (bin->maxsize <= MEDIUM_THRESHOLD){
    size_t tsize;
    tsize=bin->maxsize+OVERHEAD;
    unsigned int nalloc=(unsigned int)(SMALLEST_ALLOCSIZE/tsize);
    if(nalloc<1)nalloc=1;
    size_t reqsize=nalloc*tsize;

    // Get the hunk...
    OSHunk* hunk;
    void* p;
    get_hunk(reqsize, hunk, p);
    for(int i=0;i<(int)nalloc;i++){
      Tag* t=(Tag*)p;
      t->bin=bin;
      t->tag="never used";
#  ifdef USE_TAG_LINENUM
      t->linenum=0;
#  endif
      t->next=bin->free;
      if(bin->free)
        bin->free->prev=t;
      t->prev=0;
      bin->free=t;
      t->hunk=hunk;
      p=(void*)((char*)p+tsize);
      char* data=(char*)t;
      data+=sizeof(Tag);
      Sentinel* sent1=(Sentinel*)data;
      data+=sizeof(Sentinel);
      char* d=(char*)data;
      data+=t->bin->maxsize;
      Sentinel* sent2=(Sentinel*)data;

      sent1->first_word=sent1->second_word=
        sent2->first_word=sent2->second_word=SENT_VAL_FREE;
      if(strict){
        // Fill in the data region with markers.
        unsigned int i = 0xffff5a5a;
        for(unsigned int* p=(unsigned int*)d;
            p<(unsigned int*)sent2;p++)
          *p++=i;
      }
    }
    bin->ntotal+=nalloc;
  } else {
    AllocError("fill_bin not finished...");
  }
}

void Allocator::init_bin(AllocBin* bin, size_t maxsize, size_t minsize)
{
  bin->maxsize=maxsize;
  bin->minsize=minsize;
  bin->free=0;
  bin->inuse=0;
  bin->ninuse=0;
  bin->ntotal=0;
}

static void printObjectAllocMessage(Tag* obj) {
#  ifdef USE_TAG_LINENUM
  fprintf(stderr, "Object was allocated with this tag:\n%s at this line number:%d\n", obj->tag, obj->linenum);
#  else
  fprintf(stderr, "Object was allocated with this tag:\n%s\n", obj->tag);
#  endif

}

void Allocator::audit(Tag* obj, int what)
{
  char* data=(char*)obj;
  data+=sizeof(Tag);
  Sentinel* sent1=(Sentinel*)data;
  data+=sizeof(Sentinel);
  char* d=data;
  if(what != OBJMEMALIGNFREEING)
    data+=obj_maxsize(obj);
  Sentinel* sent2=(Sentinel*)data;

  //    fprintf(stderr, "sentinels: %x %x %x %x\n", sent1->first_word, sent1->second_word, sent2->first_word, sent2->second_word);


  // Check that the sentinels are OK...
  if(what == OBJFREE){
    if(sent1->first_word != SENT_VAL_FREE || sent1->second_word != SENT_VAL_FREE){
      if(sent1->first_word == SENT_VAL_INUSE){
        printObjectAllocMessage(obj);
        AllocError("Object should be free, but is tagged as INUSE");
      } else {
        fprintf(stderr, "Free object has been corrupted within\n");
        fprintf(stderr, "the 8 bytes before the allocated region\n");
        printObjectAllocMessage(obj);
        AllocError("Freed object corrupt");
      }
    }
    if(sent2->first_word != SENT_VAL_FREE || sent2->second_word != SENT_VAL_FREE){
      if(sent2->first_word == SENT_VAL_INUSE){
        AllocError("Object should be free, but is tagged as INUSE (on tail only)");
      } else {
        fprintf(stderr, "Free object has been corrupted within\n");
        fprintf(stderr, "the 8 bytes following the allocated region\n");
        printObjectAllocMessage(obj);
        AllocError("Freed object corrupt");
      }
    }
  } else if(what == OBJFREEING || what == OBJINUSE || what == OBJMEMALIGNFREEING){
    if(sent1->first_word != SENT_VAL_INUSE || sent1->second_word != SENT_VAL_INUSE){
      if(sent1->first_word == SENT_VAL_FREE){
        if(what == OBJFREEING){
          fprintf(stderr, "Pointer (%p) was freed twice!\n", d);
          printObjectAllocMessage(obj);
          AllocError("Freeing pointer twice");
        } else {
          printObjectAllocMessage(obj);
          AllocError("Object should be inuse, but is tagged as FREE");
        }
      } else {
        fprintf(stderr, "Object has been corrupted within\n");
        fprintf(stderr, "the 8 bytes before the allocated region\n");
        printObjectAllocMessage(obj);
        AllocError("Memory Object corrupt");
      }
    }
    if(what != OBJMEMALIGNFREEING){
      if(sent2->first_word != SENT_VAL_INUSE || sent2->second_word != SENT_VAL_INUSE){
        if(sent2->first_word == SENT_VAL_FREE){
          if(what == OBJFREEING){
            fprintf(stderr, "Pointer (%p) was freed twice! (tail only?)\n", d);
            printObjectAllocMessage(obj);
            AllocError("Freeing pointer twice");
          } else {
            printObjectAllocMessage(obj);
            AllocError("Object should be inuse, but is tagged as FREE");
          }
        } else {
          fprintf(stderr, "Object has been corrupted within\n");
          fprintf(stderr, "the 8 bytes after the allocated region\n");
          printObjectAllocMessage(obj);
          AllocError("Memory Object corrupt");
        }
      }
    }
  }

  // Check the space between the end of the allocation and the sentinel...
  if(strict && (what == OBJFREEING || what == OBJINUSE)){
    unsigned int i = 0xffff5a5a;
    unsigned int start =
      (unsigned int)((obj->reqsize+sizeof(int))/sizeof(int));
    for(unsigned int* p=(unsigned int*)d+start;
        p<(unsigned int*)sent2;p++){
      unsigned int p1=*p++;
      if(p1 != i){
        fprintf(stderr, "p1=0x%x (should be 0x%x)\n", (int)p1, (int)i);
        fprintf(stderr, "Object has been corrupted immediately ");
        fprintf(stderr, "after the allocated region\n");
        printObjectAllocMessage(obj);
        AllocError("Memory Object corrupt");
      }
    }
  }
  if(strict && what == OBJFREE){
    // Check the markers in the data region...
    unsigned int i = 0xffff5a5a;
    for(unsigned int* p=(unsigned int*)d;
        p<(unsigned int*)sent2;p++){
      unsigned int p1=*p++;
      if(p1 != i){
        fprintf(stderr, "Object has been written after free\n");
        printObjectAllocMessage(obj);
        AllocError("Write after free");
      }
    }
  }
}


void Allocator::get_hunk(size_t reqsize, OSHunk*& ret_hunk, void*& ret_p)
{
  // See if we have room in any of the current hunks...
  OSHunk* hunk;
  for(hunk=hunks; hunk!=0; hunk=hunk->next){
    if(hunk->spaceleft >= reqsize)
      break;
  }
  if(!hunk){
    // Always request big chunks
    size_t s=reqsize>NORMAL_OS_ALLOC_SIZE?reqsize:NORMAL_OS_ALLOC_SIZE;
    hunk=OSHunk::alloc(s, false, this);
    hunk->next=hunks;
    hunks=hunk;
    hunk->spaceleft=s;
    hunk->curr=hunk->data;
    nmmap++;
    sizemmap+=s+sizeof(OSHunk);
    size_t diffmmap=sizemmap-sizemunmap;
    if(diffmmap > highwater_mmap)
      highwater_mmap=diffmmap;
  }
  hunk->spaceleft-=reqsize;
  ret_p=hunk->curr;
  hunk->curr=(void*)((char*)hunk->curr+reqsize);
  if(default_allocator && default_allocator->trace_out)
    fprintf(default_allocator->trace_out, "H %p %p "UCONV"\n", hunk, ret_p, (SIZET)reqsize);
  ret_hunk=hunk;
}

void PrintTag(void* dobj)
{
  char* dd=(char*)dobj;
  dd-=sizeof(Sentinel);
  Sentinel* sent1=(Sentinel*)dd;
  dd-=sizeof(Tag);
  Tag* obj=(Tag*)dd;

#  ifdef USE_TAG_LINENUM
  fprintf(stderr, "tag %p: allocated by: %s at %d\n", obj, obj->tag, obj->linenum);
#  else
  fprintf(stderr, "tag %p: allocated by: %s\n", obj, obj->tag);
#  endif
  fprintf(stderr, "requested object size: "UCONV" bytes\n", (SIZET)obj->reqsize);
  fprintf(stderr, "maximum bin size: "UCONV" bytes\n", (SIZET)obj->bin->maxsize);
  fprintf(stderr, "range of object: %p - "UCONV"\n", dobj,
          (SIZET)dobj+(SIZET)obj->reqsize);
  fprintf(stderr, "range of object with overhead and sentinels: %p - %p\n",
          obj, obj+OVERHEAD);
  fprintf(stderr, "range of hunk: "UCONV" - "UCONV"\n", (SIZET)obj->hunk->data, (SIZET)obj->hunk->data+obj->hunk->len);
  fprintf(stderr, "pre-sentinels: %x %x\n",
          sent1->first_word, sent1->second_word);
  if(sent1->first_word == SENT_VAL_FREE && sent1->second_word == SENT_VAL_FREE){
    fprintf(stderr, "object should be free\n");
  } else if(sent1->first_word == SENT_VAL_INUSE && sent1->second_word == SENT_VAL_INUSE){
    fprintf(stderr, "object should be inuse\n");
  } else {
    fprintf(stderr, "status of object is unknown - sentinels must be messed up\n");
  }
}

void GetGlobalStats(Allocator* a,
                    size_t& nalloc, size_t& sizealloc,
                    size_t& nfree, size_t& sizefree,
                    size_t& nfillbin,
                    size_t& nmmap, size_t& sizemmap,
                    size_t& nmunmap, size_t& sizemunmap,
                    size_t& highwater_alloc, size_t& highwater_mmap,
                    size_t& bytes_overhead,
                    size_t& bytes_free,
                    size_t& bytes_fragmented,
                    size_t& bytes_inuse,
                    size_t& bytes_inhunks)
{
  if(!a){
    nalloc=sizealloc=nfree=sizefree=nfillbin=0;
    nmmap=sizemmap=nmunmap=sizemunmap=0;
    highwater_alloc=highwater_mmap=0;
    bytes_overhead=bytes_free=bytes_fragmented=bytes_inuse=0;
    bytes_inhunks=0;
    return;
  }
  a->lock();
  nalloc=a->nalloc;
  sizealloc=a->sizealloc;
  nfree=a->nfree;
  sizefree=a->sizefree;
  nfillbin=a->nfillbin;
  nmmap=a->nmmap;
  sizemmap=a->sizemmap;
  nmunmap=a->nmunmap;
  sizemunmap=a->sizemunmap;
  highwater_alloc=a->highwater_alloc;
  highwater_mmap=a->highwater_mmap;

  // Full accounting - go through each bin...
  bytes_overhead=bytes_free=bytes_fragmented=bytes_inuse=bytes_inhunks=0;
  int i;
  for(i=0;i<NSMALL_BINS;i++)
    account_bin(a, &a->small_bins[i], 0, bytes_overhead, bytes_free,
                bytes_fragmented, bytes_inuse);
  for(i=0;i<NMEDIUM_BINS;i++)
    account_bin(a, &a->medium_bins[i], 0, bytes_overhead, bytes_free,
                bytes_fragmented, bytes_inuse);
  account_bin(a, &a->big_bin, 0, bytes_overhead, bytes_free,
              bytes_fragmented, bytes_inuse);

  // Count hunks...
  for(OSHunk* hunk=a->hunks;hunk!=0;hunk=hunk->next){
    bytes_overhead+=sizeof(OSHunk);
    bytes_inhunks+=hunk->spaceleft;
  }
  // And the ones in the bigbin...
  Tag* p;
  for(p=a->big_bin.free;p!=0;p=p->next)
    bytes_overhead+=sizeof(OSHunk);
  for(p=a->big_bin.inuse;p!=0;p=p->next)
    bytes_overhead+=sizeof(OSHunk);
  bytes_overhead+=a->mysize;

  a->unlock();
}

// Shorter/faster version that doesn't do full accounting...
void GetGlobalStats(Allocator* a,
                    size_t& nalloc, size_t& sizealloc,
                    size_t& nfree, size_t& sizefree,
                    size_t& nfillbin,
                    size_t& nmmap, size_t& sizemmap,
                    size_t& nmunmap, size_t& sizemunmap,
                    size_t& highwater_alloc, size_t& highwater_mmap)
{
  if(!a){
    nalloc=sizealloc=nfree=sizefree=nfillbin=0;
    nmmap=sizemmap=nmunmap=sizemunmap=0;
    highwater_alloc=highwater_mmap=0;
    return;
  }
  a->lock();
  nalloc=a->nalloc;
  sizealloc=a->sizealloc;
  nfree=a->nfree;
  sizefree=a->sizefree;
  nfillbin=a->nfillbin;
  nmmap=a->nmmap;
  sizemmap=a->sizemmap;
  nmunmap=a->nmunmap;
  sizemunmap=a->sizemunmap;
  highwater_alloc=a->highwater_alloc;
  highwater_mmap=a->highwater_mmap;

  a->unlock();
}

int GetNbins(Allocator*)
{
  return NSMALL_BINS+NMEDIUM_BINS+1;
}

void GetBinStats(Allocator* a, int binno, size_t& minsize, size_t& maxsize,
                 size_t& nalloc, size_t& nfree, size_t& ninlist)
{
  AllocBin* bin;
  if(binno < NSMALL_BINS)
    bin=&a->small_bins[binno];
  else if(binno < NSMALL_BINS+NMEDIUM_BINS)
    bin=&a->medium_bins[binno-NSMALL_BINS];
  else
    bin=&a->big_bin;

  a->lock();
  minsize=bin->minsize;
  maxsize=bin->maxsize;
  nalloc=bin->nalloc;
  nfree=bin->nfree;
  ninlist=bin->ntotal-bin->ninuse;
  a->unlock();
}

Allocator* DefaultAllocator()
{
  MakeDefaultAllocator();

  return default_allocator;
}

static void audit_bin(Allocator* a, AllocBin* bin)
{
  Tag* p;
  for(p=bin->free;p!=0;p=p->next){
    if(p->next && p->next->prev != p)
      AllocError("Free list confused");
    a->audit(p, OBJFREE);
  }
  for(p=bin->inuse;p!=0;p=p->next){
    if(p->next && p->next->prev != p)
      AllocError("Inuse list confused");
    a->audit(p, OBJINUSE);
  }
}

void AuditAllocator(Allocator* a)
{
  a->lock();
  int i;
  for(i=0;i<NSMALL_BINS;i++)
    audit_bin(a, &a->small_bins[i]);
  for(i=0;i<NMEDIUM_BINS;i++)
    audit_bin(a, &a->medium_bins[i]);
  audit_bin(a, &a->big_bin);
  a->unlock();
}

void AuditDefaultAllocator()
{
  AuditAllocator(default_allocator);
}

static void dump_bin(Allocator*, AllocBin* bin, FILE* fp)
{
  for(Tag* p=bin->inuse;p!=0;p=p->next){
#  ifdef USE_TAG_LINENUM
    fprintf(fp, "%p "UCONV" %s:%d\n", (p+sizeof(Tag)+sizeof(Sentinel)),
            (SIZET)p->reqsize, p->tag, p->linenum);
#  else
    fprintf(fp, "%p "UCONV" %s\n", (p+sizeof(Tag)+sizeof(Sentinel)),
            (SIZET)p->reqsize, p->tag);
#  endif
  }
}

void DumpAllocator(Allocator* a, const char* filename)
{
  FILE* fp=fopen(filename, "w");

  if( a == NULL )
    {
      printf( "WARNING: In DumpAllocator: Allocator is NULL.\n");
      printf( "         Therefore no information to dump.");
      return;
    }
  if( fp == NULL )
    {
      perror("DumpAllocator fopen");
      exit( 1 );
    }
  fprintf(fp, "\n");
  a->lock();

  int i;
  for(i=0;i<NSMALL_BINS;i++)
    dump_bin(a, &a->small_bins[i], fp);
  for(i=0;i<NMEDIUM_BINS;i++)
    dump_bin(a, &a->medium_bins[i], fp);
  dump_bin(a, &a->big_bin, fp);
  a->unlock();
  fclose(fp);
}

void Allocator::noninline_unlock()
{
  unlock();
}

} // End namespace SCIRun

#endif // !defined( DISABLE_SCI_MALLOC )
