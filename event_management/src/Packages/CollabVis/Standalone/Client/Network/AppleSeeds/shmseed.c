/* $Id$ */


/*
 * Copyright © 2000 The Regents of the University of California. 
 * All Rights Reserved. 
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for educational, research and non-profit purposes, without
 * fee, and without a written agreement is hereby granted, provided that the
 * above copyright notice, this paragraph and the following three paragraphs
 * appear in all copies. 
 *
 * Permission to incorporate this software into commercial products may be
 * obtained by contacting
 * Eric Lund
 * Technology Transfer Office 
 * 9500 Gilman Drive 
 * 411 University Center 
 * University of California 
 * La Jolla, CA 92093-0093
 * (858) 534-0175
 * ericlund@ucsd.edu
 *
 * This software program and documentation are copyrighted by The Regents of
 * the University of California. The software program and documentation are
 * supplied "as is", without any accompanying services from The Regents. The
 * Regents does not warrant that the operation of the program will be
 * uninterrupted or error-free. The end-user understands that the program was
 * developed for research purposes and is advised not to rely exclusively on
 * the program for any reason. 
 *
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
 * EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED
 * HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO
 * OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS. 
 */


/*
 * This is an implementation of shmseed using the SysV shm{at,get} functions.
 * The locking mechanism uses F_SETLKW fcntl calls on a unique (named using the
 * client-supplied id) /tmp file.
 */


#include "config.h"
#define ASSHM_SHORT_NAMES
#include "shmseed.h"


#ifndef NULL
#  define NULL 0
#endif


#ifdef ASSHM_HAVE_SHM_H


#include <fcntl.h>    /* fcntl open */
#include <stdio.h>    /* sprintf */
#include <stdlib.h>   /* atoi realloc */
#include <string.h>   /* strchr strcpy */
#include <unistd.h>   /* fcntl getpid unlink */
#include <sys/ipc.h>  /* ftok shmat shmget */
#include <sys/shm.h>  /* shmat shmget */
#include <sys/stat.h> /* fchmod open */


/* Permission values. */
#define GROUP_WRITE  0040
#define GROUP_READ   0020
#define OTHERS_WRITE 0004
#define OTHERS_READ  0002
#define USER_WRITE   0400
#define USER_READ    0200
static mode_t READ_PERMIT[] = {GROUP_READ, OTHERS_READ, USER_READ};
static mode_t WRITE_PERMIT[] = {GROUP_WRITE, OTHERS_WRITE, USER_WRITE};


/*
 * Information on every shared memory created or attached.  #base# is the
 * address returned from shmat().  #id# is the segment identifier returned from
 * shmget().  #key# is the IPC resource key returned from ftok().  #lockFileFd#
 * is a file descriptor for the file used to lock portions of the memory.
 * #name# is the client-supplied shared memory name.
 */
typedef struct {
  void *base;
  int id;
  key_t key;
  int lockFileFd;
  char *name;
} SharedMemoryInfo;
static SharedMemoryInfo *sharedMemories = NULL;
static size_t sharedMemoriesCount = 0;


/*
 * Searches the sharedMemories global for an entry with a base of #base#.
 * Returns a pointer to the entry if found, else NULL.
 */
static SharedMemoryInfo *
FindInfo(void *base) {
  int i;
  for(i = 0; i < sharedMemoriesCount; i++) {
    if(sharedMemories[i].base == base)
      return &sharedMemories[i];
  }
  return NULL;
}


void
CloseMemory(void *base) {

  SharedMemoryInfo *info;
  char lockFilePath[63 + 1];
  struct shmid_ds shmState;

  shmdt(base);
  if((info = FindInfo(base)) != NULL) {
    (void)close(info->lockFileFd);
    if(shmctl(info->id, IPC_STAT, &shmState) == 0 && shmState.shm_nattch == 0) {
      /* No process left attached.  Delete the segment and lock file. */
      (void)shmctl(info->id, IPC_RMID, NULL);
      (void)sprintf(lockFilePath, "/tmp/shm%s", info->name);
      (void)unlink(lockFilePath);
    }
  }

}


int
LockMemory(void *base,
           size_t offset,
           AS_LockTypes how) {

  SharedMemoryInfo *info = FindInfo(base);
  struct flock flockLock;

  memset(&flockLock, 0, sizeof(flockLock));
  /* Lock 1 byte #offset# bytes from the beginning of the file. */
  flockLock.l_type   = (how == AS_READ_LOCK) ? F_RDLCK : F_WRLCK;
  flockLock.l_start  = offset;
  flockLock.l_whence = SEEK_SET;
  flockLock.l_len    = 1;
  return fcntl(info->lockFileFd, F_SETLKW, &flockLock) >= 0;

}


void *
OpenMemory(const char *name,
           size_t size) {

  SharedMemoryInfo info;
  char lockFilePath[63 + 1];
  int openFlags;
  int shmgetFlags;

  sprintf(lockFilePath, "/tmp/shm%s", name);
  openFlags = O_RDWR | ((size > 0) ? O_CREAT : 0);
  shmgetFlags = SHM_R | SHM_W | ((size > 0) ? IPC_CREAT : 0);

  /*
   * In order, try to open the lock file, use its path to create a resource
   * key, use the key to get the shared memory identifier, and attach the
   * shared memory to local memory.
   */
  if((info.lockFileFd = open(lockFilePath, openFlags, S_IRWXU)) < 0) {
    return NULL;
  }
  else if((info.key = ftok(lockFilePath, 1)) < 0) {
    (void)close(info.lockFileFd);
    return NULL;
  }
  else if((info.id = shmget(info.key, size, shmgetFlags)) < 0) {
    (void)close(info.lockFileFd);
    return NULL;
  }
  else if((info.base = shmat(info.id, NULL, 0)) == (void *)-1) {
    (void)close(info.lockFileFd);
    return NULL;
  }

  /* Success.  Add the shared memory to the list of known memories. */
  info.name = strdup(name);
  sharedMemories = realloc
    (sharedMemories, (sharedMemoriesCount + 1) * sizeof(SharedMemoryInfo));
  sharedMemories[sharedMemoriesCount++] = info;
  return info.base;

}


void
SetMemoryPermission(void *base,
                    AS_Entities who,
                    int allowReading,
                    int allowWriting) {

  struct shmid_ds ds;
  AS_Entities entity;
  SharedMemoryInfo *info = FindInfo(base);
  mode_t permissions;

  if(info != NULL) {
    shmctl(info->id, IPC_STAT, &ds);
    permissions = 0;   
    for(entity = AS_GROUP; entity <= AS_USER; entity++) {
      if(entity != who) {
        permissions |= ds.shm_perm.mode & READ_PERMIT[entity]; 
        permissions |= ds.shm_perm.mode & WRITE_PERMIT[entity]; 
      }
      else {
        if(allowReading) 
          permissions |= READ_PERMIT[entity];
        if(allowWriting) 
          permissions |= WRITE_PERMIT[entity];
      }
    }
    ds.shm_perm.mode = permissions;
    shmctl(info->id, IPC_SET, &ds);
    fchmod(info->lockFileFd, permissions);
  }

}


void
UnlockMemory(void *base,
             size_t offset) {

  SharedMemoryInfo *info = FindInfo(base);
  struct flock flockUnlock;

  memset(&flockUnlock, 0, sizeof(flockUnlock));
  /* Unlock 1 byte #offset# bytes from the beginning of the file. */
  flockUnlock.l_type   = F_UNLCK;
  flockUnlock.l_start  = offset;
  flockUnlock.l_whence = SEEK_SET;
  flockUnlock.l_len    = 1;
  (void)fcntl(info->lockFileFd, F_SETLK, &flockUnlock);

}


#else


void
CloseMemory(void *base) {
}


int
LockMemory(void *base,
           size_t offset,
           AS_LockTypes how) {
  return 0;
}


void *
OpenMemory(const char *name,
           size_t size) {
  return NULL;
}


void
SetMemoryPermission(void *base,
                    AS_Entities who,
                    int allowReading,
                    int allowWriting) {
}


void
UnlockMemory(void *base,
             size_t offset) {
}


#endif


/*
 * This shared heap implementation links all of the heap into an intermixed
 * list of free and allocated blocks.  Each block begins with a unsigned int
 * size field, which means that, on machines with four-byte ints, blocks cannot
 * exceed 2^32 - 1 bytes, of which 2^32 - 5 is available to the user.  (Since
 * the entire heap is initially a single free block, this limit also applies to
 * the total memory size, although this restriction could be removed by
 * initially dividing the memory into multiple blocks.)  Every block contains
 * an even number of bytes (reducing the maximum block size by 1); this allows
 * the least-significant bit in the size field to indicate whether the block is
 * allocated (set) or free (clear).  Block allocation is done on a first-fit
 * basis, and merging of adjacent free blocks is done during block allocation.
 */


#define ODD(n) ((n) & 1)
typedef unsigned int BlockSizeType;


/*
 * Determine CPU alignment requirements/preferences by testing how much padding
 * the compiler adds to a struct in order to force double alignment.
 */
struct AlignmentChecker {
  double aligned;
  char excess;
};
#define PADDING (sizeof(struct AlignmentChecker)-sizeof(double)-sizeof(char))
static const unsigned int alignment = PADDING + 1;
static const unsigned int blockOverhead =
  sizeof(BlockSizeType) + sizeof(BlockSizeType) % (PADDING + 1);


void *
CopyIntoHeap(SharedHeap *heap,
             const void *data,
             size_t size) {

  size_t allocSize;
  BlockSizeType *currentBlock;
  BlockSizeType *nextBlock;
  BlockSizeType remnantSize;
  void *returnValue;

  /*
   * Extend size to leave room for the 'size' indicator, reserve the low-order
   * bit as a 'used' flag, and make sure the next block is properly aligned.
   */
  allocSize = size + blockOverhead;
  if(ODD(allocSize))
    allocSize++;
  allocSize += allocSize % alignment;

  currentBlock = (BlockSizeType *)((char *)heap->base + heap->offset);
  LockMemory(heap->base, heap->offset, AS_READ_WRITE_LOCK);

  while(1) {
    if(ODD(*currentBlock)) {
      /* Block in use; try the next. */
      currentBlock = (BlockSizeType *)((char*)currentBlock + *currentBlock - 1);
      continue;
    }
    else if(*currentBlock >= allocSize) {
      /* Found a block big enough. */
      break;
    }
    nextBlock = (BlockSizeType *)((char *)currentBlock + *currentBlock);
    if(!ODD(*nextBlock)) {
      /* Merge adjacent free blocks. */
      *currentBlock += *nextBlock;
    }
    else {
      currentBlock = nextBlock;
    }
  }

  /* Allocate the whole block if the remnant is too small for a size field. */
  remnantSize = *currentBlock - allocSize;
  if(remnantSize > blockOverhead) {
    /* Otherwise, split. */
    nextBlock = (BlockSizeType *)((char *)currentBlock + allocSize);
    *nextBlock = remnantSize;
    *currentBlock = (BlockSizeType)allocSize;
  }

  (*currentBlock)++; /* Set in-use bit. */
  UnlockMemory(heap->base, heap->offset);
  returnValue = (char *)currentBlock + blockOverhead;
  if(data != NULL)
    memcpy(returnValue, data, size);
  return returnValue;

}


void
FreeInHeap(SharedHeap *heap,
                 void *garbage) {
  BlockSizeType *trashBlock =
    (BlockSizeType *)((char *)garbage - blockOverhead);
  LockMemory(heap->base, heap->offset, AS_READ_WRITE_LOCK);
  if(ODD(*trashBlock))
    (*trashBlock)--;  /* Clear in-use bit. */
  UnlockMemory(heap->base, heap->offset);
}


SharedHeap
OpenHeap(void *base,
         size_t offset,
         size_t size) {
  void *heapStart;
  SharedHeap returnValue;
  if(size > 0) {
    if(ODD(size))
      size++;
    heapStart = (char *)base + offset;
    *((BlockSizeType *)heapStart) = size;
  }
  returnValue.base = base;
  returnValue.offset = offset;
  return returnValue;
}
