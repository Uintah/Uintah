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


#ifndef SHMSEED_H
#define SHMSEED_H


/*
 * This package facilities for defining and using regions shared memory.
 * Shared memory may be accessed directly or via functions that allow malloc-
 * style heap management.
 */


#include <sys/types.h>  /* size_t */
#include "appleseeds.h" /* AS_Entities AS_LockTypes */


#ifdef __cplusplus
extern "C" {
#endif


/*
 * Closes shared memory segment whose base address is #base#.  Once all
 * processes have closed a shared memory segment, the segment is destroyed.
 */
void
ASSHM_CloseMemory(void *base);


/*
 * Locks the memory region beginning #offset# bytes into the shared memory
 * segment whose base is #base#.  Acquires either a read lock or a read/write
 * lock, depending on #how#.  Note: only a single byte of the memory is locked,
 * so the caller must ensure by convention that this is sufficient to protect
 * the entire region to be read/written.
 */
int
ASSHM_LockMemory(void *base,
                 size_t offset,
                 AS_LockTypes how);


/*
 * Attempts to open the shared memory segment named #name# -- a client-
 * generated unique name.  If #size# is non-zero, creates a new segment
 * containing that many bytes; otherwise, attaches to an existing segment.
 * Returns the segment base address if successful, else NULL.
 */
void *
ASSHM_OpenMemory(const char *name,
                 size_t size);


/*
 * Determines whether #who# is allowed read and/or write access to the memory
 * whose base address is #base#, depending on whether #allowReading# and
 * #allowWriting# are non-zero.  By default, only AS_USER may access memories.
 */
void
ASSHM_SetMemoryPermission(void *base,
                          AS_Entities who,
                          int allowReading,
                          int allowWriting);


/* Removes any lock placed on the shared memory #offset# bytes from #base#. */
void
ASSHM_UnlockMemory(void *base,
                   size_t offset);


/* Information needed to access a shared heap. */
typedef struct {
  void *base;
  size_t offset;
} ASSHM_SharedHeap;


/*
 * Returns the address of #size# newly-allocated bytes within #heap#, or NULL
 * on error.  Copies #data# into the allocated memory if #data# is not NULL.
 */
void *
ASSHM_CopyIntoHeap(ASSHM_SharedHeap *heap,
                   const void *data,
                   size_t size);
/* A convenience macro for allocating unintialized shared heap memory. */
#define ASSHM_AllocateInHeap(heap,size) ASSHM_CopyIntoHeap(heap, 0, size)


/*
 * Frees the bytes pointed to by #garbage# that were previously allocated
 * within #heap# by ASSHM_AllocateInHeap.
 */
void
ASSHM_FreeInHeap(ASSHM_SharedHeap *heap,
                 void *garbage);


/*
 * Returns a heap that starts #offset# bytes into the shared memory segment
 * whose base is #base#.  If #size# is non-zero, a new heap of that size is
 * created; otherwise, the function returns an existing heap.
 */
ASSHM_SharedHeap
ASSHM_OpenHeap(void *base,
               size_t offset,
               size_t size);


#ifdef ASSHM_SHORT_NAMES

#define CloseMemory ASSHM_CloseMemory
#define LockMemory ASSHM_LockMemory
#define OpenMemory ASSHM_OpenMemory
#define SetMemoryPermission ASSHM_SetMemoryPermission
#define UnlockMemory ASSHM_UnlockMemory

#define SharedHeap ASSHM_SharedHeap
#define CopyIntoHeap ASSHM_CopyIntoHeap
#define AllocateInHeap ASSHM_AllocateInHeap
#define FreeInHeap ASSHM_FreeInHeap
#define OpenHeap ASSHM_OpenHeap

#endif


#ifdef __cplusplus
}
#endif


#endif
