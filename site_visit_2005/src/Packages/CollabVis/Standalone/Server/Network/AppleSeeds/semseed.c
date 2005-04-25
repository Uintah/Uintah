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
 * This is an implementation of semseed using the SysV semaphore operations.
 * Because SysV does not provide open/close/unlink semantics, we use semaphores
 * in sets of two -- the first is the client semaphore, the second a count of
 * processes using the semaphore.  Note that, if a process terminates without
 * decrementing the second semaphore (e.g., because of a signal), the set will
 * not be freed.
 */


#include "config.h"
#define ASSEM_SHORT_NAMES
#include "semseed.h"

#ifdef ASSEM_HAVE_SEM_H


#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <stdio.h>


/* Permission values. */
#define GROUP_ALTER  0040
#define GROUP_READ   0020
#define OTHERS_ALTER 0004
#define OTHERS_READ  0002
#define USER_ALTER   0400
#define USER_READ    0200
static mode_t PERMIT[] =
  {GROUP_ALTER|GROUP_READ, OTHERS_ALTER|OTHERS_READ, USER_ALTER|USER_READ};

/* Semaphore usage constants. */
#define CLIENT_SEMAPHORE 0
#define PROCESS_COUNT 1
#define DOWN -1
#define UP 1


#ifndef ASSEM_HAVE_SEMUN
union semun {
  struct semid_ds *buf;
  ushort *array;
  int val;
};
#endif
static union semun semunZero = {0};


void
CloseSemaphore(sem) {
  struct sembuf decrementProcessCount = {PROCESS_COUNT, DOWN, 0};
  (void)semop(sem, &decrementProcessCount, 1);
  if(semctl(sem, PROCESS_COUNT, GETVAL, semunZero) == 0)
    semctl(sem, 0, IPC_RMID, semunZero);
}


void
Down(Semaphore sem) {
  struct sembuf down = {CLIENT_SEMAPHORE, DOWN, 0};
  semop(sem, &down, 1);
}


Semaphore
OpenSemaphore(const char *name,
              int create) {

  int i;
  struct sembuf incrementProcessCount = {PROCESS_COUNT, UP, 0};
  key_t key;
  Semaphore sem;
  int semgetFlags = USER_ALTER | USER_READ | (create ? IPC_CREAT : 0);

  /*
   * Since we have no need for associating a file with the semaphore, we
   * generate a key value directly from #name# instead of using ftok().  This
   * key generation algorithm is incredibly lame and should be improved.
   */
  key = 0;
  for(i = 0; name[i] != '\0'; i++)
    key += name[i] << i;

  if((sem = semget(key, 2, semgetFlags)) < 0)
    return NO_SEMAPHORE;
  if(create) {
    (void)semctl(sem, CLIENT_SEMAPHORE, SETVAL, semunZero);
    (void)semctl(sem, PROCESS_COUNT, SETVAL, semunZero);
  }
  (void)semop(sem, &incrementProcessCount, 1);
  return sem;

}


void
SetSemaphorePermission(Semaphore sem,
                       AS_Entities who,
                       int allowAccess) {

  struct semid_ds ds;
  AS_Entities entity;
  mode_t permissions;
  union semun semunDs;

  semunDs.buf = &ds;
  semctl(sem, CLIENT_SEMAPHORE, IPC_STAT, semunDs);
  permissions = 0;
  for(entity = AS_GROUP; entity <= AS_USER; entity++) {
    if(entity != who)
      permissions |= ds.sem_perm.mode & PERMIT[entity];
    else if(allowAccess)
      permissions |= PERMIT[entity];
  }
  ds.sem_perm.mode = permissions;
  semctl(sem, CLIENT_SEMAPHORE, IPC_SET, semunDs);

}


void
Up(Semaphore sem) {
  struct sembuf up = {CLIENT_SEMAPHORE, UP, 0};
  semop(sem, &up, 1);
}


unsigned
Value(Semaphore sem) {
  return semctl(sem, CLIENT_SEMAPHORE, GETVAL, semunZero);
}


#else


void
CloseSemaphore(Semaphore sem) {
}


void
Down(Semaphore sem) {
}


Semaphore
OpenSemaphore(const char *name,
              int create) {
  return NO_SEMAPHORE;
}


void
SetSemaphorePermission(Semaphore sem,
                       AS_Entities who,
                       int allowAccess) {
}


void
Up(Semaphore sem) {
}


unsigned
Value(Semaphore sem) {
  return 0;
}


#endif
