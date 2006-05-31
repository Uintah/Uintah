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


#ifndef SEMSEED_H
#define SEMSEED_H


/*
 * This package provides facilities for manipulating semaphores.  Semaphore
 * values may be incremented, decremented, and queried, and an attempt to
 * decrement a semaphore with a zero value will block the process/thread until
 * another process/thread increments the semaphore.
 */


#include <sys/types.h>  /* size_t */
#include "appleseeds.h" /* AS_Entities */


#ifdef __cplusplus
extern "C" {
#endif


/* A semaphore handle. */
typedef int ASSEM_Semaphore;
/* An invalid semaphore value. */
#define ASSEM_NO_SEMAPHORE ((ASSEM_Semaphore)-1)


/*
 * Removes any connection to #sem# from this process.  Once all processes have
 * closed a semaphore, the semaphore is destroyed.
 */
void
ASSEM_CloseSemaphore(ASSEM_Semaphore sem);


/*
 * Decrements the value of #sem#.  If #sem# is zero, blocks the process until
 * another process performs an ASSEM_Up.
 */
void
ASSEM_Down(ASSEM_Semaphore sem);


/*
 * Returns a semaphore referenced by the client-generated unique name #name#,
 * or ASSEM_NO_SEMAPHORE on error.  If #create# is non-zero, creates a new
 * semaphore.
 */
ASSEM_Semaphore
ASSEM_OpenSemaphore(const char *name,
                    int create);

/*
 * Determines whether #who# is allowed access to #sem#, depending on whether
 * #allowAccess# is non-zero.  By default, only AS_USER may access semaphores.
 */
void
ASSEM_SetSemaphorePermission(ASSEM_Semaphore sem,
                             AS_Entities who,
                             int allowAccess);


/*
 * Increases the value of #sem# by one.  If #sem# is zero, unblocks a process
 * that's waiting on a call to Down.
 */
void
ASSEM_Up(ASSEM_Semaphore sem);


/* Returns the current value of #sem#. */
unsigned
ASSEM_Value(ASSEM_Semaphore sem);


#ifdef ASSEM_SHORT_NAMES

#define Semaphore ASSEM_Semaphore
#define NO_SEMAPHORE ASSEM_NO_SEMAPHORE

#define CloseSemaphore ASSEM_CloseSemaphore
#define Down ASSEM_Down
#define OpenSemaphore ASSEM_OpenSemaphore
#define SetSemaphorePermission ASSEM_SetSemaphorePermission
#define Up ASSEM_Up
#define Value ASSEM_Value

#endif


#ifdef __cplusplus
}
#endif


#endif
