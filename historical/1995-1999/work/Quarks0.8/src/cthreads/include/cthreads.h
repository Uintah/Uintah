/* 
 * Copyright (c) 1990, 1995 The University of Utah and
 * the Computer Systems Laboratory (CSL).  All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software is hereby
 * granted provided that (1) source code retains these copyright, permission,
 * and disclaimer notices, and (2) redistributions including binaries
 * reproduce the notices in supporting documentation, and (3) all advertising
 * materials mentioning features or use of this software display the following
 * acknowledgement: ``This product includes software developed by the Computer 
 * Systems Laboratory at the University of Utah.''
 *
 * THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
 * IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
 * ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * CSL requests users of this software to return to csl-dist@cs.utah.edu any
 * improvements that they make and grant CSL redistribution rights.
 *
 * 	Utah $Hdr$
 */

/* 
 * File:	cthread.h
 * Description: Define just enough to compile cthread programs.
 * Author:	Leigh Stoller
 * 		Computer Science Dept.
 * 		University of Utah
 * Date:	15-Nov-90
 * 
 */

#ifndef CTHREAD_H
#define CTHREAD_H


#ifdef __cplusplus			/* do not leave open across includes */
extern "C" {					/* for C++ V2.0 */
#endif




       /* Basic Data Types */
typedef unsigned long	any_t;
typedef any_t (*cthread_fn_t)(any_t);



/*
 * Threads primitives.
 */
typedef struct cthread  *cthread_t;

/*
 *  Start a new thread of control in which "func(arg)" is executed
 *  concurrently with the caller's thread.  
 */
cthread_t cthread_fork( cthread_fn_t func, any_t arg);


/*
 * Terminate the calling thread.  An implicit cthread_exit() occurs
 * when the top-level function of a thread returns, but it may also
 * be called explicitly.
 */
void cthread_exit( any_t result);


/*
 * Suspend the caller until the specified thread "t" terminates.
 */
any_t cthread_join(cthread_t t);


/*
 * The detach operation is used to indicate that the given thread will
 * never be joined.  This is usually known at the time the thread is forked
 * so the most efficient usage is the following.
 *
 *        cthread_detach( cthread_fork( procedure, argument ) );
 * 
 */
void cthread_detach(cthread_t t);


/* 
 * This procedure is a hint to the scheduler, suggesting that this would be
 * a convenient point to schedule another thread to run on the current
 * processor.
 *
 * Note that this implementation uses a preemptive scheduler.  This means
 * that explicit calls to cthread_yield are not really necessary since a 
 * switch to another thread will occur behind your back anyway.
 * 
 */
void cthread_yield(void);

/*
 * Return the caller's own thread identifer which is the same value that
 * was returned by cthread_fork() to the creator of the thread.
 */
cthread_t cthread_self(void);



/*
 * Mutex objects.
 */
typedef struct mutex *mutex_t;

/*
 * Allocate a new mutex
 */
mutex_t mutex_alloc(void);

/*
 * Free a mutex.  Note that it is erroneous to free a mutex
 * which is still active (e.g. which has a thread blocked waiting
 * for it to become free).
 */
void mutex_free(mutex_t m);


/* 
 * Acquire a lock on mutex m.  The calling thread will block until
 * the lock is available.  Only one thread may hold the lock at a time.
 */
void mutex_lock(mutex_t m);


/* 
* Attempt to acquire a lock on mutex m.  Return TRUE if successful, FALSE if
 * not.  No blocking occurs and there is also no guarantee you'll get
 * the lock.
 */
int mutex_try_lock(mutex_t m);


/*
 * Unlock the mutex m, giving other threads a chance to lock it.
 */
void mutex_unlock(mutex_t m);




/*
 * Condition variables.
 */
typedef struct condition  *condition_t;

/*
 * Allocate a condition variable 
 */
condition_t condition_alloc(void);

/*
 * Free a condtition variable.  As with mutex_free, it is erroneous
 * to free an active condition.
 */
void condition_free(condition_t c);

/* 
 * Signal that the condition represented by the condition variable "c" 
 * is now true.  If any threads are waiting for the condition (via
 * condition_wait()) then at least one of them will be awakened.
 * If none are waiting, nothing happens.
 */
void condition_signal(condition_t c );
void condition_broadcast(condition_t c );

/*
 *  Unlock the mutex m and suspend the calling thread until the condition
 *  c is likely to be true. At this point, the mutex lock is acquired again
 *  and the thread resumes.  There is no guarantee that the condition
 *  is actually true when the thread resumes so use of this procedure
 *  show always be in the form:
 *
 *      mutex_lock(m);
 *         .
 *         .
 *         .
 *     while ( (* condition not true *) )
 *         condition_wait(c,m);
 *         .
 *         .
 *         .
 *     mutex_unlock(m);
 */
 void condition_wait(condition_t c, mutex_t m );


      /* Global data (ugh) */
extern int cthread_debug;    /* set to 1 for debugging output */
extern mutex_t stdio_mutex;  /* mutex to hold when doing I/O to std{in,out,err} */

/*
 * Write out an error message and halt 
 */
void cthread_panic( char *s);

#ifdef __cplusplus
}						/* for C++ V2.0 */
#endif

#endif /* CTHREAD_H  */
