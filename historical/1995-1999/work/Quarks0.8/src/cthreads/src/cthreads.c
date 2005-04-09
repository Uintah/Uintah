/*
 *  Purpose: cthreads sample implementation
 *
 *  James Painter,  University of Utah
 *   Last Modified: October 4, 1991
 *
 */
/*
 * Copyright (c) 1991, Univeristy of Utah
 * Copying, use and development for non-commercial purposes permitted.
 *                  All rights for commercial use reserved.
 */


#include <stdio.h>
#include <assert.h>

#include "cthread_internals.h"
#include "preempt.h"


/* Prototypes for externals */
void malloc_init(void);


/* Exported globals */
int cthread_debug = 0;
mutex_t stdio_mutex;		/* used when we do I/O */


/* Macros */
     /* Handy for short 1 line critical sections */
#define CRITICAL_SECTION(lock,code) \
  { spin_lock_yield(lock);   code;    spin_unlock(lock);  }


/* Macro to grab the next thread off the ready queue, or return the
 * idle thread if no other is available.
 */
static cthread_t _next;
#define NEXT_THREAD()     \
      (_next = cthread_dequeue(&readyq)) ? _next : idle_cthread

/*
 * Some important threads
 */
static cthread_t idle_cthread = 0; /* The idle thread runs the scheduling loop */

static cthread_t main_cthread = 0; /* the main program runs as this thread */

/*
 * Important Queues
 */

static int dummy1[4];

    /* The queue of ready threads */
static struct cthread_queue readyq;

/*
 * The free queue is the queue of threads contexts ready to be reused.
 */
static struct cthread_queue free_queue;

static spin_lock_t living_lock;
static int cthreads_living = 0; 	/* counts the # of living threads */


void 
cthread_panic(char *s)
{
    splhigh();
    fprintf( stderr, "cthread panic: %s\n", s );
    abort();
    /* NOTREACHED */
}

/* ---------------------------------------------------------------------- */
   /* Cthread Queue implementation */


/*
 * Sanity checking code to be sure out queues don't get corrupted by
 * locking errors.  The hope is to make debugging easier by catching
 * problems as early as possible 
 */
#ifdef DEBUG
void valid_queue(cthread_queue_t q)
{
  cthread_t last = 0, next = q->head;
  int count = 0;

  while (next != 0)
    {
      count++;
      last = next;
      next = next->t_next;
    }
  if (!( last == q->tail && count == q->len))
    abort();
}
#else
#define valid_queue(q) assert( (q->len && q->head && q->tail) || \
                                !q->len && !q->head && !q->tail)
#endif

void 
queue_init(cthread_queue_t q)
{
    spin_lock_init(&q->lock);
    q->head = q->tail = 0;
    q->len = 0;
}


void 
cthread_enqueue(cthread_queue_t q, cthread_t t)
{
  cthread_t oldtail = q->tail;

  if (t == oldtail)
    abort();

  valid_queue(q);		/* queue okay? */

  /* MUST hold q->lock */
    if (q->tail)  {
	q->tail->t_next = t;
	q->tail = t;
    } else  {
	q->head = q->tail = t;
    }
    t->t_next = 0;
    q->len++;
  valid_queue(q);
}



cthread_t
cthread_dequeue(cthread_queue_t q)
{
    cthread_t t  = 0;

    /* MUST hold q->lock */
    valid_queue(q);
    if (q->head)  {
	t = q->head;
	q->head = t->t_next;
	if (!q->head)
	    q->tail = 0;
	t->t_next = 0;
	q->len--;
    }
    if (q->len && (!q->head || !q->tail) )
	cthread_panic("corrupt queue");
    valid_queue(q);
    return t;
}

/* ------------------------------------------------------------------------ */


/*
 * Atomically enqueue the current thread on a wait queue, release the
 * lock on the queue and block further execution of the thread.
 *
 * We have to be a little careful here since we don't want to be put on
 * the ready queue after we've already put ourselves on the wait queue 
 */
static void 
atomic_enqueue_and_block( cthread_queue_t q, spin_lock_t *qlock )
{
    cthread_t next, self = cthread_self();

    spin_lock(&readyq.lock);
    cthread_enqueue( q, self );
    spin_unlock(qlock);
    next = NEXT_THREAD();
    cthread_switch( &self->t_sp, &next->t_sp, &readyq.lock );
}



/*
 * Internal scheduling loop. The idle thread runs this loop.
 */
void
scheduler_loop(void)
{
    cthread_t t;

    for(;;)
    {
	/* Fire up the next ready thread */
	if (!queue_empty(&readyq)) {   /* a thread  is ready */
	    spin_lock(&readyq.lock);
	    t = cthread_dequeue(&readyq);
	    if (t)
		cthread_switch(&idle_cthread->t_sp, &t->t_sp, &readyq.lock );
	    else
		spin_unlock(&readyq.lock);
	} else {
 	      /* This would be a good place to spin the console lights
                 or some other equally useless activity.  On a uniprocessor,
                 being here means we are either all done (only the idle
                 thread left) or deadlocked since any waiting threads
                 have no way to be awakened.
              */
	    if (cthreads_living != 1)
	      {
		if (cthread_debug)
		  cthread_panic( "--> Deadlock <--" );
		else
		  {
		    fprintf( stderr, "all threads are blocked; program exiting\n");
		    exit(0);
		  }
	      }
	    else {
		if ( cthread_debug)
		    printf( "All threads finished\n" );
		exit(main_cthread->result);
	    }
	}
    }
    /* NOTREACHED */

}

/*
 * Allocate a new thread structure and stack reusing a free one if possible
 */
cthread_t 
cthread_alloc(void)
{
    cthread_t t;

    spin_lock_yield(&free_queue.lock);
    if (!queue_empty(&free_queue)) {
	t = cthread_dequeue(&free_queue);
    } else {
	t = (cthread_t) calloc( sizeof(struct cthread), 1 );
	if (!t) cthread_panic( "unable to malloc memory\n" );
	stack_alloc( t, &t->t_stackbase );
    }
    spin_unlock(&free_queue.lock);

    CRITICAL_SECTION(&living_lock,cthreads_living++);

    t->detached = t->done = 0;
    t->joiner = (cthread_t) 0;
    t->t_next = (cthread_t) 0;
    spin_lock_init(&t->result_lock);

    return t;
}

/*
 * Initialize cthreads creating a cthread for main and an idle cthread which
 * runs the scheduling loop.
 */
stack_pointer_t
cthread_init(void)
{
    int dummy;

    if (cthreads_living)
	return 0;		/* no reason to do it more than once */

#ifdef PORTABLE_CSW
    set_top_context(&dummy);
#endif
		    
    /* Initialize the low level support */
    malloc_init();
    stack_init();
    stdio_mutex = mutex_alloc();

    queue_init(&readyq);
    queue_init(&free_queue);
    spin_lock_init(&living_lock);


    /* make a thread for main */
    main_cthread = cthread_alloc();
    cthread_prepare( main_cthread, &main_cthread->t_sp, 
		     main_cthread->t_stackbase );

    /* make an idle thread */
    idle_cthread = cthread_alloc();
    idle_cthread->func = (cthread_fn_t) scheduler_loop;
    idle_cthread->arg  = 0;
    cthread_prepare( idle_cthread, &idle_cthread->t_sp, 
		     idle_cthread->t_stackbase );

    /* On a multiprocessor, we would probably want to make
       additional idle threads for each processor.

       The originating processor starts running the main thread.
       Each coprocessor would start in a idle thread waiting
       for something to do.
    */
    if (cthread_debug) {
	mutex_lock(stdio_mutex);
	  printf( "[main thread: %x started]\n", main_cthread );
	mutex_unlock(stdio_mutex);
    }
    
    startclock(); 
    return main_cthread->t_stackbase;
	/* tells the startup code what stack main should run on */
}


/*
 * Every cthread (except main) starts life in this function 
 */
void 
cthread_body(cthread_t self)
{
  any_t result;

    /* This is the body of every thread
       We run the function associated with the thread then clean up.
    */
    if (cthread_debug && self != idle_cthread) {
	mutex_lock(stdio_mutex);
	printf( "[thread: %x started]\n", self );
	mutex_unlock(stdio_mutex);
    }

    result = (*self->func)(self->arg);
    cthread_exit(result);
    /* NOTREACHED */
}


/*
 * Clean up when a cthread exists and handle the details of pairing up
 * the exit with the corresponding join.
 */
void 
cthread_exit(any_t result)
{
    cthread_t t = cthread_self(), next = (cthread_t) 0;
    int free;

    if (cthread_debug) {
	mutex_lock( stdio_mutex );
	if (t == main_cthread)
	    printf( "[main thread: %x exit]\n", t );
	else
	    printf( "[thread: %x exit]\n", t );
	mutex_unlock( stdio_mutex );
    }

    /* Set the return value */
    spin_lock_yield( &t->result_lock );
    t->result = result;
    t->done = 1;
    if (t->joiner) next = t->joiner;

    CRITICAL_SECTION(&living_lock,cthreads_living--);

    if (t->detached) {
	spin_lock_yield(&free_queue.lock);
	spin_unlock(&t->result_lock);
	atomic_enqueue_and_block( &free_queue, &free_queue.lock);
	cthread_panic( "You can't get here from there\n" );
    } else {
	spin_lock(&readyq.lock);
	spin_unlock( &t->result_lock );
	if (!next) next = NEXT_THREAD();
	cthread_switch(&t->t_sp, &next->t_sp, &readyq.lock );
	cthread_panic( "You can't get here from there\n" );
    }
    /* NOTREACHED */
}


/*
 * Detach a thread.
 */
void 
cthread_detach(cthread_t t)
{
  spin_lock_yield(&t->result_lock);
  if (t->joiner)
	cthread_panic( "Attempt to detach a thread which is to be joined" );
  if (t->detached)
	cthread_panic( "Attempt to detach a thread which is already detached");
  if (t->done)  {
    spin_lock_yield(&free_queue.lock);
    cthread_enqueue(&free_queue,t);
    spin_unlock(&free_queue.lock);
  }
  t->detached = 1;
  spin_unlock(&t->result_lock);
}


/* Join with another thread.   */
any_t 
cthread_join(cthread_t t)
{
    any_t result;
    cthread_t next, self = cthread_self();

    spin_lock_yield( &t->result_lock );
    if (t->detached)
	cthread_panic( "attempt to join with a detached thread" );

    if (!t->done) {
	/* Have to wait for it to finish */
	spin_lock(&readyq.lock);
	t->joiner = self;
	spin_unlock(&t->result_lock);
	next = NEXT_THREAD();
	cthread_switch( &self->t_sp, &next->t_sp, &readyq.lock );
	/* When we return the thread has completed */
	spin_lock_yield(&t->result_lock);
    }

    result = t->result;
    spin_unlock( &t->result_lock );

    CRITICAL_SECTION(&free_queue.lock,cthread_enqueue(&free_queue,t));

    /* It is an unchecked error if another thread attempts to join t 
       now.  
       */
    return result;
}


/*
 * Fork a thread.  Allocate and prepare it then put it on the readyq
 */
cthread_t 
cthread_fork( cthread_fn_t func, any_t arg)
{
    cthread_t t = cthread_alloc();  

    t->func = func;
    t->arg  = arg;
    cthread_prepare( t, &t->t_sp, t->t_stackbase ); 

    CRITICAL_SECTION(&readyq.lock, cthread_enqueue(&readyq,t) );
    return t;
}


/*
 *  If we're running in the idle thread or if we can't acquire
 *  the scheduler lock we just return.  cthread_yield is only
 *  a hint anyway.
 *
 *  Otherwise, we put ourselves at the end of the readyq and
 *  run the next thread.
 */
void 
cthread_yield(void)
{
    cthread_t self = cthread_self(), next;

/*
    printf("Trying to yield\n");
    if (spin_lock_locked(&readyq.lock))
	printf("readyqlock is locked\n");
    else
	printf("readyqlock is NOT locked\n");
*/

    if((self != idle_cthread) && !queue_empty(&readyq) &&
        spin_try_lock(&readyq.lock)) {

	cthread_enqueue( &readyq, self );
	next = NEXT_THREAD();
/*	fprintf( stderr, "cthread_switch: %08x %08x\n", self, next ); */
	cthread_switch( &self->t_sp, &next->t_sp, &readyq.lock );
    }
/*
    printf("After yield\n");
    if (spin_lock_locked(&readyq.lock))
	printf("readyqlock is locked\n");
    else
	printf("readyqlock is NOT locked\n");
*/
}


/*
 * Malloc memory for a mutex and initialize it to the unheld state
 */
mutex_t 
mutex_alloc(void)
{
    mutex_t m = (mutex_t) malloc(sizeof(struct mutex));

    if (!m) cthread_panic( "no memory for mutex");
    queue_init(&m->waiters);
    m->held = 0;
    return m;
}
 
/*
 * Free a mutex.  It is an error to free it while it is still held
 */
void 
mutex_free(mutex_t m)
{
    if (m->held) cthread_panic( "Freeing a held mutex" );
    free(m);
}


/*
 * Lock down a mutex using a wait queue. 
 */
void 
mutex_lock(mutex_t m)
{
    spin_lock_yield(&m->waiters.lock);		
    if (!m->held) {
	m->held = 1;
	spin_unlock(&m->waiters.lock);
	return;
    } else {
	atomic_enqueue_and_block( &m->waiters, &m->waiters.lock );
	/* HANDOFF: When we start running again we will have the mutex */
    }
}


/*
 * Grab the mutex if we can, otherwise return
 */
int 
mutex_try_lock(mutex_t m)
{
    int result;

    CRITICAL_SECTION(&m->waiters.lock, {result=!m->held; if(result) m->held=1;} );
    return result;
}


/* 
 * Unlock the mutex or hand it off to the next cthread in the wait queue
 */
void 
mutex_unlock(mutex_t m)
{
    cthread_t t;

    spin_lock_yield(&m->waiters.lock);
    if (!queue_empty(&m->waiters)) {
	t = cthread_dequeue(&m->waiters);
	spin_unlock(&m->waiters.lock);
	CRITICAL_SECTION(&readyq.lock, cthread_enqueue(&readyq,t) );
    } else {
	m->held = 0;
	spin_unlock(&m->waiters.lock);
    }
}



/*
 * Allocate memory for a condition 
 */
condition_t 
condition_alloc(void)
{
    condition_t c = (condition_t) malloc( sizeof(struct condition) );

    if (!c) cthread_panic( "no memory for condition");

    queue_init(&c->waiters);
    return c;
}

 
/*
 * Free the condition 
 */
void
condition_free(condition_t c)
{
    if (!queue_empty(&c->waiters))
	cthread_panic( "freeing a non-empty condition" );
    free(c);
}


/*
 * Unconditional wait.  We release the mutex while we're out waiting
 * on the back porch and reacquire it when we come back it.
 */
void 
condition_wait(condition_t c, mutex_t m)
{
    spin_lock_yield(&c->waiters.lock);
    mutex_unlock(m);
    atomic_enqueue_and_block( &c->waiters, &c->waiters.lock );
    mutex_lock(m);
}


/*
 * Signal the condition.  Nothing happens if nobody is waiting
 */
void 
condition_signal(condition_t c)
{
    cthread_t t;

    spin_lock_yield(&c->waiters.lock);
    if (!queue_empty(&c->waiters)) {
	t = cthread_dequeue(&c->waiters);
	spin_unlock(&c->waiters.lock);
	CRITICAL_SECTION(&readyq.lock, cthread_enqueue(&readyq,t) );
    } else {
	spin_unlock(&c->waiters.lock);
    }
} 


/*
 * Same as condition_signal but we wake EVERYONE waiting 
 */
void 
condition_broadcast(condition_t c)
{
    cthread_t t;

    spin_lock_yield(&c->waiters.lock);
    while (!queue_empty(&c->waiters)) {
	t = cthread_dequeue(&c->waiters);
	spin_unlock(&c->waiters.lock);
	CRITICAL_SECTION( &readyq.lock, cthread_enqueue(&readyq,t) );
	spin_lock_yield(&c->waiters.lock);
    }
    spin_unlock(&c->waiters.lock);
} 
