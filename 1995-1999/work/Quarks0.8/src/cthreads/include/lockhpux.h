#ifndef _LOCK_
#define _LOCK_

typedef struct spin_lock_t {
	int _ldcws[4];
} spin_lock_t;

/*
 * we'll always assign a block of 16 bytes to the spinlock and then assume
 * that the real lock is the one aligned on a 16 byte boundary
 */
#define	_align_spin_lock(s) \
  ((int *)(((int) (s) + sizeof(spin_lock_t) - 1) & ~(sizeof(spin_lock_t) - 1)))

#define spin_lock_init(s) (*_align_spin_lock(s) = -1)
#define spin_lock_locked(s) (*_align_spin_lock(s) == 0)

     
/* Operations implemented as functions */
extern int spin_try_lock( spin_lock_t *lock );
extern void spin_unlock( spin_lock_t *lock );

     /* Operations Implemented as macros */
#define spin_lock(the_lock) { while(! spin_try_lock(the_lock)) ; }

/* This one is like spin_lock but yields the processor between trys */
#define spin_lock_yield(the_lock) { \
  while(! spin_try_lock(the_lock)) \
      cthread_yield(); \
}

#endif _LOCK_
