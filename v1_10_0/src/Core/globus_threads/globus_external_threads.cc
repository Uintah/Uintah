/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.

  Description:

  Bindings for the Globus threads package , to be used when Globus has been
  configured to use the sci thread library.
*/

#include "globus_common.h"
#include "globus_thread_common.h"
#include "globus_hashtable.h"
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <string.h>

#define GLOBUS_L_LIBC_MAX_ERR_SIZE 512

using namespace SCIRun;

#define globus_i_thread_test_rc( a, b )				\
    do								\
    {                                                   	\
	if( a != GLOBUS_SUCCESS && a != EINTR )			\
	{							\
	    globus_i_thread_report_bad_rc( a, b );		\
	}							\
	else							\
	{							\
	    a = GLOBUS_SUCCESS;;				\
	}							\
    } while(0)

void
globus_i_thread_report_bad_rc(int rc,
			      char *message );
extern "C" int globus_i_thread_ignore_sigpipe(void);

#define MAXKEYS 10
#define GLOBUS_L_KEY_HASHTABLE_SIZE 65

class GlobusThread;

static Mutex thread_pool_mutex("Globus thread pool lock");
#define MAXPOOLTHREADS 10
static GlobusThread* thread_pool[MAXPOOLTHREADS];
static int npoolthreads;

static Mutex thread_key_mutex("Globus thread key lock");
static int next_key;
static globus_hashtable_t global_keys[MAXKEYS];
static globus_thread_key_destructor_func_t key_table_destructor_funcs[MAXKEYS];

class GlobusThread : public Runnable {
public:
    Semaphore wait;
    GlobusThread();
    virtual ~GlobusThread();
    void* threadSpecific[MAXKEYS];
    globus_thread_func_t user_func;
    void* user_arg;
    Thread* thread;
    virtual void run();
};

GlobusThread::GlobusThread()
    : wait("Globus idle thread wait", 0)
{
    for(int i=0;i<MAXKEYS;i++)
	threadSpecific[i]=0;
}

GlobusThread::~GlobusThread()
{
}

void GlobusThread::run()
{
  for(;;){
    wait.down();
    (*user_func)(user_arg);
    if(npoolthreads >= MAXPOOLTHREADS)
      break;
    thread_pool_mutex.lock();
    if(npoolthreads >= MAXPOOLTHREADS){
      thread_pool_mutex.unlock();
      break;
    } else {
      thread_pool[npoolthreads++]=this;
      thread_pool_mutex.unlock();
      /* Call the thread specific data destructors */
      for (int i = 0; i < MAXKEYS; i++){
	globus_thread_key_destructor_func_t func = 
	  key_table_destructor_funcs[i];
	void* value = threadSpecific[i];
	if (func && value){
	  (*func)(value);
	}
	threadSpecific[i]=0;
      }
    }
  }
}

/*
 * globus_i_thread_id()
 *
 * Set *Thread_ID (int *) to be the thread id of the calling thread.
 */
extern "C" void
globus_i_thread_id(globus_thread_t *Thread_ID)
{ 
    *Thread_ID = (globus_thread_t)Thread::self();
}


#ifndef GLOBUS_L_THREAD_DEFAULT_STACK_SIZE
#define GLOBUS_L_THREAD_DEFAULT_STACK_SIZE 0
#endif

//static globus_bool_t	arg_got_stack_size;
//static long		arg_stack_size;
static long     	stack_size;
static globus_bool_t    globus_l_thread_already_initialized = GLOBUS_FALSE;

/* For globus_thread_once() */
static Mutex globus_l_thread_once_mutex("Globus thread_once lock");

static int globus_l_thread_activate(void);
static int globus_l_thread_deactivate(void);

globus_module_descriptor_t globus_i_thread_module =
{
    "globus_thread",
    globus_l_thread_activate,
    globus_l_thread_deactivate,
    GLOBUS_NULL,
    GLOBUS_NULL
};
/*
 * globus_i_thread_pre_activate()
 *
 * If you need to call a thread package initialization routine, then
 * do it here.  
 */
extern "C" int
globus_i_thread_pre_activate( void )
{
    globus_mutex_init(&globus_libc_mutex, GLOBUS_NULL);
    for(int i=0;i<MAXKEYS;i++){
	if(globus_hashtable_init(&global_keys[i], GLOBUS_L_KEY_HASHTABLE_SIZE,
				 globus_hashtable_voidp_hash,
				 globus_hashtable_voidp_keyeq) != GLOBUS_SUCCESS)
	    return GLOBUS_FAILURE;
	key_table_destructor_funcs[i]=0;
    }
    
    return globus_i_thread_ignore_sigpipe();
} /* globus_l_thread_pre_activate() */


/*
 * globus_l_thread_activate()
 */
static int
globus_l_thread_activate()
{
    globus_module_activate(GLOBUS_THREAD_COMMON_MODULE);

    if(globus_l_thread_already_initialized)
    {
	return GLOBUS_SUCCESS;
    }

    globus_l_thread_already_initialized = GLOBUS_TRUE;

    stack_size = (long) GLOBUS_L_THREAD_DEFAULT_STACK_SIZE;
    //    arg_got_stack_size = GLOBUS_FALSE;

    globus_thread_set_diagnostics_file(GLOBUS_NULL);

    return GLOBUS_SUCCESS;
} /* globus_l_thread_activate() */


/*
 * globus_l_thread_deactivate()
 */
static int globus_l_thread_deactivate(void)
{
    int rc;
    
    rc = globus_module_deactivate(GLOBUS_THREAD_COMMON_MODULE);
    return rc;
} /* globus_l_thread_deactivate() */

/*
 * globus_thread_exit()
 */
extern "C" void globus_thread_exit( void */*status*/ )
{
    /* Call the thread specific data destructors */
    for (int i = 0; i < MAXKEYS; i++)
    {
	globus_thread_key_destructor_func_t func = key_table_destructor_funcs[i];
	void* value = globus_thread_getspecific(i);
	if (func && value)
	{
	    (*func)(value);
	}
    }
    Thread::exit();
} /* globus_thread_exit() */

static GlobusThread* get_thread()
{
  if(npoolthreads){
    thread_pool_mutex.lock();
    if(npoolthreads){
      GlobusThread* t=thread_pool[--npoolthreads];
      thread_pool_mutex.unlock();
      return t;
    } else {
      thread_pool_mutex.unlock();
    }
  }
  GlobusThread* thread=new GlobusThread();
  Thread* t=new Thread(thread, "Globus Thread");
  t->detach();
  t->setDaemon(true);
  thread->thread=t;
  return thread;
}

/*
 * globus_thread_create
 */
extern "C" int
globus_thread_create(globus_thread_t *user_thread,
		     globus_threadattr_t *attr,
		     globus_thread_func_t func,
		     void *user_arg )
{
  size_t stacksize;
  GlobusThread* thread=get_thread();
  thread->user_func = func;
  thread->user_arg = user_arg;

  if (attr) {
    globus_threadattr_getstacksize(attr, &stacksize);
  } else{
    stacksize = ::stack_size;
  }
  thread->wait.up();

  if(user_thread){
    *user_thread = (Core_Thread_Thread*)thread->thread;
  }

  return (0);
} /* globus_thread_create() */


/*
 * globus_preemptive_threads
 *
 * Return GLOBUS_TRUE (non-zero) if we are using preemptive threads.
 */
extern "C" globus_bool_t
globus_preemptive_threads(void)
{
  return GLOBUS_TRUE;
} /* globus_preemptive_threads() */


/*
 * globus_threadattr_init()
 */
#undef globus_threadattr_init
extern "C" int
globus_threadattr_init(globus_threadattr_t *attr)
{
  attr->stacksize=stack_size;
  return GLOBUS_SUCCESS;
}

/*
 * globus_threadattr_destroy()
 */
#undef globus_threadattr_destroy
extern "C" int
globus_thread_destroy(globus_threadattr_t */*attr*/)
{
  return GLOBUS_SUCCESS;
}

/*
 * globus_threadattr_setstacksize()
 */
#undef globus_threadattr_setstacksize
extern "C" int
globus_threadattr_setstacksize(globus_threadattr_t *attr, 
			       size_t stacksize)
{
  attr->stacksize=stacksize;
  return GLOBUS_SUCCESS;
}

/*
 * globus_threadattr_getstacksize()
 */
#undef globus_threadattr_getstacksize
extern "C" int
globus_threadattr_getstacksize(globus_threadattr_t *attr,
			       size_t *stacksize)
{
  *stacksize=attr->stacksize;
  return GLOBUS_SUCCESS;
}

/*
 * globus_thread_key_create()
 */
#undef globus_thread_key_create
extern "C" int
globus_thread_key_create(globus_thread_key_t *key,
			 globus_thread_key_destructor_func_t func)
{
  thread_key_mutex.lock();
  int k = next_key++;
  if(next_key >= MAXKEYS){
    thread_key_mutex.unlock();
    int rc= GLOBUS_FAILURE;
    globus_i_thread_test_rc( rc, "GLOBUS_THREAD: keycreate failed\n" );
    return rc;
  }
  key_table_destructor_funcs[k]=func;
  *key=k;
  thread_key_mutex.unlock();
  return GLOBUS_SUCCESS;
} /* globus_thrad_key_create() */


/*
 * globus_thread_setspecific()
 */
#undef globus_thread_setspecific
extern "C" int
globus_thread_setspecific(globus_thread_key_t key,
			  void *value)
{
  int rc=0;
  if ( key < 0 || key >= MAXKEYS) {
    rc=GLOBUS_FAILURE;
  } else {
    Thread* t=Thread::self();
    if(t){
      //fprintf(stderr, "setspecific: key=%d, thread=%x, value=%x\n", key, t, value);
      Runnable* runnable = t->getRunnable();
      GlobusThread* gt=dynamic_cast<GlobusThread*>(runnable);
      if(!gt) {
	thread_key_mutex.lock();
	if(globus_hashtable_insert(&global_keys[key], t, value) != GLOBUS_SUCCESS){
	  if(globus_hashtable_remove(&global_keys[key], t) == 0){
	    rc=GLOBUS_FAILURE;
	  } else {
	    if(globus_hashtable_insert(&global_keys[key], t, value) != GLOBUS_SUCCESS){
	      rc=GLOBUS_FAILURE;
	    }
	  }
	} else {
	  // Success..
	}
	thread_key_mutex.unlock();
      } else {
	gt->threadSpecific[key]=value;
      }
    } else {
      fprintf(stderr, "t=0?\n");
      rc=GLOBUS_FAILURE;
    }
  }
  globus_i_thread_test_rc(rc, "GLOBUS_THREAD: set specific failed\n");
  return rc;
} /* globus_thread_setspecific() */
	
/*
 * globus_thread_getspecific()
 */
#undef globus_thread_getspecific
extern "C" void *
globus_thread_getspecific(globus_thread_key_t key)
{
  int rc=0;
  if ( key < 0 || key >= MAXKEYS) {
    rc=GLOBUS_FAILURE;
  } else {
    Thread* t=Thread::self();
    Runnable* runnable = t->getRunnable();
    GlobusThread* gt=dynamic_cast<GlobusThread*>(runnable);
    if(!gt) {
      thread_key_mutex.lock();
      void* data=globus_hashtable_lookup(&global_keys[key], t);
      thread_key_mutex.unlock();
      //fprintf(stderr, "1. getspecific: key=%d, thread=%x, value=%x\n", key, t, data);
      return data;
    } else {
      //fprintf(stderr, "2. getspecific: key=%d, thread=%x, value=%x\n", key, t, gt->threadSpecific[key]);
      return gt->threadSpecific[key];
    }
  }
  globus_i_thread_test_rc(rc, "GLOBUS_THREAD: set specific failed\n");
  return (void*)-1;
} /* globus_thread_getspecific() */

/*
 * globus_thread_self
 */
#undef globus_thread_self
extern "C" globus_thread_t
globus_thread_self( void )
{
  return (globus_thread_t)Thread::self();
}

/*
 * globus_thread_equal()
 */
#undef globus_thread_equal
extern "C" int
globus_thread_equal(globus_thread_t t1,
		    globus_thread_t t2)
{
  return t1 == t2;
} /* globus_thread_equal() */

/*
 * globus_thread_yield
 */
#undef globus_thread_yield
extern "C" void
globus_thread_yield( void )
{
  Thread::yield();
}

/*
 * globus_i_am_only_thread()
 */
#undef globus_i_am_only_thread
extern "C" globus_bool_t
globus_i_am_only_thread(void)
{
  return GLOBUS_FALSE;
}

#undef globus_mutex_init
extern "C" int
globus_mutex_init(globus_mutex_t *mut,
		  globus_mutexattr_t */*attr*/ )
{
  try {
    *mut=(globus_mutex_t)new Mutex("Globus mutex");
  } catch (...) {
    return GLOBUS_FAILURE;
  }
  return GLOBUS_SUCCESS;
}

/*
 *  globus_mutex_destroy()
 */
#undef globus_mutex_destroy
extern "C" int
globus_mutex_destroy( globus_mutex_t *mut )
{
  delete *(Mutex**)mut;
  return GLOBUS_SUCCESS;
}

/*
 * globus_cond_init()
 */
#undef globus_cond_init
extern "C" int
globus_cond_init(globus_cond_t *cv,
		 globus_condattr_t */*attr*/ )
{
  *cv=(globus_cond_t)new ConditionVariable("Globus Condition Variable");
  return GLOBUS_SUCCESS;
}

/*
 *  globus_cond_destroy()
 */
#undef globus_cond_destroy
extern "C" int
globus_cond_destroy(globus_cond_t *cv)
{
  delete *(ConditionVariable**)cv;
  return GLOBUS_SUCCESS;
}

/* 
 *  globus_mutex_lock()
 */
#undef globus_mutex_lock
extern "C" int
globus_mutex_lock( globus_mutex_t *mut )
{
  (*(Mutex**)mut)->lock();
  return GLOBUS_SUCCESS;
}


/* 
 *  globus_mutex_trylock()
 */
#undef globus_mutex_trylock
extern "C" int
globus_mutex_trylock(globus_mutex_t *mut)
{
  if((*(Mutex**)mut)->tryLock())
    return GLOBUS_SUCCESS;
  else
    return GLOBUS_FAILURE;
} /* globus_mutex_trylock() */


/*
 *  globus_mutex_unlock()
 */
#undef globus_mutex_unlock
extern "C" int
globus_mutex_unlock( globus_mutex_t *mut )
{
  (*(Mutex**)mut)->unlock();
  return GLOBUS_SUCCESS;
}

/*
 *  globus_cond_wait()
 */
#undef globus_cond_wait
extern "C" int
globus_cond_wait(globus_cond_t *cv,
		 globus_mutex_t *mut )
{
  globus_thread_blocking_will_block();
  (*(ConditionVariable**)cv)->wait(**(Mutex**)mut);
  return GLOBUS_SUCCESS;
}

/*
 *  globus_cond_timedwait()
 */
#undef globus_cond_timedwait
extern "C" int
globus_cond_timedwait(globus_cond_t *cv,
		      globus_mutex_t *mut,
		      globus_abstime_t *time)
{
  globus_thread_blocking_will_block();

  struct timespec at;
  at.tv_sec = time->tv_sec;
  at.tv_nsec = time->tv_nsec;
  if((*(ConditionVariable**)cv)->timedWait(**(Mutex**)mut, &at))
    return GLOBUS_SUCCESS;
  else
    return ETIMEDOUT;
}

/*
 *  globus_cond_signal()
 */
#undef globus_cond_signal
extern "C" int
globus_cond_signal( globus_cond_t *cv )
{
  (*(ConditionVariable**)cv)->conditionSignal();
  return GLOBUS_SUCCESS;
}

/*
 *  globus_cond_broadcast()
 */
#undef globus_cond_broadcast
extern "C" int
globus_cond_broadcast( globus_cond_t *cv )
{
  (*(ConditionVariable**)cv)->conditionBroadcast();
  return GLOBUS_SUCCESS;
}


/*
 * globus_i_thread_actual_thread_once()
 */
extern "C" int
globus_i_thread_actual_thread_once(globus_thread_once_t *once_control,
				       void (*init_routine)(void))
{
  int rc;
  globus_l_thread_once_mutex.lock();
  if (*once_control){
    /* Someone beat us to it.  */
    rc = 0;
  } else {
    /* We're the first one here */
    (*init_routine)();
    *once_control = 1;
    rc = 0;
  }
  globus_l_thread_once_mutex.unlock();
  return (rc);
} /* globus_i_thread_actual_thread_once() */

#undef globus_thread_once
extern "C" int
globus_thread_once(globus_thread_once_t *once_control,
		   void (*init_routine)(void))
{
  return (globus_i_thread_actual_thread_once(once_control, init_routine));
}

extern "C" void
globus_thread_prefork(void)
{
/* Do nothing */
}

extern "C" void
globus_thread_postfork(void)
{
/* Do nothing */
}

/*
 * globus_i_thread_report_bad_rc()
 */
void
globus_i_thread_report_bad_rc(int rc,
			      char *message )
{
  char achMessHead[] = "[Thread System]";
  char achDesc[GLOBUS_L_LIBC_MAX_ERR_SIZE];
    
  if(rc != GLOBUS_SUCCESS) {
    switch( rc ) {
    case EAGAIN:
      strcpy(achDesc, "system out of resources (EAGAIN)");
      break;
    case ENOMEM:
      strcpy(achDesc, "insufficient memory (ENOMEM)");
      break;
    case EINVAL:
      strcpy(achDesc, "invalid value passed to thread interface (EINVAL)");
      break;
    case EPERM:
      strcpy(achDesc, "user does not have adequate permission (EPERM)");
      break;
    case ERANGE:
      strcpy(achDesc, "a parameter has an invalid value (ERANGE)");
      break;
    case EBUSY:
      strcpy(achDesc, "mutex is locked (EBUSY)");
      break;
    case EDEADLK:
      strcpy(achDesc, "deadlock detected (EDEADLK)");
      break;
    case ESRCH:
      strcpy(achDesc, "could not find specified thread (ESRCH)");
      break;
    default:
      globus_fatal("%s %s\n%s unknown error number: %d\n",
		   achMessHead, message, achMessHead, rc);
      break;
    }
    globus_fatal("%s %s\n%s %s",
		 achMessHead, message, achMessHead, achDesc);
  }
} /* globus_i_thread_report_bad_rc() */

