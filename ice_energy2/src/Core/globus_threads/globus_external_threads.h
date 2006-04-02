/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *
 * External header for globus interface to sci threads
 *
 *
 */

#if !defined(GLOBUS_INCLUDE_GLOBUS_THREAD)
#define GLOBUS_INCLUDE_GLOBUS_THREAD 1

#ifndef EXTERN_C_BEGIN
#ifdef __cplusplus
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C_BEGIN
#define EXTERN_C_END
#endif
#endif


EXTERN_C_BEGIN

typedef struct timespec      globus_abstime_t;

typedef struct Core_Thread_Mutex*		globus_mutex_t;
typedef struct Core_Thread_ConditionVariable*        	globus_cond_t;
typedef int  	globus_thread_key_t;
typedef int           	globus_mutexattr_t;
typedef int           	globus_condattr_t;
typedef struct Core_Thread_Thread*	globus_thread_t;

typedef struct globus_i_threadattr_s
{
    size_t stacksize;
} globus_threadattr_t;

typedef size_t globus_thread_size;
typedef void *(*globus_thread_func_t)(void *);


#define GLOBUS_THREAD_ONCE_INIT 0
typedef int globus_thread_once_t;
extern  int globus_i_thread_actual_thread_once(
    globus_thread_once_t *once_control,
    void (*init_routine)(void));

typedef void (*globus_thread_key_destructor_func_t)(void *value);

extern void globus_i_thread_report_bad_rc( int, char * );
extern int globus_thread_create(globus_thread_t *thread,
				globus_threadattr_t *attr,
				globus_thread_func_t tar_func,
				void *user_arg );
extern void globus_thread_exit(void *status);

extern int globus_thread_key_create(globus_thread_key_t *key,
				    globus_thread_key_destructor_func_t func);
extern void *globus_thread_getspecific(globus_thread_key_t key);


#define globus_mutexattr_init(attr) 0 /* successful return */
#define globus_mutexattr_destroy(attr) 0 /* successful return */
#define globus_condattr_init(attr) 0 /* successful return */
#define globus_condattr_destroy(attr) 0 /* successful return */

extern int	globus_threadattr_init(globus_threadattr_t *attr);
extern int	globus_threadattr_destroy(globus_threadattr_t *attr);
extern int	globus_threadattr_setstacksize(globus_threadattr_t *attr,
					       size_t stacksize);
extern int	globus_threadattr_getstacksize(globus_threadattr_t *attr,
					       size_t *stacksize);

extern int	globus_thread_key_create(globus_thread_key_t *key,
				 globus_thread_key_destructor_func_t func);
extern int	globus_thread_key_delete(globus_thread_key_t key);
extern int	globus_thread_setspecific(globus_thread_key_t key,
					  void *value);
extern void *   globus_i_thread_getspecific(globus_thread_key_t key);

extern globus_thread_t	globus_thread_self(void);
extern int	globus_thread_equal(globus_thread_t t1,
				    globus_thread_t t2);
extern int	globus_thread_once(globus_thread_once_t *once_control,
				   void (*init_routine)(void));
extern void	globus_thread_yield(void);
extern globus_bool_t    globus_i_am_only_thread(void);

extern int	globus_mutex_init(globus_mutex_t *mutex,
				  globus_mutexattr_t *attr);
extern int	globus_mutex_destroy(globus_mutex_t *mutex);
extern int	globus_mutex_lock(globus_mutex_t *mutex);
extern int	globus_mutex_trylock(globus_mutex_t *mutex);
extern int	globus_mutex_unlock(globus_mutex_t *mutex);

extern int	globus_cond_init(globus_cond_t *cond,
				 globus_condattr_t *attr);
extern int	globus_cond_destroy(globus_cond_t *cond);
extern int	globus_cond_wait(globus_cond_t *cond,
				 globus_mutex_t *mutex);
extern int	globus_cond_timedwait(globus_cond_t *cond,
				      globus_mutex_t *mutex,
				      globus_abstime_t * abstime);
extern int	globus_cond_signal(globus_cond_t *cond);
extern int	globus_cond_broadcast(globus_cond_t *cond);

/******************************************************************************
			       Module definition
******************************************************************************/
extern int globus_i_thread_pre_activate();

extern globus_module_descriptor_t	globus_i_thread_module;

#define GLOBUS_THREAD_MODULE (&globus_i_thread_module)

EXTERN_C_END

#endif /* ! defined(GLOBUS_INCLUDE_GLOBUS_THREAD) */
