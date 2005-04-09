
/*
 *  Task_irix.cc: Task implementation for Solaris 2.3 and above.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <Lib/Args.h>
#include <iostream.h>
#include <stdlib.h>
#include <synch.h>
#include <thread.h>
#include <unistd.h>

struct TaskPrivate {
    thread_t tid;
    int retval;
};

struct TaskArgs {
    int arg;
    Task* t;
};

// Register arguments
static Arg_flag multi_threaded("multithreaded", "mt");
static Arg_name concurrency("concurrency", "-1");
static Arg_name nproc_arg("nprocessors", "-1");

void* runbody(void* vargs)
{
    TaskArgs* args=(TaskArgs*)vargs;
    int arg=args->arg;
    Task* t=args->t;
    delete args;

    t->startup(arg);
    return 0; // Never reached...
}

int Task::startup(int task_arg)
{
    thr_setprio(priv->tid, priority);
    int retval=body(task_arg);
    Task::taskexit(this, retval);
    return 0; // Never reached.
}

void Task::taskexit(Task*, int retval)
{
    // Should do something with the retval
    thr_exit((void*)retval);
} 

void Task::activate(int task_arg)
{
    if(activated){
	cerr << "Error: task is being activated twice!" << endl;
	exit(-1);
    }
    priv=new TaskPrivate;
    if(multi_threaded.is_set()){
	TaskArgs* args=new TaskArgs;
	args->arg=task_arg;
	args->t=this;
	if(thr_create(NULL, 0, runbody, (void*)args,
		      detached?THR_DETACHED:0, &priv->tid) != 0){
	   perror("thr_create");
	   exit(-1);
       }
    } else {
	priv->retval=body(task_arg);
    }
}

void TaskManager::main_exit()
{
    thr_exit(0);
}

void Task::yield()
{
    thr_yield();
}

void TaskManager::initialize()
{
    if(multi_threaded.is_set()){
	int nproc;
	if(!concurrency.name().get_int(nproc)){
	    cerr << "concurrency must be an integer" << endl;
	    exit(-1);
	}
	if(nproc==-1){
	    nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
	}
	thr_setconcurrency(nproc);
	cerr << "Configured for " << nproc << " processors\n";
    }
}

int TaskManager::nprocessors()
{
    static int nproc=-1;
    if(nproc==-1){
	if(multi_threaded.is_set()){
	    if(!nproc_arg.name().get_int(nproc)){
		cerr << "concurrency must be an integer" << endl;
		exit(-1);
	    }
	    if(nproc==-1){
		nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
	    }
	} else {
	    nproc=1;
	}
    }
    return nproc;
}

int Task::wait_for_task(Task* task)
{
    int rv;
    if(multi_threaded.is_set()){
	if(task->detached){
	    cerr << "Cannot wait for a detached task!\n";
	    return 0;
	}
	void* retval;
	if(thr_join(task->priv->tid, 0, &retval) != 0){
	    perror("thr_hoin");
	    exit(-1);
	}
	rv=(int)retval;
    } else {
	rv=task->priv->retval;
    }
    return rv;
}

//
// Mutex implementation
//

struct Mutex_private {
    mutex_t mutex;
};

Mutex::Mutex(int always)
{
    if(always || multi_threaded.is_set()){
	priv=new Mutex_private;
	if(mutex_init(&priv->mutex, USYNC_THREAD, 0) != 0){
	    perror("mutex_init");
	    exit(-1);	
	}
    } else {
	priv=0;
    }
}

Mutex::~Mutex()
{
    if(priv){
	if(mutex_destroy(&priv->mutex) != 0){
	    perror("mutex_destroy");
	    exit(-1);
	}
	delete priv;
    }
}

void Mutex::lock()
{
    if(multi_threaded.is_set()){
	if(mutex_lock(&priv->mutex) != 0){
	    perror("mutex_lock");
	    exit(-1);
	}
    }
}

void Mutex::unlock()
{
    if(multi_threaded.is_set()){
	if(mutex_unlock(&priv->mutex) != 0){
	    perror("mutex_unlock");
	    exit(-1);
	}
    }
}


//
// Semaphore implementation
//
struct Semaphore_private {
    sema_t semaphore;
};

Semaphore::Semaphore(int count)
{
    if(multi_threaded.is_set()){
	priv=new Semaphore_private;
	if(sema_init(&priv->semaphore, count, USYNC_THREAD, 0) != 0){
	    perror("sema_init");
	    exit(-1);	
	}
    } else {
	priv=0;
    }
}

Semaphore::~Semaphore()
{
    if(priv){
	if(sema_destroy(&priv->semaphore) != 0){
	    perror("sema_destroy");
	    exit(-1);
	}
	delete priv;
    }
}

void Semaphore::down()
{
    if(multi_threaded.is_set()){
	if(sema_wait(&priv->semaphore) != 0){
	    perror("sema_wait");
	    exit(-1);
	}
    }
}

void Semaphore::up()
{
    if(multi_threaded.is_set()){
	if(sema_post(&priv->semaphore) != 0){
	    perror("sema_post");
	    exit(-1);
	}
    }
}


//
// Condition Variable implementation
//
struct ConditionVariable_private {
    cond_t condition;
};

ConditionVariable::ConditionVariable()
{
    if(multi_threaded.is_set()){
	priv=new ConditionVariable_private;
	if(cond_init(&priv->condition, USYNC_THREAD, 0) != 0){
	    perror("cond_init");
	    exit(-1);
	}
    } else {
	priv=0;
    }
}

ConditionVariable::~ConditionVariable()
{
    if(priv){
	if(cond_destroy(&priv->condition) != 0){
	    perror("cond_destroy");
	    exit(-1);
	}
	delete priv;
    }
}

void ConditionVariable::wait(Mutex& mutex)
{
    if(multi_threaded.is_set()){
	if(cond_wait(&priv->condition, &mutex.priv->mutex) != 0){
	    perror("cond_wait");
	    exit(-1);
	}
    }
}

void ConditionVariable::signal()
{
    if(multi_threaded.is_set()){
	if(cond_signal(&priv->condition) != 0){
	    perror("cond_signal");
	    exit(-1);
	}
    }
}

void ConditionVariable::broadcast()
{
    if(multi_threaded.is_set()){
	if(cond_broadcast(&priv->condition) != 0){
	    perror("cond_broadcast");
	    exit(-1);
	}
    }
}

