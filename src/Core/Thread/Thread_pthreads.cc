
#include "Barrier.h"
#include "ConditionVariable.h"
#include "Mutex.h"
#include "Semaphore.h"
#include "ThreadGroup.h"
#include "Thread.h"
#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include <semaphore.h>
#include <pthread.h>
};
#include <errno.h>

#define NF cerr << "Not finished: " << __FILE__ << ": " << __LINE__ << '\n';


#define MAXBSTACK 10
#define MAXTHREADS 4000
#define N_POOLMUTEX 301

// Thread states
#define STATE_STARTUP 1
#define STATE_RUNNING 2
#define STATE_IDLE 3
#define STATE_SHUTDOWN 4
#define STATE_BLOCK_SEMAPHORE 5
#define STATE_PROGRAM_EXIT 6
#define STATE_JOINING 7
#define STATE_BLOCK_MUTEX 8
#define STATE_BLOCK_ANY 9
#define STATE_DIED 10
#define STATE_BLOCK_POOLMUTEX 11

struct Thread_private {
    Thread* thread;
    pthread_t threadid;
    int state;
    int bstacksize;
    const char* blockstack[MAXBSTACK];
    sem_t done;
    sem_t delete_ready;
    bool detached;
};

static Thread_private* active[MAXTHREADS];
static int numActive;
static bool initialized;
static pthread_mutex_t sched_lock;
static pthread_key_t thread_key;
static sem_t main_sema;

static void lock_scheduler() {
    if(pthread_mutex_lock(&sched_lock)){
	perror("pthread_mutex_lock");
	Thread::niceAbort();
    }
}

static void unlock_scheduler() {
    if(pthread_mutex_unlock(&sched_lock)){
	perror("pthread_mutex_unlock");
	Thread::niceAbort();
    }
}

static int push_bstack(Thread_private* p, int state, const char* name) {
    int oldstate=p->state;
    p->state=state;
    p->blockstack[p->bstacksize]=name;
    p->bstacksize++;
    if(p->bstacksize>MAXBSTACK){
	fprintf(stderr, "Blockstack Overflow!\n");
	Thread::niceAbort();
    }
    return oldstate;
}

static void pop_bstack(Thread_private* p, int oldstate) {
    p->bstacksize--;
    p->state=oldstate;
}

// This is only used for the single processor implementation
struct Barrier_private {
    Mutex mutex;
    ConditionVariable cond0;
    ConditionVariable cond1;
    int cc;
    int nwait;
    Barrier_private();
};

Barrier_private::Barrier_private()
: mutex("Barrier lock"),
  cond0("Barrier condition 0"), cond1("Barrier condition 1"),
  cc(0), nwait(0)
{
}   

Barrier::Barrier(const std::string& name, int numThreads)
 : d_name(name), d_numThreads(numThreads), d_threadGroup(0)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=new Barrier_private;
}

Barrier::Barrier(const std::string& name, ThreadGroup* threadGroup)
 : d_name(name), d_numThreads(0), d_threadGroup(threadGroup)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=new Barrier_private;
}

Barrier::~Barrier()
{
    delete d_priv;
}

void Barrier::wait()
{
    int n=d_threadGroup?d_threadGroup->numActive(true):d_numThreads;
    Thread_private* p=Thread::currentThread()->d_priv;
    int oldstate=push_bstack(p, STATE_BLOCK_SEMAPHORE, d_name.c_str());
    d_priv->mutex.lock();
    ConditionVariable& cond=d_priv->cc?d_priv->cond0:d_priv->cond1;
    /*int me=*/d_priv->nwait++;
    if(d_priv->nwait == n){
	// Wake everybody up...
	d_priv->nwait=0;
	d_priv->cc=1-d_priv->cc;
	cond.conditionBroadcast();
    } else {
	cond.wait(d_priv->mutex);
    }
    d_priv->mutex.unlock();
    pop_bstack(p, oldstate);
}

struct Mutex_private {
    pthread_mutex_t mutex;
};

Mutex::Mutex(const std::string& name)
    : d_name(name)
{
    d_priv=new Mutex_private;
    if(pthread_mutex_init(&d_priv->mutex, NULL) != 0){
	perror("pthread_mutex_init");
	Thread::niceAbort();
    }
}

Mutex::~Mutex()
{
    if(pthread_mutex_destroy(&d_priv->mutex) != 0){
	perror("pthread_mutex_destroy");
	Thread::niceAbort();
    }
    delete d_priv;
}

void Mutex::unlock()
{
    if(pthread_mutex_unlock(&d_priv->mutex) != 0){
	perror("pthread_mutex_unlock");
	Thread::niceAbort();
    }
}

void Mutex::lock()
{
    if(pthread_mutex_lock(&d_priv->mutex) != 0){
	perror("pthread_mutex_lock");
	Thread::niceAbort();
    }
}

bool Mutex::tryLock()
{
    if(pthread_mutex_trylock(&d_priv->mutex) != 0){
	if(errno == EAGAIN || errno == EINTR)
	    return false;
	perror("pthread_mutex_trylock");
	Thread::niceAbort();
    }
    return true;
}

struct Semaphore_private {
    sem_t sem;
};

Semaphore::Semaphore(const std::string& name, int value)
    : d_name(name)
{
    d_priv=new Semaphore_private;
    if(sem_init(&d_priv->sem, 0, value) != 0){
	perror("sem_init");
	Thread::niceAbort();
    }
}
    
Semaphore::~Semaphore()
{
    if(sem_destroy(&d_priv->sem) != 0){
	perror("sem_destroy");
	Thread::niceAbort();
    }
    delete d_priv;
}

void Semaphore::up()
{
    if(sem_post(&d_priv->sem) != 0){
	perror("sem_post");
	Thread::niceAbort();
    }
}

void Semaphore::down()
{
    if(sem_wait(&d_priv->sem) != 0){
	perror("sem_wait");
	Thread::niceAbort();
    }
}

bool Semaphore::tryDown()
{
    if(sem_trywait(&d_priv->sem) != 0){
	if(errno == EAGAIN)
	    return false;
	perror("sem_trywait");
	Thread::niceAbort();
    }
    return true;
}

void ThreadGroup::gangSchedule()
{
    NF
}

void Thread::checkExit()
{
    lock_scheduler();
    int done=true;
    for(int i=0;i<numActive;i++){
	Thread_private* p=active[i];
	if(!p->thread->isDaemon()){
	    done=false;
	    break;
	}
    }
    unlock_scheduler();

    if(done)
	Thread::exitAll(0);
}

Thread* Thread::currentThread()
{
    void* p=pthread_getspecific(thread_key);
    if(!p){
	perror("pthread_getspecific");
	Thread::niceAbort();
    }
    return (Thread*)p;
}

void Thread::join()
{
    Thread* us=Thread::currentThread();
    int os=push_bstack(us->d_priv, STATE_JOINING, d_threadname.c_str());
    if(sem_wait(&d_priv->done) != 0){
	perror("sem_wait");
	Thread::niceAbort();
    }
    pop_bstack(us->d_priv, os);
    detach();
}

//void Thread::profile(FILE*, FILE*) { NF }
int Thread::numProcessors()
{
    return 1;
}

void Thread_shutdown(Thread* thread)
{
    Thread_private* priv=thread->d_priv;

    if(sem_post(&priv->done) != 0){
	perror("sem_post");
	Thread::niceAbort();
    }

    delete thread;

    // Wait to be deleted...
    if(sem_wait(&priv->delete_ready) == -1) {
	perror("sem_wait");
	Thread::niceAbort();
    }

    // Allow this thread to run anywhere...
    if(thread->d_cpu != -1)
	thread->migrate(-1);

    priv->thread=0;
    lock_scheduler();
    /* Remove it from the active queue */
    int i;
    for(i=0;i<numActive;i++){
	if(active[i]==priv)
	    break;
    }
    for(i++;i<numActive;i++){
	active[i-1]=active[i];
    }
    numActive--;
    unlock_scheduler();
    Thread::checkExit();
    if(priv->threadid == 0){
	priv->state=STATE_PROGRAM_EXIT;
	if(sem_wait(&main_sema) == -1){
	    perror("sem_wait");
	    Thread::niceAbort();
	}
    }
}

void Thread_run(Thread* t)
{
    t->run_body();
}

static void* run_threads(void* priv_v)
{
    Thread_private* priv=(Thread_private*)priv_v;
    if(pthread_setspecific(thread_key, priv->thread) != 0){
	perror("pthread_setspecific");
	Thread::niceAbort();
    }
    priv->state=STATE_RUNNING;
    Thread_run(priv->thread);
    priv->state=STATE_SHUTDOWN;
    Thread_shutdown(priv->thread);
    priv->state=STATE_IDLE;
    return 0;
}

void Thread::os_start(bool)
{
    if(!initialized){
	Thread::initialize();
    }

    d_priv=new Thread_private;

    if(sem_init(&d_priv->done, 0, 0) != 0){
	perror("sem_init");
	Thread::niceAbort();
    }
    if(sem_init(&d_priv->delete_ready, 0, 0) != 0){
	perror("sem_init");
	Thread::niceAbort();
    }
    d_priv->state=STATE_STARTUP;
    d_priv->bstacksize=0;
    d_priv->detached=0;

    d_priv->thread=this;
    d_priv->threadid=0;
    lock_scheduler();
    if(pthread_create(&d_priv->threadid, NULL, run_threads, d_priv) != 0){
	perror("pthread_create");
	Thread::niceAbort();
    }
    active[numActive]=d_priv;
    numActive++;
    unlock_scheduler();
}

void Thread::stop()
{
    NF
}

void Thread::resume()
{
    NF
}

int Thread::couldBlock(const std::string& why)
{
    Thread_private* p=Thread::currentThread()->d_priv;
    return push_bstack(p, STATE_BLOCK_ANY, why.c_str());
}

void Thread::couldBlockDone(int restore)
{
    Thread_private* p=Thread::currentThread()->d_priv;
    pop_bstack(p, restore);
}

void Thread::detach()
{
    if(sem_post(&d_priv->delete_ready) != 0){
	perror("sem_post");
	Thread::niceAbort();
    }
    d_priv->detached=1;
    if(pthread_detach(d_priv->threadid) != 0){
	perror("pthread_detach");
	Thread::niceAbort();
    }
}

void Thread::setPriority(int)
{
    NF
}

void Thread::exitAll(int)
{
    NF
}

void Thread::initialize()
{
    if(pthread_mutex_init(&sched_lock, NULL) != 0){
	perror("pthread_mutex_init");
	Thread::niceAbort();
    }

    if(pthread_key_create(&thread_key, NULL) != 0){
	perror("pthread_key_create");
	Thread::niceAbort();
    }

    ThreadGroup::s_defaultGroup=new ThreadGroup("default group", 0);
    Thread* mainthread=new Thread(ThreadGroup::s_defaultGroup, "main");
    mainthread->d_priv=new Thread_private;
    mainthread->d_priv->thread=mainthread;
    mainthread->d_priv->state=STATE_RUNNING;
    mainthread->d_priv->bstacksize=0;
    if(pthread_setspecific(thread_key, mainthread) != 0){
	perror("pthread_setspecific");
	Thread::niceAbort();
    }
    if(sem_init(&mainthread->d_priv->done, 0, 0) != 0){
	perror("sem_init");
	Thread::niceAbort();
    }
    if(sem_init(&mainthread->d_priv->delete_ready, 0, 0) != 0){
	perror("sem_init");
	Thread::niceAbort();
    }
    if(sem_init(&main_sema, 0, 0) != 0){
	perror("sem_init");
	Thread::niceAbort();
    }
    lock_scheduler();
    active[numActive]=mainthread->d_priv;
    numActive++;
    unlock_scheduler();

    //install_signal_handlers();

    initialized=1;
}

void Thread::yield()
{
    sched_yield();
}

void Thread::migrate(int proc)
{
    // Nothing for now...
}



#include "WorkQueue.h"
#include "Thread.h"
#include <iostream.h>
#include "AtomicCounter.h"
#include <stdio.h>

/*
 * Doles out work assignment to various worker threads.  Simple
 * attempts are made at evenly distributing the workload.
 * Initially, assignments are relatively large, and will get smaller
 * towards the end in an effort to equalize the total effort.
 */
struct WorkQueue_private {
    WorkQueue_private();
    AtomicCounter counter;
};

WorkQueue_private::WorkQueue_private()
    : counter("WorkQueue counter", 0)
{
}

void WorkQueue::init()
{
    d_priv->counter.set(0);
    fill();
}

WorkQueue::WorkQueue(const std::string& name, int totalAssignments,
		     int nthreads, bool dynamic, int granularity)
    : d_name(name), d_numThreads(nthreads),
      d_totalAssignments(totalAssignments), d_granularity(granularity),
      d_assignments(0), d_dynamic(dynamic)
{
    if(!initialized){
	Thread::initialize();
    }
    d_priv=new WorkQueue_private();
    init();
}

WorkQueue::WorkQueue(const std::string& name)
    : d_name(name), d_assignments(0)
{
    d_totalAssignments=0;
    d_priv=0;
}

WorkQueue::~WorkQueue()
{
    if(d_priv){
	delete d_priv;
    }
}

bool WorkQueue::nextAssignment(int& start, int& end)
{
    int i=d_priv->counter++;
    if(i >= (int)d_assignments.size())
	return false;
    start=d_assignments[i];
    end=d_assignments[i+1];
    return true;
}

void WorkQueue::refill(int new_ta, int new_numThreads,
		       bool new_dynamic, int new_granularity)
{
    if(new_ta == d_totalAssignments && new_numThreads == d_numThreads
       && new_dynamic == d_dynamic && new_granularity == d_granularity){
	d_priv->counter.set(0);
    } else {
	d_totalAssignments=new_ta;
	d_numThreads=new_numThreads;
	d_dynamic=new_dynamic;
	d_granularity=new_granularity;
	init();
    }
}
