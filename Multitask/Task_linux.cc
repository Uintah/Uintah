
/*
 *  Task_irix.cc: Task implementation for Irix 5.0 and above.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#define __KERNEL__
#include <linux/signal.h>

#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <Classlib/Args.h>
#include <Malloc/New.h>
#include <fcntl.h>
#include <iostream.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/resource.h>

typedef struct sigcontext_struct sigcontext_t;
typedef __sighandler_t SIG_PF;

#define NOT_FINISHED(what) cerr << what << ": Not Finished " << __FILE__ << " (line " << __LINE__ << ") " << endl

#define DEFAULT_STACK_LENGTH 16*1024
#define DEFAULT_SIGNAL_STACK_LENGTH 16*1024

static int aborting=0;
static int exit_code;
static int exiting=0;
static long tick;
static int pagesize;
static int devzero_fd;
static char* progname;
static int pid;
static int skip_csw=0;

extern inline char tas(char* m)
{
        char res;

        __asm__("xchgb %0,%1":"=q" (res),"=m" (*m):"0" (0x1));
        return res;
}

void spin_lock_yield(char& lock)
{
    while(tas(&lock))
	Task::yield();
}

void spin_unlock(char& lock)
{
    lock=0;
}

class MainTask : public Task {
public:
    MainTask();
    virtual ~MainTask();
    int body(int);
};

class TaskQueue {
    Task** q;
    char lock;
    int size;
    int first;
    int last;
public:
    int nitems;
    TaskQueue(int size);
    ~TaskQueue();

    void append(Task*);

    Task* pop_first();
};

TaskQueue::TaskQueue(int size)
: size(size), first(0), last(0), nitems(0)
{
    q=new Task*[size];
    spin_unlock(lock);
}

TaskQueue::~TaskQueue()
{
    if(lock){
	cerr << "Locked task queue deleted!!!\n";
    }
    if(nitems != 0){
	cerr << "Non-empty task queue deleted!!!\n";
    }
    delete[] q;
}

void TaskQueue::append(Task* task)
{
    spin_lock_yield(lock);
    q[last++]=task;
    if(last>=size)
	last=0;
    if(last==first){
	cerr << "Task queue full!!!\n";
	exit(-1);
    }
    nitems++;
    spin_unlock(lock);
}

Task* TaskQueue::pop_first()
{
    spin_lock_yield(lock);
    if(first==last){
	cerr << "Popping an empty queue!\n";
	exit(-1);
    }
    Task* ret=q[first++];
    if(first>=size)
	first=0;
    nitems--;
    spin_unlock(lock);
    return ret;
}

struct TaskPrivate {
    int retval;
    caddr_t sp;
    caddr_t stackbot;
    size_t stacklen;
    size_t redlen;
    jmp_buf context;
};

#define MAXTASKS 100
static int ntasks=0;
static Task* tasks[MAXTASKS];
static TaskQueue readyq(MAXTASKS);
static Task* current_task=0;

// Register arguments
static Arg_flag single_threaded("singlethreaded", "Turn off multithreading");
static Arg_alias stalias(&single_threaded, "st");
static Arg_intval concurrency("concurrency", -1,
			      "Set concurrency level for multithreading");
static Arg_intval nproc_arg("nprocessors", -1,
			    "Set the number of processors\n\t\t\tDefault is actal number of processors");

// Global locks...
static char sched_lock=0;

static void makestack(TaskPrivate* priv)
{
    priv->stacklen=DEFAULT_STACK_LENGTH+pagesize;
#ifdef MMAP_IS_BROKEN
    priv->stackbot=(caddr_t)mmap(0, priv->stacklen, PROT_READ|PROT_WRITE,
				 MAP_SHARED, devzero_fd, 0);
#endif
    priv->stackbot=(caddr_t)sbrk(priv->stacklen);
    if((int)priv->stackbot == -1){
	perror("mmap");
	exit(-1);
    }
    // Now unmap the bottom part of it...
    priv->redlen=DEFAULT_STACK_LENGTH-pagesize;
    priv->sp=(char*)priv->stackbot+priv->stacklen-16;
#ifdef MPROTECT_IS_BROKEN
    if(mprotect(priv->stackbot, priv->redlen, PROT_NONE) == -1){
	perror("mprotect");
	exit(-1);
    }
#endif
}

static void lock_scheduler()
{
    if(tas(&sched_lock)){
	cerr << "Deadlock - locking scheduler while already locked!" << endl;
    }
}

static void unlock_scheduler()
{
    sched_lock=0;
}

int Task::startup(int task_arg)
{
    int retval=body(task_arg);
    Task::taskexit(this, retval);
    return 0; // Never reached.
}

// We are done..
void Task::taskexit(Task* xit, int retval)
{
    xit->priv->retval=retval;
    NOT_FINISHED("Task::taskexit");
} 

// Fork a thread
void Task::activate(int task_arg)
{
    if(activated){
	cerr << "Error: task is being activated twice!" << endl;
	exit(-1);
    }
    activated=1;
    priv=new TaskPrivate;
    if(single_threaded.is_set()){
	priv->retval=body(task_arg);
    } else {
	makestack(priv);
	lock_scheduler();
	if(setjmp(priv->context) != 0){
	    unlock_scheduler();
	    current_task->startup(task_arg);
	    cerr << "Error - startup returned!\n";
	    exit(-1);
	}
	priv->context->__sp=priv->sp;
	readyq.append(this);
	tasks[ntasks++]=this;
	unlock_scheduler();
    }
}

Task* Task::self()
{
    return current_task;
}

static char* signal_name(int sig, int code, caddr_t addr)
{
    static char buf[1000];
    switch(sig){
    case SIGHUP:
	sprintf(buf, "SIGHUP (hangup)");
	break;
    case SIGINT:
	sprintf(buf, "SIGINT (interrupt)");
	break;
    case SIGQUIT:
	sprintf(buf, "SIGQUIT (quit)");
	break;
    case SIGILL:
	sprintf(buf, "SIGILL (illegal instruction)");
	break;
    case SIGTRAP:
	sprintf(buf, "SIGTRAP (trace trap)");
	break;
    case SIGABRT:
	sprintf(buf, "SIBABRT (Abort)");
	break;
    case SIGFPE:
	sprintf(buf, "SIGFPE (Floating Point Exception)");
	break;
    case SIGKILL:
	sprintf(buf, "SIGKILL (kill)");
	break;
    case SIGBUS:
	sprintf(buf, "SIGBUS (bus error)");
	break;
    case SIGSEGV:
	{
	    char* why;
	    switch(code){
	    case EFAULT:
		why="Invalid virtual address";
		break;
	    case EACCES:
		why="Invalid permissions for address";
		break;
	    default:
		why="Unknown code!";
		break;
	    }
	    sprintf(buf, "SIGSEGV at address %p (segmentation violation - %s)",
		    addr, why);
	}
	break;
    case SIGPIPE:
	sprintf(buf, "SIGPIPE (broken pipe)");
	break;
    case SIGALRM:
	sprintf(buf, "SIGALRM (alarm clock)");
	break;
    case SIGTERM:
	sprintf(buf, "SIGTERM (killed)");
	break;
    case SIGUSR1:
	sprintf(buf, "SIGUSR1 (user defined signal 1)");
	break;
    case SIGUSR2:
	sprintf(buf, "SIGUSR2 (user defined signal 2)");
	break;
    case SIGCLD:
	sprintf(buf, "SIGCLD (death of a child)");
	break;
    case SIGPWR:
	sprintf(buf, "SIGPWR (power fail restart)");
	break;
    case SIGWINCH:
	sprintf(buf, "SIGWINCH (window size changes)");
	break;
    case SIGSTOP:
	sprintf(buf, "SIGSTOP (sendable stop signal)");
	break;
    case SIGTSTP:
	sprintf(buf, "SIGTSTP (TTY stop)");
	break;
    case SIGCONT:
	sprintf(buf, "SIGCONT (continue)");
	break;
    case SIGTTIN:
	sprintf(buf, "SIGTTIN");
	break;
    case SIGTTOU:
	sprintf(buf, "SIGTTOU");
	break;
    case SIGVTALRM:
	sprintf(buf, "SIGVTALRM (virtual time alarm)");
	break;
    case SIGPROF:
	sprintf(buf, "SIGPROF (profiling alarm)");
	break;
    default:
	sprintf(buf, "unknown signal(%d)", sig);
	break;
    }
    return buf;
}

static void handle_halt_signals(int sig, int code, sigcontext_t* context)
{
    Task* self=Task::self();
    char* tname=self?self->get_name()():"main";
    if(exiting && sig==SIGQUIT){
	fprintf(stderr, "Thread \"%s\"(pid %d) exiting...\n", tname, getpid());
	exit(exit_code);
    }
	
    // Kill all of the threads...
    char* signam=signal_name(sig, code, (caddr_t)context->eip);
    fprintf(stderr, "Thread \"%s\"(pid %d) caught signal %s.  Going down...\n", tname, getpid(), signam);
    Task::exit_all(-1);
}

static void handle_abort_signals(int sig, int code, sigcontext_t* context)
{
    cerr << "handle_abort_signals called...\n";
    if(aborting)
	exit(0);
    Task* self=Task::self();
    char* tname=self?self->get_name()():"main";
    char* signam=signal_name(sig, code, (caddr_t)context->eip);

    // See if it is a segv on the stack - if so, grow it...
    if(sig==SIGSEGV && code==EACCES
       && (caddr_t)context->eip >= self->priv->stackbot
       && (caddr_t)context->eip < self->priv->stackbot+self->priv->stacklen){
	self->priv->redlen -= pagesize;
	if(self->priv->redlen <= 0){
	    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) ran off end of stack! \n",
		    7,7,7,tname, getpid());
	    fprintf(stderr, "Stack size was %d bytes\n", self->priv->stacklen-pagesize);
	} else {
	    if(mprotect(self->priv->stackbot+self->priv->redlen, pagesize,
			PROT_READ|PROT_WRITE) == -1){
		fprintf(stderr, "Error extending stack for thread \"%s\"", tname);
		Task::exit_all(-1);
	    }
	    fprintf(stderr, "extended stack for thread %s\n", tname);
	    fprintf(stderr, "stacksize is now %d bytes\n",
		    self->priv->stacklen-self->priv->redlen);
	    return;
	}
    }

    // Ask if we should abort...
    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\ndump core? ", 7,7,7,tname, getpid(), signam);
    char buf[100];
    buf[0]='n';
    if(!fgets(buf, 100, stdin)){
	// Exit without an abort...
	Task::exit_all(-1);
    }
    if(buf[0] == 'n' || buf[0] == 'N'){
	// Exit without an abort...
	Task::exit_all(-1);
    } else {
	// Abort...
	fprintf(stderr, "Dumping core...\n");
	struct rlimit rlim;
	getrlimit(RLIMIT_CORE, &rlim);
	rlim.rlim_cur=RLIM_INFINITY;
	setrlimit(RLIMIT_CORE, &rlim);
	aborting=1;
	signal(SIGABRT, SIG_DFL); // We will dump core, but not other threads
	kill(0, SIGABRT);
	sigpause(0); // Just in case....
    }
}

void Task::exit_all(int code)
{
    exit_code=code;
    exiting=1;
    kill(0, SIGQUIT);
}

static LibMutex* malloc_lock;

static void locker()
{
    malloc_lock->lock();
}

static void unlocker()
{
    malloc_lock->unlock();
}

void Task::initialize(char* pn)
{
    malloc_lock=new LibMutex;
    MemoryManager::set_locker(locker, unlocker);
    progname=strdup(pn);
    tick=CLK_TCK;
    pagesize=getpagesize();
    pid=getpid();
    devzero_fd=open("/dev/zero", O_RDWR);
    if(devzero_fd == -1){
	perror("open");
	exit(-1);
    }
    if(!single_threaded.is_set()){
	sched_lock=0;
#if 0
	// Set up the signal stack so that we will be able to 
	// Catch the SEGV's that need to grow the stacks...
	int stacklen=DEFAULT_SIGNAL_STACK_LENGTH;
	caddr_t stackbot=(caddr_t)mmap(0, stacklen, PROT_READ|PROT_WRITE,
				       MAP_SHARED, devzero_fd, 0);
	if((int)stackbot == -1){
	    perror("mmap");
	    exit(-1);
	}
	stack_t ss;
	ss.ss_sp=stackbot;
	ss.ss_sp+=stacklen-1;
	ss.ss_size=stacklen;
	ss.ss_flags=0;
	if(sigaltstack(&ss, NULL) == -1){
	    perror("sigstack");
	    exit(-1);
	}
#endif

	// Make a Task* for the main thread...
	current_task=new MainTask;
	current_task->activated=1;
	current_task->priv=new TaskPrivate;
	tasks[ntasks++]=current_task;

	// Setup the seg fault handler...
	// For SIGQUIT
	// halt all threads
	// signal(SIGINT, (SIG_PF)handle_halt_signals);
	struct sigaction action;
	action.sa_flags=SA_STACK;
	sigemptyset(&action.sa_mask);

	action.sa_handler=(SIG_PF)handle_halt_signals;
	if(sigaction(SIGQUIT, &action, NULL) == -1){
	    perror("sigaction");
	    exit(-1);
	}

	// For SIGILL, SIGABRT, SIGBUS, SIGSEGV, 
	// prompt the user for a core dump...
	action.sa_handler=(SIG_PF)handle_abort_signals;
	if(sigaction(SIGILL, &action, NULL) == -1){
	    perror("sigaction");
	    exit(-1);
	}
	if(sigaction(SIGABRT, &action, NULL) == -1){
	    perror("sigaction");
	    exit(-1);
	}
	if(sigaction(SIGBUS, &action, NULL) == -1){
	    perror("sigaction");
	    exit(-1);
	}
	if(sigaction(SIGSEGV, &action, NULL) == -1){
	    perror("sigaction");
	    exit(-1);
	}
    }
}

int Task::nprocessors()
{
    static int nproc=-1;
    if(nproc==-1){
	if(!single_threaded.is_set()){
	    if(nproc_arg.is_set()){
		nproc=nproc_arg.value();
	    } else {
		nproc = 1;
	    }
	} else {
	    nproc=1;
	}
    }
    return nproc;
}

void Task::main_exit()
{
    // Remove ourselves from the ready and run the next task
    if(!current_task){
	cerr << "Task::main_exit() called while current_task not set!\n";
	exit(-1);
    }
    lock_scheduler();
    if(readyq.nitems == 0){
	exit(0);
	unlock_scheduler();
	return;
    }
    if(setjmp(current_task->priv->context) == 0){
	current_task=readyq.pop_first();
	longjmp(current_task->priv->context, 1);
    }
    cerr << "We should never get here...\n";
    exit(-1);
}

void Task::yield()
{
    if(!current_task){
	cerr << "Task::yield() called while current_task not set!\n";
	exit(-1);
    }
    if(sched_lock){
	cerr << "yield giving up because scheduler is locked\n";
	return; // Scheduler is locked - return, since this is only a hint anyway
    }
    lock_scheduler();
    if(readyq.nitems == 0){
	cerr << "yield giving up because readyq is empty\n";
	unlock_scheduler();
	return;
    }
    if(setjmp(current_task->priv->context) == 0){
	readyq.append(current_task);
	current_task=readyq.pop_first();
	longjmp(current_task->priv->context, 1);
    }
    unlock_scheduler();
}

int Task::wait_for_task(Task* task)
{
    NOT_FINISHED("wait_for_task");
    return task->priv->retval;
}

//
// Semaphore implementation
//
struct Semaphore_private {
    int count;
    char lock;
};

Semaphore::Semaphore(int count)
{
    if(!single_threaded.is_set()){
	priv=new Semaphore_private;
	priv->count=count;
	spin_unlock(priv->lock);
    } else {
	priv=0;
    }
}

Semaphore::~Semaphore()
{
    if(priv){
	delete priv;
    }
}

void Semaphore::down()
{
    if(!single_threaded.is_set()){
	while(1){
	    spin_lock_yield(priv->lock);
	    if(priv->count>0){
		priv->count--;
		spin_unlock(priv->lock);
		return;
	    } else {
		spin_unlock(priv->lock);
		Task::yield();
	    }
	}
    }
}

void Semaphore::up()
{
    if(!single_threaded.is_set()){
	spin_lock_yield(priv->lock);
	priv->count++;
	spin_unlock(priv->lock);
    }
}

//
// Mutex implementation
//
struct Mutex_private {
    char lock;
};

Mutex::Mutex()
{
    if(!single_threaded.is_set()){
	priv=new Mutex_private;
	spin_unlock(priv->lock);
    } else {
	priv=0;
    }
}

Mutex::~Mutex()
{
    if(priv){
	delete priv;
    }
}

void Mutex::lock()
{
    if(!single_threaded.is_set()){
	spin_lock_yield(priv->lock);
    }
}

void Mutex::unlock()
{
    if(!single_threaded.is_set()){
	spin_unlock(priv->lock);
    }
}


//
// Library Mutex implementation
//

LibMutex::LibMutex()
{
}

LibMutex::~LibMutex()
{
}

void LibMutex::lock()
{
    lock_scheduler();
    skip_csw++;
    unlock_scheduler();
}

void LibMutex::unlock()
{
    lock_scheduler();
    skip_csw++;
    unlock_scheduler();
}


//
// Condition variable implementation
//
struct ConditionVariable_private {
    int nwaiters;
    Mutex mutex;
    Semaphore semaphore;
    ConditionVariable_private();
};

ConditionVariable_private::ConditionVariable_private()
: nwaiters(0), semaphore(0)
{
}


ConditionVariable::ConditionVariable()
{
    if(!single_threaded.is_set()){
	priv=new ConditionVariable_private;
    } else {
	priv=0;
    }
}

ConditionVariable::~ConditionVariable()
{
    if(priv){
	delete priv;
    }
}

void ConditionVariable::wait(Mutex& mutex)
{
    if(!single_threaded.is_set()){
	priv->mutex.lock();
	priv->nwaiters++;
	priv->mutex.unlock();
	mutex.unlock();
	// Block until woken up by signal or broadcast
	priv->semaphore.down();
	mutex.lock();
    }
}

void ConditionVariable::cond_signal()
{
    if(!single_threaded.is_set()){
	priv->mutex.lock();
	if(priv->nwaiters > 0){
	    priv->nwaiters--;
	    priv->semaphore.up();
	}
	priv->mutex.unlock();
    }
}

void ConditionVariable::broadcast()
{
    if(!single_threaded.is_set()){
	priv->mutex.lock();
	while(priv->nwaiters > 0){
	    priv->nwaiters--;
	    priv->semaphore.up();
	}
	priv->mutex.unlock();
    }
}

void Task::sleep(const TaskTime& time)
{
    NOT_FINISHED("Task::sleep");
}

TaskInfo* Task::get_taskinfo()
{
    lock_scheduler();
    TaskInfo* ti=new TaskInfo(ntasks);
    for(int i=0;i<ntasks;i++){
	ti->tinfo[i].name=tasks[i]->name;
	ti->tinfo[i].stacksize=tasks[i]->priv->stacklen-pagesize;
	ti->tinfo[i].stackused=tasks[i]->priv->stacklen-tasks[i]->priv->redlen;
	ti->tinfo[i].pid=pid;
	ti->tinfo[i].taskid=tasks[i];
    }
    unlock_scheduler();
    return ti;
}

void Task::coredump(Task* task)
{
    kill(pid, SIGABRT);
}

void Task::debug(Task* task)
{
    char buf[1000];
    char* dbx=getenv("SCI_DEBUGGER");
    if(!dbx)
	dbx="xterm -e %s %d &";
    sprintf(buf, dbx, progname, pid);
    if(system(buf) == -1)
	perror("system");
}

MainTask::MainTask()
: Task("main()", 1)
{
}

MainTask::~MainTask()
{
}

int MainTask::body(int)
{
    cerr << "Error - main's body called!\n";
    exit(-1);
}

static void fd_copy(int n, fd_set* to, fd_set* from)
{
    bcopy(from, to, sizeof(fd_set));
}

// Interface to select...
int Task::mtselect(int nfds, fd_set* readfds, fd_set* writefds,
		   fd_set* exceptfds, struct timeval* timeout)
{
    if(readyq.nitems == 0 ||
       (timeout && timeout->tv_sec == 0 && timeout->tv_usec == 0)){
	int val=select(nfds, readfds, writefds, exceptfds, timeout);
	return val;
    } else {
	struct timeval to;
	struct timeval done;
	struct timezone tz;
	if(gettimeofday(&done, &tz) != 0){
	    perror("gettimeofday");
	    exit(-1);
	}
	done.tv_sec+=timeout?timeout->tv_sec:0;
	done.tv_usec+=timeout?timeout->tv_usec:0;
	if(done.tv_usec > 1000000){
	    done.tv_usec-=1000000;
	    done.tv_sec++;
	}
	while(1){
	    fd_set rfds;
	    fd_set* prfds=0;
	    if(readfds){
		fd_copy(nfds, &rfds, readfds);
		prfds=&rfds;
	    }
	    fd_set wfds;
	    fd_set* pwfds=0;
	    if(writefds){
		fd_copy(nfds, &wfds, writefds);
		pwfds=&wfds;
	    }
	    fd_set efds;
	    fd_set* pefds=0;
	    if(exceptfds){
		fd_copy(nfds, &efds, exceptfds);
		pefds=&efds;
	    }
	    to.tv_sec=0;
	    to.tv_usec=0;
	    int val=select(nfds, prfds, pwfds, pefds, &to);
	    if(val != 0){
		if(readfds)
		    fd_copy(nfds, readfds, &rfds);
		if(writefds)
		    fd_copy(nfds, writefds, &wfds);
		if(exceptfds)
		    fd_copy(nfds, exceptfds, &efds);
		return val;
	    }
	    if(timeout){
		struct timeval now;
		if(gettimeofday(&now, &tz) != 0){
		    perror("gettimeofday");
		    exit(-1);
		}
		if(now.tv_sec > done.tv_sec ||
		   (now.tv_sec == done.tv_sec && now.tv_usec >= done.tv_usec)){
		    if(readfds)
			FD_ZERO(readfds);
		    if(writefds)
			FD_ZERO(writefds);
		    if(exceptfds)
			FD_ZERO(exceptfds);
		    return 0;
		}
	    }
	    // See if the timeout is up...
	    Task::yield();
	}
    }
}
