
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

#include <Multitask/Task.h>
#include <Multitask/ITC.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <stdlib.h>
#include <ulocks.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/signal.h>
#include <sys/types.h>
#include <sys/sysmp.h>
#include <sys/mman.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <ucontext.h>
#include <sys/time.h>

#define DEFAULT_STACK_LENGTH 64*1024
#define INITIAL_STACK_LENGTH 16*1024
#define DEFAULT_SIGNAL_STACK_LENGTH 16*1024

#define NOT_FINISHED(what) cerr << what << ": Not Finished " << __FILE__ << " (line " << __LINE__ << ") " << endl

extern "C" int Task_try_lock(unsigned long*);

static int aborting=0;
static int exit_code;
static int exiting=0;
static long tick;
static int pagesize;
static int devzero_fd;
static char* progname;

struct TaskPrivate {
    int tid;
    usema_t* running;
    int retval;
    caddr_t sp;
    caddr_t stackbot;
    size_t stacklen;
    size_t redlen;
    void (*handle_alrm)(void*);
    void* alrm_data;
};

struct TaskArgs {
    int arg;
    Task* t;
};

class MainTask : public Task {
public:
    MainTask();
    virtual ~MainTask();
    int body(int);
};


#define MAXTASKS 1000
static int ntasks=0;
static Task* tasks[MAXTASKS];

// Global locks...
static usema_t* main_sema;
static usema_t* sched_lock;
static int nsched;

// For locks...
int arena_created=0;
usptr_t* arena;

// For taskid and thread local implementation...
struct TaskLocalMemory {
    Task* current_task;
};
TaskLocalMemory* task_local;

static void make_arena()
{
    if(!arena_created){
	usconfig(CONF_ARENATYPE, US_SHAREDONLY);
	usconfig(CONF_INITSIZE, 16*1024*1024);
	usconfig(CONF_INITUSERS, (unsigned int)800);
	//char* lockfile=tempnam(NULL, "sci");
	//arena=usinit(lockfile);
	//free(lockfile);
	arena=usinit("/dev/zero");
	if(!arena){
	    perror("usinit");
	    Task::exit_all(-1);
	}
	arena_created=1;
    }
}

static void makestack(TaskPrivate* priv)
{
    priv->stacklen=DEFAULT_STACK_LENGTH+pagesize;
    priv->stackbot=(caddr_t)mmap(0, priv->stacklen, PROT_READ|PROT_WRITE,
				 MAP_SHARED, devzero_fd, 0);
    if((long)priv->stackbot == -1){
	perror("mmap");
	Task::exit_all(-1);
    }
    // Now unmap the bottom part of it...
    priv->redlen=DEFAULT_STACK_LENGTH-INITIAL_STACK_LENGTH;
    if(getenv("SCI_CATCH_SIGNALS")){
	priv->redlen=DEFAULT_STACK_LENGTH-INITIAL_STACK_LENGTH;
    } else {
	priv->redlen=pagesize;
    }
    priv->sp=(char*)priv->stackbot+priv->stacklen-1;
    if(mprotect(priv->stackbot, priv->redlen, PROT_NONE) == -1){
	perror("mprotect");
	Task::exit_all(-1);
    }
}

void* runbody(void* vargs)
{
    TaskArgs* args=(TaskArgs*)vargs;
    int arg=args->arg;
    Task* t=args->t;
    delete args;

    t->startup(arg);
    return 0;
}

int Task::startup(int task_arg)
{
    task_local->current_task=this;
    int retval=body(task_arg);
    Task::taskexit(this, retval);
    return 0; // Never reached.
}

// We are done..
void Task::taskexit(Task* xit, int retval)
{
    xit->priv->retval=retval;
    usvsema(xit->priv->running);

    // See if everyone is done..
    if(uspsema(sched_lock) == -1){
	perror("uspsema");
	Task::exit_all(-1);
    }
    if(--nsched == 0){
	if(usvsema(main_sema) == -1){
	    perror("usvsema");
	    Task::exit_all(-1);
	}
    }
    if(usvsema(sched_lock) == -1){
	perror("usvsema");
	Task::exit_all(-1);
    }
    _exit(0);
} 

// Fork a thread
void Task::activate(int task_arg)
{
    if(activated){
	cerr << "Error: task is being activated twice!" << endl;
	Task::exit_all(-1);
    }
    activated=1;
    priv=scinew TaskPrivate;
    TaskArgs* args=scinew TaskArgs;
    args->arg=task_arg;
    args->t=this;
    priv->running=usnewsema(arena, 0);
    if(!priv->running){
	perror("usnewsema");
	Task::exit_all(-1);
    }
    makestack(priv);

    if(uspsema(sched_lock) == -1){
	perror("uspsema");
	Task::exit_all(-1);
    }
    nsched++;
    priv->tid=sprocsp((void (*)(void*, size_t))runbody,
		      /*PR_SADDR|PR_SDIR|PR_SUMASK|PR_SULIMIT|PR_SID,*/
		      PR_SALL,
		      (void*)args, priv->sp, priv->stacklen);
    if(priv->tid==	-1){
	perror("sprocsp");
	Task::exit_all(-1);
    }
    tasks[ntasks++]=this;
    if(usvsema(sched_lock) == -1){
	perror("usvsema");
	Task::exit_all(-1);
    }
}

Task* Task::self()
{
    return task_local->current_task;
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
    case SIGEMT:
	sprintf(buf, "SIGEMT (Emulation Trap)");
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
    case SIGSYS:
	sprintf(buf, "SIGSYS (bad argument to system call)");
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
    case SIGURG:
	sprintf(buf, "SIGURG (urgent condition on IO channel)");
	break;
    case SIGIO:
	sprintf(buf, "SIGIO (IO possible)");
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
    case SIGXCPU:
	sprintf(buf, "SIGXCPU (CPU time limit exceeded)");
	break;
    case SIGXFSZ:
	sprintf(buf, "SIGXFSZ (Filesize limit exceeded)");
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
    char* tname=self?self->get_name():"main";
    if(exiting && sig==SIGQUIT){
	fprintf(stderr, "Thread \"%s\"(pid %d) exiting...\n", tname, getpid());
	exit(exit_code);
    }
	
    // Kill all of the threads...
#if defined(_LONGLONG)
    caddr_t addr=(caddr_t)context->sc_badvaddr;
#else
    caddr_t addr=(caddr_t)context->sc_badvaddr.lo32;
#endif
    char* signam=signal_name(sig, code, addr);
    fprintf(stderr, "Thread \"%s\"(pid %d) caught signal %s.  Going down...\n", tname, getpid(), signam);
    Task::exit_all(-1);
}

static void handle_abort_signals(int sig, int code, sigcontext_t* context)
{
    if(aborting)
	exit(0);
    Task* self=Task::self();
    char* tname=self?self->get_name():"main";
#if defined(_LONGLONG)
    caddr_t addr=(caddr_t)context->sc_badvaddr;
#else
    caddr_t addr=(caddr_t)context->sc_badvaddr.lo32;
#endif
    char* signam=signal_name(sig, code, addr);

    // See if it is a segv on the stack - if so, grow it...
    if(sig==SIGSEGV && code==EACCES
       && addr >= self->priv->stackbot
       && addr < self->priv->stackbot+self->priv->stacklen){
	self->priv->redlen -= pagesize;
	if(self->priv->redlen <= 0){
	    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) ran off end of stack! \n",
		    7,7,7,tname, getpid(), signam);
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
    } else if(buf[0] == 'd' || buf[0] == 'D') {
	// Start the debugger...
	Task::debug(Task::self());
	while(1){
	    sigpause(0);
	}
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

void Task::initialize(char* pn)
{
    progname=strdup(pn);
    tick=CLK_TCK;
    pagesize=getpagesize();
    devzero_fd=open("/dev/zero", O_RDWR);
    if(devzero_fd == -1){
	perror("open");
	exit(-1);
    }
    make_arena();
    sched_lock=usnewsema(arena, 1);
    if(!sched_lock){
	perror("usnewsema");
	exit(-1);
    }
    main_sema=usnewsema(arena, 0);
    if(!main_sema){
	perror("main_sema");
	exit(-1);
    }
    
    // Set up the task local memory...
#if 1
    task_local=(TaskLocalMemory*)mmap(0, sizeof(TaskLocalMemory),
				      PROT_READ|PROT_WRITE,
				      MAP_PRIVATE|MAP_LOCAL,
				      devzero_fd, 0);
#endif
#if 0
    task_local=(TaskLocalMemory*)mmap((void*)0x50000000, sizeof(TaskLocalMemory),
				      PROT_READ|PROT_WRITE,
				      MAP_PRIVATE|MAP_LOCAL|MAP_FIXED,
				      devzero_fd, 0);
#endif
    if(!task_local){
	cerr << "Error mapping /dev/zero!\n";
	exit(-1);
    }

    task_local->current_task=scinew MainTask;
    task_local->current_task->activated=1;
    task_local->current_task->priv=scinew TaskPrivate;

    // Set up the signal stack so that we will be able to 
    // Catch the SEGV's that need to grow the stacks...
    int stacklen=DEFAULT_SIGNAL_STACK_LENGTH;
    caddr_t stackbot=(caddr_t)mmap(0, stacklen, PROT_READ|PROT_WRITE,
				   MAP_SHARED, devzero_fd, 0);
    if((long)stackbot == -1){
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
    
    // Setup the seg fault handler...
    // For SIGQUIT
    // halt all threads
    // signal(SIGINT, (SIG_PF)handle_halt_signals);
    struct sigaction action;
    action.sa_flags=SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    
    action.sa_handler=(SIG_PF)handle_halt_signals;
    if(sigaction(SIGQUIT, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }

    // For SIGILL, SIGABRT, SIGBUS, SIGSEGV, 
    // prompt the user for a core dump...
    if(getenv("SCI_CATCH_SIGNALS")){
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
	nproc = sysmp(MP_NAPROCS);
    }
    return nproc;
}

void Task::main_exit()
{
    uspsema(main_sema);
    exit(0);
}

void Task::yield()
{
    sginap(0);
}


int Task::wait_for_task(Task* task)
{
    if(uspsema(task->priv->running) == -1){
	perror("usvsema");
	Task::exit_all(-1);
    }
    if(usvsema(task->priv->running) == -1){
	perror("usvsema");
	Task::exit_all(-1);
    }
    return task->priv->retval;
}

//
// Semaphore implementation
//
struct Semaphore_private {
    usema_t* semaphore;
};

Semaphore::Semaphore(int count)
{
    priv=scinew Semaphore_private;
    make_arena();
    priv->semaphore=usnewsema(arena, count);
    if(!priv->semaphore){
	perror("usnewsema");
	Task::exit_all(-1);
    }
}

Semaphore::~Semaphore()
{
    if(priv){
	usfreesema(priv->semaphore, arena);
	delete priv;
    }
}

void Semaphore::down()
{
    if(uspsema(priv->semaphore) == -1){
	perror("upsema");
	Task::exit_all(-1);
    }
}

int Semaphore::try_down()
{
    int stry=uscpsema(priv->semaphore);
    if(stry == -1){
	perror("upsema");
	Task::exit_all(-1);
    }
    return stry;
}

void Semaphore::up()
{
    if(usvsema(priv->semaphore) == -1){
	perror("usvsema");
	Task::exit_all(-1);
    }
}

Mutex::Mutex()
{
    priv=0;
}

Mutex::~Mutex()
{
}

void Mutex::lock()
{
    // Simple try first...
    if(Task_try_lock((unsigned long*)&priv) == 0)
	return;
    int rt=1000000;
    int cnt=0;
    while(rt){
	// Try the lock...
	for(int i=0;i<10;i++){
	    if(Task_try_lock((unsigned long*)&priv) == 0)
		return;
	}
	// Sleep...
	if(cnt++ < 1000)
	    sginap(0);
	else
	    sginap(1);

	rt--;
    }
    fprintf(stderr, "Possible deadlock after 100000 retries\n");
    Task::exit_all(-1);
}

void Mutex::unlock()
{
    priv=0;
}

int Mutex::try_lock()
{
    if(Task_try_lock((unsigned long*)&priv) == 0)
	return 1;
    return 0;
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
: semaphore(0), nwaiters(0)
{
}


ConditionVariable::ConditionVariable()
{
    priv=scinew ConditionVariable_private;
}

ConditionVariable::~ConditionVariable()
{
    if(priv){
	delete priv;
    }
}

void ConditionVariable::wait(Mutex& mutex)
{
    priv->mutex.lock();
    priv->nwaiters++;
    priv->mutex.unlock();
    mutex.unlock();
    // Block until woken up by signal or broadcast
    priv->semaphore.down();
    mutex.lock();
}

void ConditionVariable::cond_signal()
{
    priv->mutex.lock();
    if(priv->nwaiters > 0){
	priv->nwaiters--;
	priv->semaphore.up();
    }
    priv->mutex.unlock();
}

void ConditionVariable::broadcast()
{
    priv->mutex.lock();
    while(priv->nwaiters > 0){
	priv->nwaiters--;
	priv->semaphore.up();
    }
    priv->mutex.unlock();
}

void Task::sleep(const TaskTime& time)
{
    sginap((time.secs*1000+time.usecs)/tick);
}

TaskInfo* Task::get_taskinfo()
{
    if(uspsema(sched_lock) == -1){
	perror("uspsema");
	Task::exit_all(-1);
    }
    TaskInfo* ti=scinew TaskInfo(ntasks);
    for(int i=0;i<ntasks;i++){
	ti->tinfo[i].name=tasks[i]->name;
	ti->tinfo[i].stacksize=tasks[i]->priv->stacklen-pagesize;
	ti->tinfo[i].stackused=tasks[i]->priv->stacklen-tasks[i]->priv->redlen;
	ti->tinfo[i].pid=tasks[i]->priv->tid;
	ti->tinfo[i].taskid=tasks[i];
    }
    if(usvsema(sched_lock) == -1){
	perror("usvsema");
	Task::exit_all(-1);
    }
    return ti;
}

void Task::coredump(Task* task)
{
    kill(task->priv->tid, SIGABRT);
}

void Task::debug(Task* task)
{
    char buf[1000];
    char* dbx=getenv("SCI_DEBUGGER");
    if(!dbx)
	dbx="winterm -c dbx -p %d &";
    sprintf(buf, dbx, task->priv->tid, progname);
    if(system(buf) == -1)
	perror("system");
}

static void handle_alrm(int, int, sigcontext_t*)
{
    Task* task=Task::self();
    if(task->priv->handle_alrm)
       (*task->priv->handle_alrm)(task->priv->alrm_data);
}

int Task::start_itimer(const TaskTime& start, const TaskTime& interval,
		       void (*handler)(void*), void* cbdata)
{
    if(ntimers > 0){
	NOT_FINISHED("Multiple timers in a single thread");
	return 0;
    }
    ITimer** new_timers=scinew ITimer*[ntimers+1];
    if(timers){
	for(int i=0;i<ntimers;i++){
	    new_timers[i]=timers[i];
	}
	delete[] timers;
    }
    timers=new_timers;
    ITimer* t=scinew ITimer;
    timers[ntimers]=t;
    ntimers++;
    t->start=start;
    t->interval=interval;
    t->handler=handler;
    t->cbdata=cbdata;
    t->id=timer_id++;
    struct itimerval it;
    it.it_interval.tv_sec=interval.secs;
    it.it_interval.tv_usec=interval.usecs;
    it.it_value.tv_sec=start.secs;
    it.it_value.tv_usec=start.usecs;
    struct sigaction action;
    action.sa_flags=SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    action.sa_handler=(SIG_PF)handle_alrm;
    if(sigaction(SIGALRM, &action, NULL)== -1){
	perror("sigaction");
	return 0;
    }
    if(setitimer(ITIMER_REAL, &it, 0) == -1){
	perror("setitimer");
	return 0;
    }
    priv->handle_alrm=handler;
    priv->alrm_data=cbdata;
    return t->id;
}

void Task::cancel_itimer(int which_timer)
{
    priv->handle_alrm=0;
    for(int idx=0;idx<ntimers;idx++){
	if(timers[idx]->id == which_timer)
	    break;
    }
    if(idx==ntimers){
	cerr << "Cancelling bad timer!\n" << endl;
	return;
    }
    struct itimerval it;
    timerclear(&it.it_interval);
    timerclear(&it.it_value);
    if(setitimer(ITIMER_REAL, &it, 0) == -1){
	perror("setitimer");
	return;
    }
    for(int i=idx;i<ntimers-1;i++)
	timers[i]=timers[i+1];
    ntimers--;
}

struct Barrier_private {
    barrier_t* barrier;
};

Barrier::Barrier()
{
    priv=new Barrier_private;
    make_arena();
    priv->barrier=new_barrier(arena);
    if(!priv->barrier){
      perror("new_barrier");
      Task::exit_all(-1);
    }
}

Barrier::~Barrier()
{
    free_barrier(priv->barrier);
}

void Barrier::wait(int n)
{
    barrier(priv->barrier, n);
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
    return 0;
}

