
/*
 * Implementation of threads for IRIX, using sproc.
 * Irix will also run the Thread_posix threads implementation.
 */

/*
 * User signals:
 *    SIGQUIT: Tells all of the processes to quit
 *    SIGUSR1: Tells another thread to start profiling.
 *    SIGUSR2: Tells the main thread when another thread exits.
 *             NOTE - this might not be one of our threads!
 */

#include <sys/types.h>
#include <sys/prctl.h>
#include <ulocks.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <sys/signal.h>
#include <sys/sysmp.h>
#include <sys/errno.h>
#include <sys/syssgi.h>
#include <sys/time.h>
#include <assert.h>
extern "C" {
#include <sys/pmo.h>
#include <fetchop.h>
}
#include "Thread.h"
#include "ThreadGroup.h"
#include "Mutex.h"
#include "Semaphore.h"
#include "Barrier.h"
#include "ConditionVariable.h"
#include "PoolMutex.h"

#define TOPBIT ((unsigned int)0x80000000)

extern "C" int __ateachexit(void(*)());
#define THREAD_STACKSIZE 256*1024

#define MAXBSTACK 10
#define MAXTHREADS 4000

#define N_POOLMUTEX 301

struct Thread_private {
    Thread* thread;
    int pid;
    caddr_t sp;
    caddr_t stackbot;
    size_t stacklen;
    usema_t* startup;
    usema_t* done;
    usema_t* delete_ready;
    int state;
    int bstacksize;
    const char* blockstack[MAXBSTACK];
};

static Thread_private* idle_main;
static Thread_private* idle[MAXTHREADS];
static int nidle;
static Thread_private* active[MAXTHREADS];
static int nactive;
static bool initialized;
static usema_t* schedlock;
static usptr_t* arena;
static fetchop_reservoir_t reservoir;
static usptr_t* poolmutex_arena;
static ulock_t poolmutex_lock;
static ulock_t poolmutex[N_POOLMUTEX];
static char* poolmutex_names[N_POOLMUTEX];
static char poolmutex_namearray[N_POOLMUTEX*20]; // Enough to hold "PoolMutex #%d"
static int poolmutex_count;
static int poolmutex_average;
static int poolmutex_idx;
static int poolmutex_users[N_POOLMUTEX];
static usptr_t* main_sema;
static int devzero_fd;
static int main_pid;
static bool exiting;
static int exit_code;
static bool aborting;
static int nprocessors;
static int last_processor;
static int timer_32bit;
unsigned int iotimer_high;
volatile unsigned int* iotimer_addr32;
#if _MIPS_ISA == _MIPS_ISA_MIPS1 || _MIPS_ISA ==  _MIPS_ISA_MIPS2
volatile unsigned int* iotimer_addr;
#define TIMERTYPE unsigned int
#else
volatile SysClock *iotimer_addr;
#define TIMERTYPE SysClock
#endif
static SysClock orig_timer;
static double ticks_to_seconds;
static double seconds_to_ticks;
static int hittimer;
static usema_t* control_c_sema;
static void handle_profile(int, int, sigcontext_t*);
static pmo_handle_t *mlds=0;
static pmo_handle_t mldset=0;
static pmo_handle_t rr_policy=0;


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
#define STATE_BLOCK_BARRIER 12
#define STATE_BLOCK_FETCHOP 13

struct ThreadLocalMemory {
    Thread* current_thread;
};
ThreadLocalMemory* thread_local;

static void lock_scheduler() {
    if(uspsema(schedlock) == -1){
	perror("uspsema");
	Thread::niceAbort();
    }
}

static void unlock_scheduler() {
    if(usvsema(schedlock) == -1){
	perror("usvsema");
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

int Thread::couldBlock(const char* why) {
    Thread_private* p=Thread::currentThread()->priv;
    return push_bstack(p, STATE_BLOCK_ANY, why);
}

void Thread::couldBlock(int restore) {
    Thread_private* p=Thread::currentThread()->priv;
    pop_bstack(p, restore);
}

/*
 * Return the statename for p
 */
static const char* getstate(Thread_private* p){
    switch(p->state) {
    case STATE_STARTUP:
	return "startup";
    case STATE_RUNNING:
	return "running";
    case STATE_IDLE:
	return "idle";
    case STATE_SHUTDOWN:
	return "shutting down";
    case STATE_BLOCK_SEMAPHORE:
	return "blocking on semaphore";
    case STATE_PROGRAM_EXIT:
	return "waiting for program exit";
    case STATE_JOINING:
	return "joining with thread";
    case STATE_BLOCK_MUTEX:
	return "blocking on mutex";
    case STATE_BLOCK_ANY:
	return "blocking";
    case STATE_DIED:
	return "died";
    case STATE_BLOCK_POOLMUTEX:
	return "blocking on pool mutex";
    case STATE_BLOCK_BARRIER:
	return "spinning in barrier";
    case STATE_BLOCK_FETCHOP:
	return "performing fetch&op";
    default:
	return "UNKNOWN";
    }
}

static void print_threads(FILE* fp, int print_idle) {
    for(int i=0;i<nactive;i++){
	Thread_private* p=active[i];
	const char* tname=p->thread?p->thread->threadName():"???";
	fprintf(fp, "%d: %s (", p->pid, tname);
	if(p->thread){
	    if(p->thread->isDaemon())
		fprintf(fp, "daemon, ");
	    if(p->thread->isDetached())
		fprintf(fp, "detached, ");
	}
	fprintf(fp, "state=%s", getstate(p));
	for(int i=0;i<p->bstacksize;i++){
	    fprintf(fp, ", %s", p->blockstack[i]);
	}
	fprintf(fp, ")\n");
    }
    if(print_idle){
	for(int i=0;i<nidle;i++){
	    Thread_private* p=idle[i];
	    fprintf(fp, "%d: Idle worker\n", p->pid);
	}
	if(idle_main){
	    fprintf(fp, "%d: Completed main thread\n", idle_main->pid);
	}
    }
}

static Thread_private* find_thread_from_tid(int tid){
    for(int i=0;i<nactive;i++){
	Thread_private* p=active[i];
	if(p->pid == tid)
	    return p;
    }
    return 0;
}

/*
 * Shutdown the threads...
 */
void Thread_exit() {
    if(exiting)
	return;
    Thread* self=Thread::currentThread();

    Thread_shutdown(self);
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

static void wait_shutdown() {
    long n;
    if((n=prctl(PR_GETNSHARE)) > 1){
	fprintf(stderr, "Waiting for %d threads to shut down: ", (int)(n-1));
	sginap(10);
	int delay=10;
	while((n=prctl(PR_GETNSHARE)) > 1){
	    fprintf(stderr, "%d...", (int)(n-1));
	    sginap(delay);
	    delay+=10;
	}
	fprintf(stderr, "done\n");
    }
}

/*
 * Handle sigquit - usually sent by control-C
 */
static void handle_quit(int sig, int code, sigcontext_t*)
{
    if(exiting){
	if(getpid() == main_pid){
	    wait_shutdown();
	}
	exit(exit_code);
    }
    // Try to acquire a lock.  If we can't, then assume that somebody
    // else already caught the signal...
    Thread* self=Thread::currentThread();
    if(self==0)
	return; // This is an idle thread...
    if(sig == SIGINT){
	int st=uscpsema(control_c_sema);
	if(st==-1){
	    perror("uscsetlock");
	    Thread::niceAbort();
	}
    
	if(st == 0){
	    // This will wait until the other thread is done
	    // handling the interrupt
	    uspsema(control_c_sema);
	    usvsema(control_c_sema);
	    return;
	}
	// Otherwise, we handle the interrupt
    }

    const char* tname=self?self->threadName():"main?";

    // Kill all of the threads...
    char* signam=signal_name(sig, code, 0);
    int pid=getpid();
    fprintf(stderr, "Thread \"%s\"(pid %d) caught signal %s\n", tname, pid, signam);
    if(sig==SIGINT){
	// Print out the thread states...
	fprintf(stderr, "\n\nActive threads:\n");
	print_threads(stderr, 1);
	Thread::niceAbort();
	usvsema(control_c_sema);
    } else {
	exiting=true;
	exit_code=1;
	exit(1);
    }
}


/*
 * Handle an abort signal - like segv, bus error, etc.
 */
static void handle_abort_signals(int sig, int code, sigcontext_t* context)
{
    if(aborting)
	exit(0);
    struct sigaction action;
    sigemptyset(&action.sa_mask);
    action.sa_handler=SIG_DFL;
    action.sa_flags=0;
    if(sigaction(sig, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }    

    Thread* self=Thread::currentThread();
    const char* tname=self?self->threadName():"idle or main";
#if defined(_LONGLONG)
    caddr_t addr=(caddr_t)context->sc_badvaddr;
#else
    caddr_t addr=(caddr_t)context->sc_badvaddr.lo32;
#endif
    char* signam=signal_name(sig, code, addr);
    fprintf(stderr, "%c%c%cThread \"%s\"(pid %d) caught signal %s\n", 7,7,7,tname, getpid(), signam);
    Thread::niceAbort();

    action.sa_handler=(SIG_PF)handle_abort_signals;
    action.sa_flags=0;
    if(sigaction(sig, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }
}

/*
 * Handle SIGUSR2 - the signal that gets sent when another thread dies.
 */
static void handle_thread_death(int, int, sigcontext_t*) {
    if(exiting){
	if(getpid() == main_pid)
	    return;
	exit(exit_code);
    }
    Thread* self=Thread::currentThread();
    if(!self)
	return; // This is an idle thread...
    Thread_private* priv=self->getPrivate();
    priv->state=STATE_DIED;
    if(priv->pid != main_pid)
	return;
}

#if 0
void Thread_send_event(Thread* t, Thread::ThreadEvent event) {
    Thread::event(t, event);
}
#endif

/*
 * Setup signals for the current thread
 */
static void install_signal_handlers(){
    struct sigaction action;
    sigemptyset(&action.sa_mask);
    action.sa_flags=0;

    action.sa_handler=(SIG_PF)handle_abort_signals;
    if(sigaction(SIGILL, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }
    if(sigaction(SIGABRT, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }
    if(sigaction(SIGTRAP, &action, NULL) == -1){
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

    action.sa_handler=(SIG_PF)handle_thread_death;
#if 0
    if(sigaction(SIGUSR2, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }

    action.sa_handler=(SIG_PF)handle_profile;
    if(sigaction(SIGUSR1, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }
#endif
    
    action.sa_handler=(SIG_PF)handle_quit;
    if(sigaction(SIGQUIT, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }    
    if(sigaction(SIGINT, &action, NULL) == -1){
	perror("sigaction");
	exit(-1);
    }    
}

static void handle_alrm(int, int, sigcontext_t*)
{
    unsigned int t=*iotimer_addr32;
    unsigned int h=iotimer_high;
    if((t&TOPBIT) != (h&TOPBIT)){
	if((t&TOPBIT) == 0){
	    iotimer_high=(h&(~TOPBIT))+1;
	} else {
	    iotimer_high=h|TOPBIT;
	}
    }
    if(!hittimer)
	hittimer=1;
}

static void mld_alloc(int size, int nmld, 
		      pmo_handle_t *(&mlds), pmo_handle_t &mldset)
{
  int i;

  mlds = new pmo_handle_t[nmld];
  assert(mlds);

  for(i=0; i<nmld; i++) 
    {
      mlds[i] = mld_create( 0, size );
      if ((long)mlds[i] < 0) 
	perror("mld_create()");
      
    }
  mldset = mldset_create( mlds, nmld );
  if ((long) mldset < 0) 
    perror("mldset_create");

  if ( mldset_place( mldset, TOPOLOGY_FREE, 0, 0, RQMODE_ADVISORY ) < 0)
    {
      perror("mldset_place");
      fprintf( stderr, "set: %p nmld: %d ( ", (void *)mldset, nmld );
      for(i=0; i<nmld; i++)
	fprintf( stderr, "%d ", mlds[i] );
      fprintf( stderr, ")\n" );
    }
}

/*
 * Intialize threads for irix
 */
void Thread::initialize() {

    int poffmask = getpagesize() - 1;
    unsigned int cycleval;
    __psunsigned_t phys_addr = syssgi(SGI_QUERY_CYCLECNTR, &cycleval);
    __psunsigned_t raddr = phys_addr & ~poffmask;
    int fd = open("/dev/mmem", O_RDONLY);

    iotimer_addr = (volatile TIMERTYPE *)mmap(0, poffmask, PROT_READ,
					      MAP_PRIVATE, fd, (off_t)raddr);
    iotimer_addr = (volatile TIMERTYPE *)((__psunsigned_t)iotimer_addr +
					 (phys_addr & poffmask));
    iotimer_addr32 = (volatile unsigned int*)iotimer_addr;
    ticks_to_seconds=(double)cycleval*1.e-12;
    seconds_to_ticks=1./ticks_to_seconds;

    long ccsize=syssgi(SGI_CYCLECNTR_SIZE);
    if(ccsize == 32){
	timer_32bit=true;
    }

    double overflow=(65536.*65536.);
    if(!timer_32bit)
	overflow=overflow*overflow;
    overflow*=ticks_to_seconds;

    orig_timer=0;
    orig_timer=Thread::currentTicks();

    iotimer_high=(*iotimer_addr32)&TOPBIT;

    if(timer_32bit){
	// Set up sigalrm handler...
	struct sigaction action;
	action.sa_flags=0;
	sigemptyset(&action.sa_mask);

	action.sa_handler=(SIG_PF)handle_alrm;
	if(sigaction(SIGALRM, &action, NULL) == -1){
	    perror("sigaction");
	    exit(-1);
	}

	int ticks=overflow/8;
	struct itimerval dt;
	dt.it_interval.tv_sec=ticks;
	dt.it_interval.tv_usec=0;
	dt.it_value.tv_sec=0;
	dt.it_value.tv_usec=1;
	struct itimerval old;
	if(setitimer(ITIMER_REAL, &dt, &old) != 0){
	    perror("setitimer");
	    exit(1);
	}
	while(!hittimer)
	    sigsuspend(0);
    }


    usconfig(CONF_ARENATYPE, US_SHAREDONLY);
    usconfig(CONF_INITSIZE, 3*1024*1024);
    usconfig(CONF_INITUSERS, (unsigned int)140);
    poolmutex_arena=usinit("/dev/zero");
    if(!poolmutex_arena){
	perror("usinit 1");
	exit(1);
    }
    poolmutex_lock=usnewlock(poolmutex_arena);
    if(!poolmutex_lock){
	perror("usnewlock");
	exit(-1);
    }
    int c=0;
    for(int i=0;i<N_POOLMUTEX;i++){
	poolmutex[i]=usnewlock(poolmutex_arena);
	if(!poolmutex[i]){
	    fprintf(stderr, "Pool mutex allocate failed on %d of %d\n", i, N_POOLMUTEX);
	    perror("usnewlock");
	    exit(-1);
	}
	poolmutex_names[i]=&poolmutex_namearray[c];
	sprintf(poolmutex_names[i], "PoolMutex #%d", i);
	c+=strlen(poolmutex_names[i])+1;
	if(c > sizeof(poolmutex_namearray)){
	    fprintf(stderr, "PoolMutex name array overflow!\n");
	    exit(1);
	}
    }

    usconfig(CONF_ARENATYPE, US_SHAREDONLY);
    usconfig(CONF_INITSIZE, 3*1024*1024);
    usconfig(CONF_INITUSERS, (unsigned int)140);
    //char* lockfile=tempnam(NULL, "sci");
    //arena=usinit(lockfile);
    //free(lockfile);
    arena=usinit("/dev/zero");
    if(!arena){
	perror("usinit 2");
	exit(1);
    }
    reservoir=fetchop_init(USE_DEFAULT_PM, 10);
    if(!reservoir){
	perror("fetchop_init");
	exit(1);
    }
    devzero_fd=open("/dev/zero", O_RDWR);
    if(devzero_fd == -1){
	perror("open");
	exit(1);
    }
    schedlock=usnewsema(arena, 1);
    main_sema=usnewsema(arena, 0);
    nprocessors=Thread::numProcessors();

    control_c_sema=usnewsema(poolmutex_arena, 1);
    if(!poolmutex_lock){
	perror("usnewlock");
	exit(-1);
    }
    main_pid=getpid();
    /*
     * The functionality relies on SGI's __ateachexit function.  If
     * that ever goes away, then we will probably have to write our
     * own exit.
     */
    __ateachexit(Thread_exit);

    thread_local=(ThreadLocalMemory*)mmap(0, sizeof(ThreadLocalMemory),
					  PROT_READ|PROT_WRITE,
					  MAP_PRIVATE|MAP_LOCAL,
					  devzero_fd, 0);
    /*
     * We have to say that we are initialized here, because the
     * following calls can create synchronization primitives that
     * will rely only on the above initialization.
     */
    initialized=true;

    ThreadGroup::default_group=new ThreadGroup("default group", 0);
    Thread* mainthread=new Thread(ThreadGroup::default_group, "main");
    mainthread->priv=new Thread_private;
    mainthread->priv->pid=main_pid;
    mainthread->priv->thread=mainthread;
    mainthread->priv->state=STATE_RUNNING;
    mainthread->priv->bstacksize=0;
    mainthread->priv->done=usnewsema(arena, 0);
    mainthread->priv->delete_ready=0;
    lock_scheduler();
    active[nactive]=mainthread->priv;
    nactive++;
    unlock_scheduler();

    thread_local->current_thread=mainthread;
    install_signal_handlers();

#if 0
    if(prctl(PR_SETEXITSIG, SIGUSR2) == -1){
	perror("prctl");
	exit(1);
    }
#endif

    /* Setup memory locality domains and policy models for memory placement 
     * (jamie@acl.lanl.gov 
     */
    mld_alloc( 32*1024*1024       /* memory needed per node */, 
	       (nprocessors+1)/2,  /* number of nodes */ 
	       mlds,
	       mldset
	       );
    policy_set_t ps;
    pm_filldefault(&ps);
    ps.placement_policy_name = "PlacementRoundRobin";
    ps.placement_policy_args = (void *) mldset;
    rr_policy = pm_create ( &ps );
    if (rr_policy == -1)
      perror("pm_create");
}

void Thread::roundRobinPlacement( void *mem, size_t len )
{
  if (mlds != 0)
    {
      int err = pm_attach( rr_policy, mem, len );
      if (err == -1)
	perror("pm_attach");
    }
}
void ThreadGroup::gangSchedule() {
    /* There are two problems with real gang scheduling.
     *
     * 1) It needs to be run as root.
     * 2) It gang schedules the ENTIRE share group, not a subset of it.
     *
     * As a result, we just try to pin them to CPU's.
     */
    if(nprocessors==1)
	return;
    for(int i=0;i<ngroups;i++)
	groups[i]->gangSchedule();
    for(int i=0;i<nthreads;i++){
	threads[i]->migrate(last_processor++);
	if(last_processor >= nprocessors)
	    last_processor=0;
    }
}

void Thread_run(Thread* t) {
    t->run_body();
}

static void run_threads(void* priv_v, size_t) {
    Thread_private* priv=(Thread_private*)priv_v;
    install_signal_handlers();
    for(;;){
	/* Wait to be started... */
	priv->state=STATE_IDLE;
	if(uspsema(priv->startup) == -1){
	    perror("uspsema");
	    Thread::niceAbort();
	}
	thread_local->current_thread=priv->thread;
	priv->state=STATE_RUNNING;
	Thread_run(priv->thread);
	priv->state=STATE_SHUTDOWN;
	Thread_shutdown(priv->thread);
	priv->state=STATE_IDLE;
    }
}

void Thread_shutdown(Thread* thread) {
    Thread_private* priv=thread->priv;
    int pid=getpid();

    if(usvsema(priv->done) == -1){
	perror("usvsema");
	Thread::niceAbort();
    }

    // Wait to be deleted...
    if(priv->delete_ready){
	if(uspsema(priv->delete_ready) == -1) {
	    perror("uspsema");
	    Thread::niceAbort();
	}
    }
    // Allow this thread to run anywhere...
    if(thread->cpu != -1)
	thread->migrate(-1);

    delete thread;

    priv->thread=0;
    thread_local->current_thread=0;
    lock_scheduler();
    /* Remove it from the active queue */
    int i;
    for(i=0;i<nactive;i++){
	if(active[i]==priv)
	    break;
    }
    for(i++;i<nactive;i++){
	active[i-1]=active[i];
    }
    nactive--;
    if(priv->pid != main_pid){
	idle[nidle]=priv;
	nidle++;
    } else {
	idle_main=priv;
    }
    unlock_scheduler();
    Thread::check_exit();
    if(pid == main_pid){
	priv->state=STATE_PROGRAM_EXIT;
	if(uspsema(main_sema) == -1){
	    perror("uspsema");
	    Thread::niceAbort();
	}
    }
}

void Thread::check_exit() {
    lock_scheduler();
    int done=true;
    for(int i=0;i<nactive;i++){
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

void Thread::exitAll(int code) {
    exiting=true;
    exit_code=code;

    // We want to do this:
    //   kill(0, SIGQUIT);
    // but that has the unfortunate side-effect of sending a signal
    // to things like par, perfex, and other things that somehow end
    // up in the same share group.  So we have to kill each process
    // independently...
    pid_t me=getpid();
    for(int i=0;i<nidle;i++){
	if(idle[i]->pid != me)
	    kill(idle[i]->pid, SIGQUIT);
    }
    for(int i=0;i<nactive;i++){
	if(active[i]->pid != me)
	    kill(active[i]->pid, SIGQUIT);
    }
    if(idle_main && idle_main->pid != me)
	kill(idle_main->pid, SIGQUIT);
    kill(me, SIGQUIT);
    fprintf(stderr, "Should not reach this point!\n");
}

/*
 * Startup the given thread...
 */
void Thread::os_start(bool stopped) {
    /* See if there is a thread waiting around ... */
    if(!initialized){
	Thread::initialize();
    }
    lock_scheduler();
    if(nidle){
	nidle--;
	priv=idle[nidle];
    } else {
	priv=new Thread_private;
	priv->stacklen=THREAD_STACKSIZE;
	priv->stackbot=(caddr_t)mmap(0, priv->stacklen, PROT_READ|PROT_WRITE,
				     MAP_SHARED, devzero_fd, 0);
	priv->sp=priv->stackbot+priv->stacklen-1;
	if((long)priv->sp == -1){
	    perror("mmap");
	    error("Not enough memory for thread stack");
	    return;
	}
	priv->startup=usnewsema(arena, 0);
	priv->done=usnewsema(arena, 0);
	priv->delete_ready=usnewsema(arena, 0);
	priv->state=STATE_STARTUP;
	priv->bstacksize=0;
	priv->pid=sprocsp(run_threads, PR_SALL, priv, priv->sp, priv->stacklen);
	if(priv->pid == -1){
	    perror("sprocsp");
	    error("Cannot start new thread");
	    return;
	}
	int imld = (nactive >= nprocessors) ? nprocessors-1 : nactive;
	imld /= 2;
	int err = process_mldlink( priv->pid, mlds[imld], RQMODE_ADVISORY );
	if (err < 0)
	  perror("process_mldlink");
    }
    priv->thread=this;
    active[nactive]=priv;
    nactive++;
    unlock_scheduler();
    if(stopped){
	if(blockproc(priv->pid) != 0){
	    perror("blockproc");
	    error("Cannot block new thread");
	}
    }
    /* The thread is waiting to be started, release it... */
    if(usvsema(priv->startup) == -1){
	perror("usvsema");
	Thread::niceAbort();
    }
    //Thread::event(this, THREAD_START);
}

void Thread::setPriority(int pri) {
    /*
     * Priorities are disabled currently, because they only
     * work from root.
     */
    priority=pri;
}

void Thread::stop() {
    if(blockproc(priv->pid) != 0){
	perror("blockproc");
	error("Cannot block thread");
    }
}

void Thread::resume() {
    if(unblockproc(priv->pid) != 0){
	perror("unblockproc");
	error("Cannot unblock thread");
    }
}

void Thread::detach() {
    if(usvsema(priv->delete_ready) == -1){
	perror("usvsema");
	Thread::niceAbort();
    }
}    

void Thread::join() {
    Thread* us=Thread::currentThread();
    int os=push_bstack(us->priv, STATE_JOINING, threadname);
    if(uspsema(priv->done) == -1){
	perror("usvsema");
	Thread::niceAbort();
    }
    pop_bstack(us->priv, os);
    detach();
}

/*
 * Return the current thread...
 */
Thread* Thread::currentThread()
{
    return thread_local->current_thread;
}


int Thread::numProcessors()
{
    static int nproc=-1;
    if(nproc==-1){
	char* np=getenv("SCI_NPROCESSORS");
	if(np){
	    nproc = atoi(np);
	    if(nproc<=0)
		nproc=1;
	} else {
	    nproc = (int)sysmp(MP_NAPROCS);
	}
    }
    return nproc;
}

SysClock Thread::currentTicks()
{
    if(timer_32bit){
	for(;;){
	    unsigned high=iotimer_high;
	    unsigned ohigh=high;
	    unsigned low=*iotimer_addr32;
	    if((low&TOPBIT) != (high&TOPBIT)){
		// Possible rollover...
		if(!(low&TOPBIT))
		    high++;
	    }
	    if (ohigh == iotimer_high) {
		return ((long long)(high&(~TOPBIT))<<32|(long long)low)-orig_timer;
	    }
	    fprintf(stderr, "ROLLOVER loop around...\n");
	}
    } else {
#if _MIPS_ISA == _MIPS_ISA_MIPS1 || _MIPS_ISA ==  _MIPS_ISA_MIPS2
	while (1) {
	    unsigned high = *iotimer_addr;
	    unsigned low = *(iotimer_addr + 1);
	    if (high == *iotimer_addr) {
		return ((long long)high<<32|(long long)low)-orig_timer;
	    }
	}
#else
	return *iotimer_addr-orig_timer;
#endif
    }
}

double Thread::currentSeconds() {
    return Thread::currentTicks()*ticks_to_seconds;
}

double Thread::secondsPerTick() {
    return ticks_to_seconds;
}

double Thread::ticksPerSecond() {
    return seconds_to_ticks;
}

void Thread::waitUntil(double seconds) {
    waitFor(seconds-currentSeconds());
}

void Thread::waitFor(double seconds) {
    if(seconds<=0)
	return;
    static long tps=0;
    if(tps==0)
	tps=CLK_TCK;
    long ticks=(long)(seconds*(double)tps);
    while (ticks != 0){
	ticks=sginap(ticks);
    }
}

void Thread::waitUntil(SysClock time) {
    waitFor(time-currentTicks());
}

void Thread::waitFor(SysClock time) {
    if(time<=0)
	return;
    static double tps=0;
    if(tps==0)
	tps=(double)CLK_TCK*ticks_to_seconds;
    int ticks=time*tps;
    while (ticks != 0){
	ticks=(int)sginap(ticks);
    }
}

/*
 * Migrate the thread to a CPU.
 */
void Thread::migrate(int proc) {
#if 0
    if(proc==-1){
	if(sysmp(MP_RUNANYWHERE_PID, priv->pid) == -1){
	    perror("sysmp - MP_RUNANYWHERE_PID");
	}
    } else {
	if(sysmp(MP_MUSTRUN_PID, proc, priv->pid) == -1){
	    perror("sysmp - MP_MUSTRUN_PID");
	}
    }
#endif
    cpu=proc;
}

/*
 * Mutex implementation
 */

Mutex::Mutex(const char* name) : name(name) {
    if(!initialized){
	Thread::initialize();
    }
    priv=(Mutex_private*)usnewlock(arena);
    if(!priv){
	perror("usnewlock");
	Thread::niceAbort();
    }
}

Mutex::~Mutex() {
    usfreelock((ulock_t)priv, arena);
}

void Mutex::lock() {
    Thread* t=Thread::currentThread();
    int os;
    Thread_private* p=0;
    if(t){
	p=t->priv;
	os=push_bstack(p, STATE_BLOCK_MUTEX, name);
    }
    if(ussetlock((ulock_t)priv) == -1){
	perror("ussetlock");
	Thread::niceAbort();
    }
    if(t)
	pop_bstack(p, os);
}

bool Mutex::try_lock() {
    int st=uscsetlock((ulock_t)priv, 100);
    if(st==-1){
	perror("uscsetlock");
	Thread::niceAbort();
    }
    return st!=0;
}

void Mutex::unlock() {
    if(usunsetlock((ulock_t)priv) == -1){
	perror("usunsetlock");
	Thread::niceAbort();
    }
}

/*
 * PoolMutex implementation
 */

PoolMutex::PoolMutex() {
    if(ussetlock((ulock_t)poolmutex_lock) == -1){
	perror("ussetlock");
	Thread::niceAbort();
    }
    poolmutex_count++;
    poolmutex_average=(2*poolmutex_count)/N_POOLMUTEX;
    for(;;){
	mutex_idx=poolmutex_idx++;
	if(poolmutex_idx==N_POOLMUTEX)
	    poolmutex_idx=0;
	// If the number of users is more than twice the average,
	// then try the next mutex in the pool.  This will always
	// succeed eventually, and I doubt it will ever even get
	// through the loop more than once.
	if(poolmutex_users[mutex_idx] <= poolmutex_average)
	    break;
    }
    poolmutex_users[mutex_idx]++;
    if(usunsetlock((ulock_t)poolmutex_lock) == -1){
	perror("usunetlock");
	Thread::niceAbort();
    }
}

PoolMutex::~PoolMutex() {
    if(ussetlock((ulock_t)poolmutex_lock) == -1){
	perror("ussetlock");
	Thread::niceAbort();
    }
    poolmutex_users[mutex_idx]--;
    poolmutex_count--;
    poolmutex_average=(2*poolmutex_count)/N_POOLMUTEX;
    if(usunsetlock((ulock_t)poolmutex_lock) == -1){
	perror("usunetlock");
	Thread::niceAbort();
    }
}

void PoolMutex::lock() {
    Thread* t=Thread::currentThread();
    Thread_private* p=t->priv;
    int os=push_bstack(p, STATE_BLOCK_POOLMUTEX, poolmutex_names[mutex_idx]);
    if(ussetlock((ulock_t)poolmutex[mutex_idx]) == -1){
	perror("ussetlock");
	Thread::niceAbort();
    }
    pop_bstack(p, os);
}

bool PoolMutex::try_lock() {
    // Note - the Irix 6.2 manpage is wrong about what this returns...
    int st=uscsetlock((ulock_t)poolmutex[mutex_idx], 100);
    if(st==-1){
	perror("uscsetlock");
	Thread::niceAbort();
    }
    return st==0;
}

void PoolMutex::unlock() {
    if(usunsetlock((ulock_t)poolmutex[mutex_idx]) == -1){
	perror("usunsetlock");
	Thread::niceAbort();
    }
}

/*
 * Semaphore implementation
 */
struct Semaphore_private {
    usema_t* semaphore;
};

Semaphore::Semaphore(const char* name, int count)
: name(name)
{
    if(!initialized){
	Thread::initialize();
    }
    priv=new Semaphore_private;
    priv->semaphore=usnewsema(arena, count);
    if(!priv->semaphore){
	perror("usnewsema");
	Thread::niceAbort();
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
    Thread_private* p=Thread::currentThread()->priv;
    int oldstate=push_bstack(p, STATE_BLOCK_SEMAPHORE, name);
    if(uspsema(priv->semaphore) == -1){
	perror("upsema");
	Thread::niceAbort();
    }
    pop_bstack(p, oldstate);
}

bool Semaphore::tryDown()
{
    int stry=uscpsema(priv->semaphore);
    if(stry == -1){
	perror("upsema");
	Thread::niceAbort();
    }
    return stry;
}

void Semaphore::up()
{
    if(usvsema(priv->semaphore) == -1){
	perror("usvsema");
	Thread::niceAbort();
    }
}

struct Barrier_private {
    Barrier_private();

    // These variables used only for single processor implementation
    Mutex mutex;
    ConditionVariable cond0;
    ConditionVariable cond1;
    int cc;
    int nwait;

    // Only for MP implementation
    fetchop_var_t* pvar;
    char pad[128];
    int flag;  // We want this on it's own cache line
    char pad2[128];
};

Barrier_private::Barrier_private()
: cond0("Barrier condition 0"), cond1("Barrier condition 1"),
  mutex("Barrier lock"), nwait(0), cc(0)
{
    if(nprocessors > 1){
	flag=0;
	pvar=fetchop_alloc(reservoir);
	if(!pvar){
	    perror("fetchop_alloc");
	    Thread::niceAbort();
	}
	storeop_store(pvar, 0);
    }
}   

Barrier::Barrier(const char* name, int nthreads)
 : name(name), nthreads(nthreads), threadGroup(0)
{
    if(!initialized){
	Thread::initialize();
    }
    priv=new Barrier_private;
}

Barrier::Barrier(const char* name, ThreadGroup* threadGroup)
 : name(name), nthreads(0), threadGroup(threadGroup)
{
    if(!initialized){
	Thread::initialize();
    }
    priv=new Barrier_private;
}

Barrier::~Barrier()
{
    delete priv;
}

void Barrier::wait()
{
    int n=threadGroup?threadGroup->nactive(true):nthreads;
    Thread_private* p=Thread::currentThread()->priv;
    int oldstate=push_bstack(p, STATE_BLOCK_BARRIER, name);
    if(nprocessors > 1){
	int gen=priv->flag;
	fetchop_var_t val=fetchop_increment(priv->pvar);
	if(val == n-1){
	    storeop_store(priv->pvar, 0);
	    priv->flag++;
	}
	while(priv->flag==gen)
	    /* spin */ ;
    } else {
	priv->mutex.lock();
	ConditionVariable& cond=priv->cc?priv->cond0:priv->cond1;
	priv->nwait++;
	if(priv->nwait == n){
	    // Wake everybody up...
	    priv->nwait=0;
	    priv->cc=1-priv->cc;
	    cond.cond_broadcast();
	} else {
	    cond.wait(priv->mutex);
	}
	priv->mutex.unlock();
    }
    pop_bstack(p, oldstate);
}

#include <sys/profil.h>
#include <sys/procfs.h>
#include <strings.h>
#include <libelf.h>
#include <dlfcn.h>
#include <dem.h>
#include <errno.h>

static Mutex proflock("profile startup lock");
static Semaphore profsema("profile startup semaphore", 0);
static struct prof* pass_profp;
static int pass_profcnt;
static struct timeval* pass_tvp;
static int pass_flags;
static int prof_status;
static unsigned short _junk;

static void handle_profile(int, int, sigcontext_t*) {
    if(pass_profp){
	// Start it.
	prof_status=sprofil(pass_profp, pass_profcnt, pass_tvp, pass_flags);
    } else {
	// Cancel it.
	prof_status=profil(&_junk, 4, 0, 0);
    }
    profsema.up();
}

static void pass_sprofil(FILE* out, int pid, struct prof* profp, int profcnt,
			 struct timeval* tvp, unsigned int flags) {
    proflock.lock();
    pass_profp=profp;
    pass_profcnt=profcnt;
    pass_tvp=tvp;
    pass_flags=flags;

    if(kill(pid, SIGUSR1) != 0){
	fprintf(out, "Error sending signal to thread: %d\n", pid);
	fflush(out);
	proflock.unlock();
	return;
    }

    // When the other thread has called sprofil, it will do an up here...
    profsema.down();
    if(prof_status == -1){
	fprintf(out, "ERROR: Cannot turn off profiling\n");
	fflush(out);
	proflock.unlock();
	return;
    }
    proflock.unlock();
}

struct ProfProc {
    char* name;
    char* start_addr;
    size_t length;
    unsigned long instr_hit;
    unsigned long  total_hit;
};

struct ProfRegion {
    int elf_fd;
    unsigned long size;
    caddr_t start;
    unsigned int* count_start;

    char* strtab;
};

Elf_Scn* get_section_byname(Elf* elf, Elf32_Half ndx, char* wantname) {
    Elf_Scn* scn = 0;
    while ((scn = elf_nextscn(elf, scn)) != 0){
	Elf32_Shdr* shdr;
	if ((shdr = elf32_getshdr(scn)) != 0) {
	    char* name = elf_strptr(elf, ndx, (size_t)shdr->sh_name);
	    if(name != 0 && strcmp(name, wantname) == 0)
		return scn;
	}
    }
    return scn;
}

static void find_symbols(void* dlhandle, ProfRegion* region, ProfProc* procs, int& nprocs) {
    Elf_Cmd cmd=ELF_C_READ_MMAP;
    if (elf_version(EV_CURRENT) == EV_NONE) {
	/* library out of date */
	int err=elf_errno();
	fprintf(stderr, "Error in elf_version: %s\n", elf_errmsg(err));
	return;
    }
    Elf* arf=elf_begin(region->elf_fd, cmd, 0);
    if(!arf){
	int err=elf_errno();
	fprintf(stderr, "Error in elf_begin: %s\n", elf_errmsg(err));
	return;
    }
    Elf* elf;
    while(elf = elf_begin(region->elf_fd, cmd, arf)){
	Elf32_Ehdr* ehdr;
	if ((ehdr = elf32_getehdr(elf)) != 0){
	    Elf32_Half ndx = ehdr->e_shstrndx;
	    
	    Elf_Scn* str_scn=get_section_byname(elf, ndx, ELF_DYNSTR);
	    if(!str_scn){
		fprintf(stderr, "Can't find .dynstr\n");
	    }
	    Elf_Data* data=elf_getdata(str_scn, 0);
	    region->strtab=new char[data->d_size];
	    bcopy(data->d_buf, region->strtab, data->d_size);
	    Elf_Scn* sym_scn=get_section_byname(elf, ndx, ELF_DYNSYM);
	    if(!sym_scn){
		fprintf(stderr, "Can't find .dynsym\n");
	    }

	    if(str_scn && sym_scn){
		Elf32_Shdr* shdr;
		int count=0;
		if ((shdr = elf32_getshdr(sym_scn)) != 0) {
		    Elf_Data* data=0;
			
		    while ((data = elf_getdata(sym_scn, data)) != 0){
			count++;
			unsigned long nsyms=data->d_size/shdr->sh_entsize;
			if(data->d_type != ELF_T_SYM){
			    fprintf(stderr, "Not symbols?\n");
			}
			Elf32_Sym* sym=(Elf32_Sym*)data->d_buf;
			unsigned long dso_off=0;
			int i;
			for(i=0;i<nsyms;i++){
			    if(ELF32_ST_TYPE(sym[i].st_info) == STT_FUNC
			       && sym[i].st_shndx != SHN_UNDEF){
				char* sname = region->strtab+sym[i].st_name;
				char* dladdr=(char*)dlsym(dlhandle, sname);
				if(dladdr && dladdr > region->start && dladdr <= (region->start+region->size)){
				    char* addr=(char*)sym[i].st_value;
				    dso_off=dladdr-addr;
				    if(dso_off){
					fprintf(stderr, "Keyed offset of %lx to %s\n", dso_off, sname);
				    }
				    break;
				}
			    }
			}
			if(i==nsyms){
			    fprintf(stderr, "WARNING: no offset found?\n");
			}
			for(i=0;i<nsyms;i++){
			    if(ELF32_ST_TYPE(sym[i].st_info) == STT_FUNC
			       && sym[i].st_shndx != SHN_UNDEF){
				char* addr=(char*)sym[i].st_value+dso_off;
				if(addr && addr > region->start && addr <= (region->start+region->size)){
				    if(nprocs < 65536){
					ProfProc* proc=&procs[nprocs];
					proc->name=region->strtab+sym[i].st_name;
					proc->start_addr=addr;
					proc->length=sym[i].st_size;
					nprocs++;
				    }
				}
			    }
			}
		    }
		} else {
		    fprintf(stderr, "didn't get shdr\n");
		}
	    } else {
		fprintf(stderr, "Couldn't find section?\n");
	    }
	} else {
	    fprintf(stderr, "Error in elf32_getehdr\n");
	    return;
	}
	cmd = elf_next(elf);
	elf_end(elf);
    }
    elf_end(arf);
    close(region->elf_fd);
}

int lookup_proc(ProfRegion* regions, int nregions,
		ProfProc* procs, int nprocs,
		unsigned int* stataddr) {
    char* addr=0;
    for(int k=0;k<nregions;k++){
	ProfRegion* region=&regions[k];
	int off=stataddr-region->count_start;
	off<<=2;
	if(off <region->size){
	    addr=(char*)region->start+off;
	    break;
	}
    }
    if(!addr){
	fprintf(stderr, "Region not found at address: %p\n", stataddr);
    }
    for(int j=0;j<nprocs;j++){
	ProfProc* proc=&procs[j];
	if(addr >= proc->start_addr && addr < (proc->start_addr+proc->length)){
	    return j;
	}
    }
    return nprocs-1; // Special <static> entry...
}

void gather_stats(unsigned long ninstr,
		  unsigned int* stats, unsigned short* procidx,
		  ProfRegion* regions, int nregions,
		  ProfProc* procs, int nprocs,
		  ProfProc** sortedprocs, int& activeprocs,
		  unsigned long& nsamples) {
    for(unsigned long i=0;i<ninstr;i++){
	if(stats[i] != 0){
	    int pi=procidx[i];
	    if(pi== 0){
		procidx[i]=pi=lookup_proc(regions, nregions, procs, nprocs, &stats[i]);
		ProfProc* proc=&procs[pi];
		if(proc->instr_hit == 0)
		    sortedprocs[activeprocs++]=proc;
		proc->instr_hit++;
	    }
	    ProfProc* proc=&procs[pi];
	    proc->total_hit+=stats[i];
	    nsamples+=stats[i];
	    stats[i]=0;
	}
    }
}

int compare_procs(const void* p1, const void* p2) {
    ProfProc** proc1=(ProfProc**)p1;
    ProfProc** proc2=(ProfProc**)p2;
    return (int)((*proc2)->total_hit-(*proc1)->total_hit);
}

void Thread::profile(FILE* in, FILE* out) {
    int tid=-1;
    Thread_private* priv;
    lock_scheduler();
    while(tid==-1){
	print_threads(out, 0);
	fprintf(out, "\nProfile which thread? ");
	fflush(out);
	if(fscanf(in, "%d", &tid) !=1){
	    tid=-1;
	    fprintf(out, "Error reading response\n");
	    fflush(out);
	    int c=fgetc(in);
	    while(c != -1 && c != '\n')
		c=fgetc(in);
	    continue;
	}
	priv=find_thread_from_tid(tid);
	if(!priv){
	    fprintf(out, "Not a valid thread\n");
	    int c=fgetc(in);
	    while(c != -1 && c != '\n')
		c=fgetc(in);
	    tid=-1;
	}
    }
    unlock_scheduler();

    SysClock start_time=Thread::currentTicks();
    // Query the text segments in the process
    char buf[100];
    sprintf(buf, "/proc/%d", tid);
    int procfd=open(buf, O_RDONLY);
    if(procfd == -1){
	perror("open");
	fprintf(out, "Error opening: %s\n", buf);
	fflush(out);
	return;
    }
    int nmap;
    if(ioctl(procfd, PIOCNMAP, &nmap) == -1){
	perror("ioctl");
	fprintf(out, "Error querying number of segments\n");
	fflush(out);
	close(procfd);
	return;
    }

    nmap*=2; // Comfort zone...
    nmap+=2;
    prmap_sgi_t* mapbuf=new prmap_sgi_t[nmap];
    prmap_sgi_arg_t prmap;
    prmap.pr_vaddr=(caddr_t)mapbuf;
    prmap.pr_size=sizeof(prmap_sgi_t)*nmap;
    if(ioctl(procfd, PIOCMAP_SGI, &prmap) == -1){
	perror("ioctl");
	fprintf(out, "Error querying address space\n");
	fflush(out);
	close(procfd);
	return;
    }

    int profcnt=0;
    int i;
    size_t tsize=0;
    for(i=0;i<nmap;i++){
	prmap_sgi_t* map=&mapbuf[i];
	if(map->pr_size == 0)
	    break;
	if(map->pr_mflags & MA_EXEC){
	    profcnt++;
	    tsize+=map->pr_size;
        }
    }
    tsize+=sizeof(unsigned int);
    struct prof* profp=new struct prof[profcnt];
    ProfRegion* regions=new ProfRegion[profcnt];
    profcnt=0;
#if _MIPS_SZPTR==32
    unsigned long ninstr=tsize>>2;
#else
    unsigned long ninstr=tsize>>3;
#endif
    unsigned int* stats=new unsigned int[ninstr];
    bzero(stats, ninstr*sizeof(int));
    char* sp=(char*)stats;
    for(i=0;i<nmap;i++){
	prmap_sgi_t* map=&mapbuf[i];
	if(map->pr_size == 0)
	    break;
	if(map->pr_mflags & MA_EXEC){
	    caddr_t addr=map->pr_vaddr;
	    int fd=ioctl(procfd, PIOCOPENM, &addr);
	    if(fd == -1){
		perror("ioctl - PIOCOPENM");
	    } else {
		ProfRegion* region=&regions[profcnt];
		region->elf_fd=fd;
		region->size=map->pr_size;
		region->start=map->pr_vaddr;
		region->count_start=(unsigned int*)sp;

		profp[profcnt].pr_base=sp;
		sp+=map->pr_size;
		profp[profcnt].pr_size=map->pr_size;
		profp[profcnt].pr_off=(__psunsigned_t)map->pr_vaddr;
		profp[profcnt].pr_scale=0x10000;
		profcnt++;
	    }
	}
    }
    profp[profcnt].pr_base=sp;
    sp+=1;
    profp[profcnt].pr_size=sizeof(unsigned int);
    profp[profcnt].pr_off=0;
    profp[profcnt].pr_scale=0x00002;
    profcnt++;
    fprintf(out, "%x %x\n", &stats[ninstr], sp);
    fprintf(out, "Found %d text segments of total size: %d\n",
	    profcnt, tsize);
    fflush(out);
    close(procfd);

    // Set up Elf descriptors to access the symbol table...
    ProfProc* procs=new ProfProc[65536]; // Size of unsigned short
    int nprocs=1;
    void* dlhandle=dlopen(0, RTLD_LAZY);

    for(i=0;i<profcnt-1;i++){
	ProfRegion* region=&regions[i];
	find_symbols(dlhandle, region, procs, nprocs);
    }
    dlclose(dlhandle);
    if(nprocs >65534){
	fprintf(out, "Maximum number of procedures exceeded - some may not get profiled\n");
    }
    procs[nprocs++].name="<unknown static functions>";
    ProfProc** sortedprocs=new ProfProc*[nprocs];
    int activeprocs=0;
    for(i=0;i<nprocs+1;i++){
	procs[i].total_hit=procs[i].instr_hit=0;
    }

    // Setup dats structures and signal to the thread to have it call sprofil
    struct timeval tvp;
    pass_sprofil(out, tid, profp, profcnt, &tvp, PROF_UINT);

    unsigned short* procidx=new unsigned short[ninstr];
    bzero(procidx, sizeof(unsigned short)*ninstr);

    bool done=false;
    int delay_sec=1;
    fflush(out);
    double spt=Thread::secondsPerTick();
    SysClock ticks=Thread::currentTicks()-start_time;
    fprintf(out, "Setup took %g seconds\n\n", ticks*spt);
    unsigned long nsamples=0;
    while(!done){
	// select for 1 second
	int maxfd=fileno(in)+1;
	fd_set readfds;
	FD_ZERO(&readfds);
	FD_SET(fileno(in), &readfds);
	struct timeval timeout;
	timeout.tv_sec=delay_sec;
	timeout.tv_usec=0;
	int status=select(maxfd, &readfds, 0, 0, &timeout);
	SysClock start_time=Thread::currentTicks();
	if(status == -1){
	    // Error...
	    if(errno != EINTR){
		perror("select");
		return;
	    }
	} else if(status != 0){
	    char buf[100];
	    if(!fgets(buf, 100, in)){
		perror("fgets");
		return;
	    }
	    if(buf[0]=='q' || buf[0] == 'Q') {
		done=true;
	    } else if(buf[0]=='s' || buf[0] == 'S'){
		pass_sprofil(out, tid, profp, profcnt, &tvp, PROF_UINT);
	    } else if(buf[0]=='f' || buf[0]=='F'){
		pass_sprofil(out, tid, profp, profcnt, &tvp, PROF_UINT|PROF_FAST);
	    } else if(buf[0]=='z' || buf[0]=='Z'){
		bzero(stats, ninstr*sizeof(int));
	    } else  {
		fprintf(out, "Unknown command: %s", buf);
		fflush(out);
	    }
	}
	// Redraw the output...
	gather_stats(ninstr, stats, procidx, regions, profcnt-1,
		     procs, nprocs, sortedprocs, activeprocs, nsamples);
	qsort(sortedprocs, activeprocs, sizeof(ProfProc*), compare_procs);
	double ns=nsamples;
	fprintf(out, "\n\n\n\n\n");
	for(i=0;i<activeprocs;i++){
	    ProfProc* proc=sortedprocs[i];
	    //char outbuf[1024];
	    //int status=demangle(proc->name, outbuf);
	    double percent=100.*proc->total_hit/ns;
	    fprintf(out, "%6.02f%% %s [%d of %d instrs %.02f%%]\n",
		    percent, proc->name, proc->instr_hit, proc->length>>2,
		    100.*double(proc->instr_hit)/(proc->length>>2));
	}
	SysClock ticks=Thread::currentTicks()-start_time;
	double comptime=ticks*spt;
	fprintf(out, "\n%ld samples - profile took %g seconds\n",
		nsamples, comptime);
	fflush(out);
    }
    pass_sprofil(out, tid, 0, 0, 0, 0);
}

void Thread::alert(int) {
    fprintf(stderr, "Thread::alert not finished\n");
}

