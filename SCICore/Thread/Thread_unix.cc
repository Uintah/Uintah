/*
 * Return the statename for p
 */
static
const char*
getstate(Thread_private* p)
{
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

static
void
print_threads(FILE* fp, int print_idle)
{
    for(int i=0;i<nactive;i++){
	Thread_private* p=active[i];
	const char* tname=p->thread?p->thread->getThreadName():"???";
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

static
char*
signal_name(int sig, int code, caddr_t addr)
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

/*
 * Handle sigquit - usually sent by control-C
 */
static
void
handle_quit(int sig, int code, sigcontext_t*)
{
    if(exiting){
	if(getpid() == main_pid){
	    wait_shutdown();
	}
	exit(exit_code);
    }
    // Try to acquire a lock.  If we can't, then assume that somebody
    // else already caught the signal...
    Thread* self=Thread::self();
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

    const char* tname=self?self->getThreadName():"main?";

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
static
void
handle_abort_signals(int sig, int code, sigcontext_t* context)
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

    Thread* self=Thread::self();
    const char* tname=self?self->getThreadName():"idle or main";
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
static
void
handle_thread_death(int, int, sigcontext_t*)
{
    if(exiting){
	if(getpid() == main_pid)
	    return;
	exit(exit_code);
    }
    Thread* self=Thread::self();
    if(!self)
	return; // This is an idle thread...
    Thread_private* priv=self->getPrivate();
    priv->state=STATE_DIED;
    if(priv->pid != main_pid)
	return;
}

/*
 * Setup signals for the current thread
 */
static
void
install_signal_handlers()
{
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

static
void
handle_alrm(int, int, sigcontext_t*)
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

