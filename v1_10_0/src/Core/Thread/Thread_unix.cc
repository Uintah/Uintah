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
*/


/*
 *  Thread_unix.cc: Utilities for all unix implementations of the
 * 		    the thread library
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Thread/Thread_unix.h>
#include <stdio.h>
#include <sys/errno.h>
#include <sys/signal.h>


char*
Core_Thread_signal_name(int sig, void* addr)
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
	sprintf(buf, "SIGILL at address %p (illegal instruction)", addr);
	break;
    case SIGTRAP:
	sprintf(buf, "SIGTRAP (trace trap)");
	break;
    case SIGABRT:
	sprintf(buf, "SIGABRT (Abort)");
	break;
#ifdef SIGEMT
    case SIGEMT:
	sprintf(buf, "SIGEMT (Emulation Trap)");
	break;
#endif
#ifdef SIGIOT
#if SIGEMT != SIGIOT && SIGIOT != SIGABRT
    case SIGIOT:
	sprintf(buf, "SIGIOT (IOT Trap)");
	break;
#endif
#endif
    case SIGBUS:
	sprintf(buf, "SIGBUS at address %p (bus error)", addr);
	break;
    case SIGFPE:
	sprintf(buf, "SIGFPE (floating point exception)");
	break;
    case SIGKILL:
	sprintf(buf, "SIGKILL (kill)");
	break;
    case SIGSEGV:
	sprintf(buf, "SIGSEGV at address %p (segmentation violation)", addr);
	break;
#ifdef SIGSYS
    case SIGSYS:
	sprintf(buf, "SIGSYS (bad argument to system call)");
	break;
#endif
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
    case SIGCHLD:
	sprintf(buf, "SIGCLD (death of a child)");
	break;
#ifdef SIGPWR
    case SIGPWR:
	sprintf(buf, "SIGPWR (power fail restart)");
	break;
#endif
    case SIGWINCH:
	sprintf(buf, "SIGWINCH (window size changes)");
	break;
    case SIGURG:
	sprintf(buf, "SIGURG (urgent condition on IO channel)");
	break;
    case SIGIO:  // Also SIGPOLL
	sprintf(buf, "SIGIO/SIGPOLL (i/o possible)");
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
