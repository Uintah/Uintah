/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */

#include <Core/Thread/Thread_unix.h>
#include <cstdio>
#ifdef _WIN32
#include <signal.h>
#include <cerrno>
#else
#include <sys/errno.h>
#include <sys/signal.h>
#endif

char*
Core_Thread_signal_name(int sig, void* addr)
{
    static char buf[1000];
    switch(sig){
    case SIGINT:
	sprintf(buf, "SIGINT (interrupt)");
	break;
    case SIGILL:
	sprintf(buf, "SIGILL at address %p (illegal instruction)", addr);
	break;
    case SIGABRT:
	sprintf(buf, "SIGABRT (Abort)");
	break;
    case SIGSEGV:
	sprintf(buf, "SIGSEGV at address %p (segmentation violation)", addr);
	break;
    case SIGTERM:
	sprintf(buf, "SIGTERM (killed)");
	break;
    case SIGFPE:
	sprintf(buf, "SIGFPE (floating point exception)");
	break;
#ifdef SIGBREAK
    case SIGBREAK:
	sprintf(buf, "SIGBREAK (CTRL-Break sequence)");
	break;
#endif

// these signals don't exist in win32
#ifndef _WIN32 
    case SIGHUP:
	sprintf(buf, "SIGHUP (hangup)");
	break;
    case SIGQUIT:
	sprintf(buf, "SIGQUIT (quit)");
	break;
    case SIGTRAP:
	sprintf(buf, "SIGTRAP (trace trap)");
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
    case SIGKILL:
	sprintf(buf, "SIGKILL (kill)");
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
#endif
    default:
	sprintf(buf, "unknown signal(%d)", sig);
	break;
    }
    return buf;
}
