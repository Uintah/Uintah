/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

/* Copyright (C) 1991-1999, 2000 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If not,
   write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

#ifdef __linux  /* This file is for linux only. */

#  include <sys/wait.h>
#  include <sys/types.h>

#  include <cstdio>
#  include <cstddef>
#  include <cstdlib>
#  include <unistd.h>
#  include <signal.h>
#  include <cerrno>
#  include <cstring>
#  include <pthread.h>

#  include <Core/Malloc/Allocator.h>

/*
 This file is straight out of the glibc-2.2.1 distribution, with the 
 exception that I've made some very minor changes: commented out some
 code, slightly modified other code and added some code.  Each instance
 of change is delimited with "/ * - cm <change-type> ... - cm * /".

 The reason for doing this is as a work around for the "fork/exec in pthread
 environment" bug introduced in the 2.1.2 and 2.1.3 releases of glibc 
 (in Linux).  The work around is to force the use of the symbol "__fork" 
 inside the libpthreads.so library rather than the symbol "fork" in the
 libc.so library.

 - Chris Moulding */

/* -- Dd for AIX xlC */
#ifdef _AIX
#  define	__environ	environ
#  define       __sigaction     sigaction
#  define       __sigemptyset   sigemptyset
#endif
/* -- Dd for PDT */
#ifdef PDT_PARSER
#  ifndef __sigemptyset
#    define __sigemptyset sigemptyset
#  endif /* __sigemptyset */
#endif /* PDT_PARSER */

/* -- end Dd */

/* - BW for PGI */
#ifdef __PGI
#  define __sigemptyset sigemptyset
#endif
/* BW */

/* This is for the Intel compiler */
#ifdef __INTEL_COMPILER
#  define __sigemptyset sigemptyset
#endif

#ifdef __digital__
#  define __environ environ
#endif

/* added by mc trying to fix hang bug on dual cpu machines */
#define WAITPID_CANNOT_BLOCK_SIGCHLD
#define NO_WAITPID
/* - mc */

/* - cm added */
extern "C" {
pid_t __fork(void);
}
/* - cm */

#define	SHELL_PATH	"/bin/sh"	/* Path of the shell.  */
#define	SHELL_NAME	"sh"		/* Name to give it.  */

/* Execute LINE as a shell command, returning its status.  */
/* - cm modified */
/*int
  __libc_system (const char *line)*/
/* returns 1 where there was a problem, otherwise the return value
   of the system call. -bigler
*/
static int
sci_system_linuxthreads(const char *line)
/* - cm */
{
  int status, save;
  pid_t pid;
  struct sigaction sa, intr, quit;
#ifndef WAITPID_CANNOT_BLOCK_SIGCHLD
  sigset_t block, omask;
#endif

  if (line == NULL)
    /* Check that we have a command processor available.  It might
       not be available after a chroot(), for example.  */
    return sci_system_linuxthreads ("exit 0") == 0;

  sa.sa_handler = SIG_IGN;
  sa.sa_flags = 0;
  __sigemptyset (&sa.sa_mask);

  if (sigaction (SIGINT, &sa, &intr) < 0)
    return -1;
  if (sigaction (SIGQUIT, &sa, &quit) < 0)
    {
      save = errno;
      (void) sigaction (SIGINT, &intr, (struct sigaction *) NULL);
      /* - cm modified 
        __set_errno (save); */
      errno = save;
      /* - cm */
      return -1;
    }

#ifndef WAITPID_CANNOT_BLOCK_SIGCHLD

/* SCO 3.2v4 has a bug where `waitpid' will never return if SIGCHLD is
   blocked.  This makes it impossible for `system' to be implemented in
   compliance with POSIX.2-1992.  They have acknowledged that this is a bug
   but I have not seen nor heard of any forthcoming fix.  */

  __sigemptyset (&block);
  __sigaddset (&block, SIGCHLD);
  save = errno;
  /* - cm modified
     if (__sigprocmask (SIG_BLOCK, &block, &omask) < 0) */
  if (sigprocmask (SIG_BLOCK, &block, &omask) < 0)
  /* - cm */
    {
      if (errno == ENOSYS)
	/* - cm modified
	   __set_errno (save); */
	errno = save;
        /* - cm */
      else
	{
	  save = errno;
	  (void) sigaction (SIGINT, &intr, (struct sigaction *) NULL);
	  (void) sigaction (SIGQUIT, &quit, (struct sigaction *) NULL);
	  /* - cm modified
	     __set_errno (save); */
	  errno = save;
	  /* - cm */
	  return -1;
	}
    }
/* - cm modified
   # define UNBLOCK __sigprocmask (SIG_SETMASK, &omask, (sigset_t *) NULL) */
#define UNBLOCK sigprocmask (SIG_SETMASK, &omask, (sigset_t *) NULL)
/* - cm */
#else
# define UNBLOCK 0
#endif

  // Create the pipe file destriptors.  This has to happen, so that
  // the parent knows that the child is done with the allocator.  The
  // parent leaves the fork call before the child is done forking.  If
  // the parent unlocks the allocator before the child is done then
  // another process has the oportunity to get a hold of the
  // allocator, thus causing the child to block.  These pipes provide
  // a mechanism for the child to tell the parent that it is done
  // forking.  In the event that vfork is actually implented (parent
  // blocks until child either calls exec or exit), you will no longer
  // need the pipe to communicate.
  int pipe_fd[2];
  if (pipe(pipe_fd) == -1) {
    fprintf(stderr,"sci_system.cc:Error in piping\n");
    return 1;
  }

#if !defined( DISABLE_SCI_MALLOC )
  // Recursively lock the allocator, to keep other threads out.
  SCIRun::LockAllocator( SCIRun::DefaultAllocator() );
#endif
  pid = __fork ();
  if (pid == (pid_t) 0)
    {
      /* Child side.  */

      // Close unneeded read pipe.
      if (close(pipe_fd[0]) == -1) {
	fprintf(stderr, "sci_system.cc:Error in closing read pipe in child\n");
	fprintf(stderr, "sci_system.cc:If this actually has a problem, the parent will block.\n");
	exit(1);
      }

      // I'm not sure why unlocking this causes a problem in the
      //child.  I have seen things on the web that indicate that this
      //mutex may need to be reset before it can be used again.  At
      //any rate it breaks with it in, and doesn't when it is
      //commented out.  - James Bigler

      //      UnLockAllocator(DefaultAllocator());

      // Send signal to parent that we are past the fork
      char to_parent[1] = {'n'};
      if (write(pipe_fd[1], to_parent, 1) != 1) {
	fprintf(stderr, "sci_system.cc:Error in writing to pipe in child\n");
	fprintf(stderr, "sci_system.cc:If this actually has a problem, the parent will block.\n");
	exit(1);
      }

      if (close(pipe_fd[1]) == -1) {
	fprintf(stderr, "sci_system.cc:Error in closing write pipe in child\n");
	exit(1);
      }
      
      const char *new_argv[4];
      new_argv[0] = SHELL_NAME;
      new_argv[1] = "-c";
      new_argv[2] = line;
      new_argv[3] = NULL;

      /* Restore the signals.  */
      (void) sigaction (SIGINT, &intr, (struct sigaction *) NULL);
      (void) sigaction (SIGQUIT, &quit, (struct sigaction *) NULL);
      (void) UNBLOCK;

      /* Exec the shell.  */
      /* - cm modified 
	 (void) __execve (SHELL_PATH, (char *const *) new_argv, __environ); */
      (void) execve (SHELL_PATH, (char *const *) new_argv, __environ);
      /* - cm */

      _exit (127);
    }
  else if (pid < (pid_t) 0)
    {
      /* The fork failed.  */
      status = -1;
#if !defined( DISABLE_SCI_MALLOC )
      SCIRun::UnLockAllocator( SCIRun::DefaultAllocator() );
#endif
    }
  else
    /* Parent side.  */
    {
      
      // Wait for signal from child that it is past the fork
      if (close(pipe_fd[1]) == -1) {
	fprintf(stderr, "sci_system.cc:Error in closing write pipe in parent\n");
	return 1;
      }
      char from_child[1];
      if (read(pipe_fd[0], from_child, 1) != 1) {
	fprintf(stderr, "sci_system.cc:Error in reading from pipe in parent\n");
#if !defined( DISABLE_SCI_MALLOC )
	SCIRun::UnLockAllocator( SCIRun::DefaultAllocator() );
#endif
	return 1;
      }
      // We can unlock the allocator, because the child has
#if !defined( DISABLE_SCI_MALLOC )
      SCIRun::UnLockAllocator( SCIRun::DefaultAllocator() );
#endif
      if (close(pipe_fd[0]) == -1) {
	fprintf(stderr, "sci_system.cc:Error in closing read pipe in parent\n");
	return 1;
      }
      
#ifdef	NO_WAITPID
      pid_t child;
      do
	{
	  /* - cm modified
	     child = __wait (&status); */
	  child = wait (&status);
	  /* - cm */
	  if (child <= -1 && errno != EINTR)
	    {
	      status = -1;
	      break;
	    }
	  /* Note that pid cannot be <= -1 and therefore the loop continues
	     when __wait returned with EINTR.  */
	}
      while (child != pid);
#else
      int n;

      do
	/* - cm modified
	   n = __waitpid (pid, &status, 0); */
	n = waitpid (pid, &status, 0);      
        /* - cm */
      while (n == -1 && errno == EINTR);
      
      if (n != pid)
	status = -1;
#endif
    }

  save = errno;
  if ((sigaction (SIGINT, &intr, (struct sigaction *) NULL) |
       sigaction (SIGQUIT, &quit, (struct sigaction *) NULL) |
       UNBLOCK) != 0)
    {
      if (errno == ENOSYS)
      /* - cm modified
	__set_errno (save); */
	errno = save;
      /* - cm */
      else
	return -1;
    }
  return status;
}
/* -cm commented
  weak_alias (__libc_system, system)
   -cm */


int
sci_system(const char *line)
{
  // We only need to wrap system on linux builds with linuxthreads
  // enabled and not nptl.
#ifdef _CS_GNU_LIBPTHREAD_VERSION
  const size_t n = confstr(_CS_GNU_LIBPTHREAD_VERSION, NULL, (size_t) 0);
  if (n > 0)
  {
    char *buf = scinew char[n];
    confstr(_CS_GNU_LIBPTHREAD_VERSION, buf, n);
    const bool nptl = strncmp(buf, "NPTL", 4) == 0;
    delete buf;
    if (nptl)
    {
      return system(line);
    }
  }
#endif
  return sci_system_linuxthreads(line);
}


#endif /* for linux only */
