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

#ifndef __sgi  /* This file is for linux only. */

#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/types.h>
#include <errno.h>

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

/* - cm commented out
#ifndef	HAVE_GNU_LD 
#define	__environ	environ
#endif
   - cm
*/

/* - cm added */
pid_t __fork(void);
/* - cm */

#define	SHELL_PATH	"/bin/sh"	/* Path of the shell.  */
#define	SHELL_NAME	"sh"		/* Name to give it.  */

/* Execute LINE as a shell command, returning its status.  */
/* - cm modified */
/*int
  __libc_system (const char *line)*/
int
sci_system(const char *line)
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
    return sci_system ("exit 0") == 0;

  sa.sa_handler = SIG_IGN;
  sa.sa_flags = 0;
  __sigemptyset (&sa.sa_mask);

  if (__sigaction (SIGINT, &sa, &intr) < 0)
    return -1;
  if (__sigaction (SIGQUIT, &sa, &quit) < 0)
    {
      save = errno;
      (void) __sigaction (SIGINT, &intr, (struct sigaction *) NULL);
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
	  (void) __sigaction (SIGINT, &intr, (struct sigaction *) NULL);
	  (void) __sigaction (SIGQUIT, &quit, (struct sigaction *) NULL);
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

  pid = __fork ();
  if (pid == (pid_t) 0)
    {
      /* Child side.  */
      const char *new_argv[4];
      new_argv[0] = SHELL_NAME;
      new_argv[1] = "-c";
      new_argv[2] = line;
      new_argv[3] = NULL;

      /* Restore the signals.  */
      (void) __sigaction (SIGINT, &intr, (struct sigaction *) NULL);
      (void) __sigaction (SIGQUIT, &quit, (struct sigaction *) NULL);
      (void) UNBLOCK;

      /* Exec the shell.  */
      /* - cm modified
	 (void) __execve (SHELL_PATH, (char *const *) new_argv, __environ); */
      (void) execve (SHELL_PATH, (char *const *) new_argv, __environ);
      /* - cm */
      _exit (127);
    }
  else if (pid < (pid_t) 0)
    /* The fork failed.  */
    status = -1;
  else
    /* Parent side.  */
    {
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
  if ((__sigaction (SIGINT, &intr, (struct sigaction *) NULL) |
       __sigaction (SIGQUIT, &quit, (struct sigaction *) NULL) |
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

#endif /* for linux only */
