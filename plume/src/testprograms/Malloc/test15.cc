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

#ifdef __linux /* only linux seems to have a problem with this */
#include <Core/Util/sci_system.h>
#include <stdlib.h>
#include <stdio.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Malloc/Allocator.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/types.h>
#include <errno.h>

#include <pthread.h>

using namespace SCIRun;

/*
  This program is designed to test a weird race condition dealing with
  threads calling fork.  One thread continually calls alloc and free, while
  the other thread attemps to call fork and exec repeadidly.  The program
  should hang.  And the child process will be hung waiting on the
  Allocator::lock() function.
*/

class AllocALot: public Runnable {
public:
  char c;
  AllocALot(char c = '+'): c(c) {}
  virtual ~AllocALot() {}
  
  virtual void run() {
    unsigned long counter = 0;
    for(;;counter++) {
      void *data = malloc(1024);
      free(data);
      if (drand48() < 1e-2) {
	printf("%c",c);
	fflush(stdout);
      }
    }
  }
};

#ifdef __digital__
#define __environ environ
#endif

#define WAITPID_CANNOT_BLOCK_SIGCHLD
#define NO_WAITPID

extern "C" {
pid_t __fork(void);
}

#define	SHELL_PATH	"/bin/sh"	/* Path of the shell.  */
#define	SHELL_NAME	"sh"		/* Name to give it.  */

class ForkALot: public Runnable {
public:
  char c;
  char buf [20];
  ForkALot(char c = '*'): c(c)
  {
    sprintf(buf, "printf \\%%c '%c'", c);
  }
  virtual ~ForkALot() {}
  
  virtual void run() {
    unsigned long counter = 0;
    for(;;counter++) {
#if 1
      //      sci_system("printf \%c '*'");
      //      LockAllocator(DefaultAllocator());
      sci_system(buf);
      //      UnLockAllocator(DefaultAllocator());
      fflush(stdout);
#else
      int status;
      
      pid_t pid;
      pthread_t me_before = pthread_self();
      // create the pipe file destriptors
      int pipe_fd[2];
      if (pipe(pipe_fd) == -1) {
	fprintf(stderr,"Error in piping\n");
	//	return -1;
	return;
      }
      
      LockAllocator(DefaultAllocator());
      //      SetDontLock(DefaultAllocator(), true);
      //      pid = vfork ();
      //      pid = fork ();
      pid = __fork ();
      //      SetDontLock(DefaultAllocator(), false);
      pthread_t me_after = pthread_self();
      if (pid == (pid_t) 0) {
	// child


	// Send signal to parent that we are past the fork
	if (close(pipe_fd[0]) == -1) {
	  fprintf(stderr, "Error in closing read pipe in child\n");
	  exit(1);
	}
	//	UnLockAllocator(DefaultAllocator());
	char to_parent[1] = {'n'};
        if (write(pipe_fd[1], to_parent, 1) != 1) {
	  fprintf(stderr, "Error in writing to pipe in child\n");
	  exit(1);
	}
	if (close(pipe_fd[1]) == -1) {
	  fprintf(stderr, "Error in closing write pipe in child\n");
	  exit(1);
	}
	
	const char *new_argv[4];
	new_argv[0] = SHELL_NAME;
	new_argv[1] = "-c";
	new_argv[2] = buf;
	//new_argv[2] = "printf \%c '*'";
	new_argv[3] = NULL;

	/* Restore the signals.  */
	//	(void) __sigaction (SIGINT, &intr, (struct sigaction *) NULL);
	//	(void) __sigaction (SIGQUIT, &quit, (struct sigaction *) NULL);
	//	(void) UNBLOCK;
	
	/* Exec the shell.  */
	/* - cm modified 
	   (void) __execve (SHELL_PATH, (char *const *) new_argv, __environ); */
	(void) execve (SHELL_PATH, (char *const *) new_argv, __environ);
	/* - cm */
	
	_exit (127);
      } else if (pid < (pid_t) 0) {
	// fork failed
	UnLockAllocator(DefaultAllocator());
      } else {
	// parent

	// Wait for signal from child that it is past the fork
	if (close(pipe_fd[1]) == -1) {
	  fprintf(stderr, "Error in closing write pipe in parent\n");
	  //	  return -1;
	  return;
	}
	char from_child[1];
        if (read(pipe_fd[0], from_child, 1) != 1) {
	  fprintf(stderr, "Error in reading from pipe in parent\n");
	  UnLockAllocator(DefaultAllocator());
	  //	  return -1;
	  return;
	}
	UnLockAllocator(DefaultAllocator());
	if (close(pipe_fd[0]) == -1) {
	  fprintf(stderr, "Error in closing read pipe in parent\n");
	  //	  return -1;
	  return;
	}
	
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
      }
      //      UnLockAllocator(DefaultAllocator());
      
#endif // end switch on system
      //      if (drand48() < 1e-4) {
      //	printf("-");
      //	fflush(stdout);
	//      }	
    }
  }
};

int
main(char **, int )
{
    fprintf(stderr, "This should hang - race condition on Allocator::lock within forked thread.\n\n\n");

    (new Thread(new AllocALot('a'), "AllocALot thread"))->detach();
    (new Thread(new AllocALot('A'), "AllocALot thread"))->detach();
    (new Thread(new AllocALot('i'), "AllocALot thread"))->detach();
    (new Thread(new ForkALot('-'), "ForkALot thread"))->detach();
    (new Thread(new ForkALot('+'), "ForkALot thread"))->detach();
    (new Thread(new ForkALot('^'), "ForkALot thread"))->detach();

    return 0;
}

#else // #ifdef __linux

#include <stdio.h>

int
main(char **, int )
{
    fprintf(stderr, "This program is designed for linux only.\n\n\n");
    return 0;
}

#endif // #ifdef __linux
