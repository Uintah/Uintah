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


#ifdef __linux /* only linux seems to have a problem with this */
#include <Core/Util/sci_system.h>
#include <cstdlib>
#include <cstdio>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Malloc/Allocator.h>
#include <cstdio>
#include <cstddef>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/types.h>
#include <cerrno>

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
main(int,char **)
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

#include <cstdio>

int
main()
{
    fprintf(stderr, "This program is designed for linux only.\n\n\n");
    return 0;
}

#endif // #ifdef __linux
