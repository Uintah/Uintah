/*
 *  ProcManager.h: Manage forking/rshing of processes.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Updated by:
 *   Jason V. Morgan
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <stdarg.h>

#if defined(_WIN32)
#error ProcManager not implemented for Unix systems
#else // we now assume we're on a Unix system

extern "C" {
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
}

#endif // !defined(_WIN32)

#include <Packages/Morgan/Core/Process/Proc.h>
#include <Packages/Morgan/Core/Process/ProcManager.h>
#include <Packages/Morgan/Core/Process/ProcManagerException.h>
#include <Packages/Morgan/Core/Process/Proc.h>
#include <Core/Util/NotFinished.h>

using std::string;
using Morgan::Process::ProcManager;
using Morgan::Process::ProcManagerException;
using Morgan::Process::Proc;

/*
   ProcManager::
   Proc* start_proc
   in:  const char* command,
        ...

   Pre:  command is a string with a command and its arguments.  This
         command should be valid, if it is not then an exception will
         be thrown.
   Post: If this function succeeds, then it will return a new
         Proc object which will allow communication with the command.  It
         will return NULL if the given file cannot be executed.
*/
Proc* ProcManager::start_proc(const char* command, ...)
{
    va_list argp;
    Proc* proc = 0;

    va_start(argp, command);

    struct stat stat_buf;

    if(stat(command, &stat_buf)) {
        return NULL;
    }

#if defined(_WIN32) 
#error ProcManager not implemented for Windows
#else // assumed we're compiling for Unix

    int p2cw = 0, // parent to child writer (parent's output)
        p2cr = 0, // parent to child reader (child's input)
        c2pw = 0, // child to parent writer (child's output)
        c2pr = 0; // child to parent reader (parent's input)
    int fds[2];

    try {
        // open pipe from parent -> child
        if(pipe(fds)) {
            throw(ProcManagerException("Can't open parent -> child"));
        }
        p2cr = fds[0];
        p2cw = fds[1];

        // open pipe from child -> parent
        if(pipe(fds)) {
            throw(ProcManagerException("Can't open child -> parent"));
        }
        c2pr = fds[0];
        c2pw = fds[1];

        int child_pid;
        child_pid = fork();

        // check for error with fork
        if(child_pid == -1) {
            throw(ProcManagerException("Can't fork"));
        }

        // we are the child process, start up the program after
        // adjusting the file handles
        if(child_pid == 0) {
            dup2(p2cr, 0);
            dup2(c2pw, 1);
            close(p2cr);
            close(c2pw);
            execlp(command, NULL); // WARNING...possibly non-portable
            perror("Cannot start program");
            exit(0);
        } else {
            // we are the parent process, store the info about
            // the child process and return to the caller
            proc = new Proc(child_pid, c2pr, p2cw);
        }
    } catch(ProcManagerException& e) {
        // make sure we close all our file handles when we fail
        if(p2cr) { close(p2cr); }
        if(p2cw) { close(p2cw); }
        if(c2pr) { close(c2pr); }
        if(c2pw) { close(c2pw); }
        throw(e);
    }
#endif // !defined(_WIN32)

    va_end(argp);

    return proc;
}

