/*
 *  Proc.cc: A process
 *
 *  Written by:
 *   Jason V. Morgan
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *  Updated by:
 *   Jason V. Morgan
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Morgan/Core/Process/Proc.h>
#include <Packages/Morgan/Core/Process/ProcManagerException.h>
#include <stdio.h>
#include <string>

#if defined(_WIN32) 
#error Windows Process stuff not implemented yet
#else
extern "C" { 
   #include <unistd.h>
   #include <sys/types.h>
   #include <signal.h>
}
#endif // !defined(_WIN32)

using Morgan::Process::Proc; 

Proc::Proc(int pid, int input_fd, int output_fd) :
	pid(pid), ifsb(input_fd), ofsb(output_fd), 
    ifs_(&ifsb), ofs_(&ofsb) {
}

Proc::~Proc() {
    kill(pid, SIGTERM);
}
