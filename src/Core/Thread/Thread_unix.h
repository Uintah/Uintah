
/*
 *  Thread_unix.h:  Header file for utiity functions for unix versions
 *		    of the thread class
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Thread_unix_h
#define Core_Thread_Thread_unix_h

#include <sys/types.h>

char* Core_Thread_signal_name(int sig, void* code);

#endif
