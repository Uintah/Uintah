
/*
 *  Thread_unix.h:  Header file for utiity functions for unix versions
 *  $Id$
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

#ifndef SCICore_Thread_Thread_unix_h
#define SCICore_Thread_Thread_unix_h

#include <sys/types.h>

char* SCICore_Thread_signal_name(int sig, void* code);

#endif
