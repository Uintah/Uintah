
/*
 *  Runnable: The base class for all threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadError.h>
namespace SCIRun {


Runnable::Runnable()
{
    my_thread_=0;
}

Runnable::~Runnable()
{
    if(my_thread_){
	throw ThreadError("Runnable is being destroyed while thread is still running\n");
    }
}


} // End namespace SCIRun
