
/*
 *  PingPong_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include "PingPong_impl.h"
#include <SCICore/Util/NotFinished.h>
using PingPong::PingPong_impl;
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

PingPong_impl::PingPong_impl()
{
}

PingPong_impl::~PingPong_impl()
{
}

int PingPong_impl::pingpong(int arg)
{
    //cerr << "Received: " << arg << "\n";
    return arg;
}
