
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

CIA::Object PingPong_impl::addReference()
{
    NOT_FINISHED("addReference");
    return 0;
}

void PingPong_impl::deleteReference()
{
    NOT_FINISHED("deleteReference");
}

CIA::Class PingPong_impl::getClass()
{
    NOT_FINISHED("getClass");
    return 0;
}

bool PingPong_impl::isSame(const CIA::Interface&)
{
    NOT_FINISHED("isSame");
    return false;
}

bool PingPong_impl::isInstanceOf(const CIA::Class&)
{
    NOT_FINISHED("isInstanceOf");
    return false;
}

bool PingPong_impl::supportsInterface(const CIA::Class&)
{
    NOT_FINISHED("supportsInterface");
    return false;
}

CIA::Interface PingPong_impl::queryInterface(const CIA::Class&)
{
    NOT_FINISHED("queryInterface");
    return 0;
}
