
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

#ifndef PingPong_PingPong_impl_h
#define PingPong_PingPong_impl_h

#include "PingPong_manual.h"

namespace PingPong {
    class PingPong_impl : public PingPong_interface {
    public:
	PingPong_impl();
	virtual ~PingPong_impl();
	virtual int pingpong(int);
    };
}

#endif

