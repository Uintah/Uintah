
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

#include "PingPong_sidl.h"

namespace PingPong {
    class PingPong_impl : public PingPong_interface {
    public:
	PingPong_impl();
	virtual ~PingPong_impl();
	virtual int pingpong(int);

	// CCA spec
	virtual CIA::Object addReference();
	virtual void deleteReference();
	virtual CIA::Class getClass();
	virtual bool isSame(const CIA::Interface& object);
	virtual bool isInstanceOf(const CIA::Class& type);
	virtual bool supportsInterface(const CIA::Class& type);
	virtual CIA::Interface queryInterface(const CIA::Class& type);
    };
}

#endif

