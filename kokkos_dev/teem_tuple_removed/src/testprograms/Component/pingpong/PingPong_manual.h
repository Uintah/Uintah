/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

// autogen

#ifndef pp_interface
#define pp_interface

#include <Core/CCA/PIDL/Object.h>
#include <Core/CCA/PIDL/Object_proxy.h>
#include <Core/CCA/PIDL/ProxyBase.h>
#include <Core/CCA/PIDL/pidl_cast.h>

namespace PingPong {

    class PingPong_interface : public PIDL::Object_interface {
    public:
	PingPong_interface(const PingPong_interface&);
	virtual ~PingPong_interface();
	virtual int pingpong(int i)=0;
    protected:
	PingPong_interface();
	PingPong_interface(const PIDL::TypeInfo);
    };

    class PingPong_proxy : public PIDL::ProxyBase, public PingPong_interface {
    public:
	virtual int pingpong(int i);
	PingPong_proxy(const PIDL::Reference&);
    protected:
	virtual ~PingPong_proxy();
    private:
	PingPong_proxy(const PingPong_proxy&);
    };

    class PingPong {
	PingPong_interface* ptr;
    public:
	typedef PingPong_proxy proxytype;
	static const PIDL::TypeSignature& type_signature();

	inline PingPong()
	{
	    ptr=0;
	}

	inline PingPong(PingPong_interface* ptr)
	    : ptr(ptr)
	{
	}

	inline ~PingPong()
	{
	}

	inline PingPong(const PingPong& copy)
	    : ptr(copy.ptr)
	{
	}

	inline PingPong& operator=(const PingPong& copy)
	{
	    ptr=copy.ptr;
	    return *this;
	}

        inline operator PIDL::Object()
	{
	    return ptr;
	}

	inline PingPong_interface* operator->() const
	{
	    return ptr;
	}
    };
} // End namespace PingPong

#endif

