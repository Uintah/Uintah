/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

