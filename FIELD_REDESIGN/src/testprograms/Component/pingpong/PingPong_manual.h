// autogen

#ifndef pp_interface
#define pp_interface

#include <Component/PIDL/Object.h>
#include <Component/PIDL/Object_proxy.h>
#include <Component/PIDL/ProxyBase.h>
#include <Component/PIDL/pidl_cast.h>

namespace PingPong {
    class PingPong_interface : public Component::PIDL::Object_interface {
    public:
	PingPong_interface(const PingPong_interface&);
	virtual ~PingPong_interface();
	virtual int pingpong(int i)=0;
    protected:
	PingPong_interface();
	PingPong_interface(const Component::PIDL::TypeInfo);
    };

    class PingPong_proxy : public Component::PIDL::ProxyBase, public PingPong_interface {
    public:
	virtual int pingpong(int i);
	PingPong_proxy(const Component::PIDL::Reference&);
    protected:
	virtual ~PingPong_proxy();
    private:
	PingPong_proxy(const PingPong_proxy&);
    };

    class PingPong {
	PingPong_interface* ptr;
    public:
	typedef PingPong_proxy proxytype;
	static const Component::PIDL::TypeSignature& type_signature();

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

        inline operator Component::PIDL::Object()
	{
	    return ptr;
	}

	inline PingPong_interface* operator->() const
	{
	    return ptr;
	}
    };
}

#endif

