
#include "PingPong_manual.h"
#include <SCICore/Exceptions/InternalError.h>
#include <Component/PIDL/Dispatch.h>
#include <Component/PIDL/GlobusError.h>
#include <Component/PIDL/ReplyEP.h>
#include <Component/PIDL/Startpoint.h>
#include <Component/PIDL/TypeInfo.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Thread/Thread.h>
#include <iostream>
#include <globus_nexus.h>

static Component::PIDL::TypeInfo typeinfo;

static void fn1(globus_nexus_endpoint_t* ep, globus_nexus_buffer_t* buffer,
		globus_bool_t)
{
    int arg1;
    globus_nexus_get_int(buffer, &arg1, 1);
    globus_nexus_startpoint_t sp;
    if(int gerr=globus_nexus_get_startpoint(buffer, &sp, 1))
	throw Component::PIDL::GlobusError("get_startpoint", gerr);
    if(int gerr=globus_nexus_buffer_destroy(buffer))
	throw Component::PIDL::GlobusError("buffer_destroy", gerr);
    void* v=globus_nexus_endpoint_get_user_pointer(ep);
    PingPong::PingPong_interface* obj=(PingPong::PingPong_interface*)v;
    if(!obj)
	throw SCICore::Exceptions::InternalError("wrong object?");
    int ret=obj->pingpong(arg1);

    int size=globus_nexus_sizeof_int(2); // one for flag, one for data
    globus_nexus_buffer_t rbuff;
    if(int gerr=globus_nexus_buffer_init(&rbuff, size, 0))
	throw Component::PIDL::GlobusError("buffer_init", gerr);
    int flag=0;
    globus_nexus_put_int(&rbuff, &flag, 1);
    globus_nexus_put_int(&rbuff, &ret, 1);
    if(int gerr=globus_nexus_send_rsr(&rbuff, &sp, 0,
				      GLOBUS_TRUE, GLOBUS_FALSE))
	throw Component::PIDL::GlobusError("send_rsr", gerr);
#if 1
    static int hack=0;
    if(hack++ > 1)
#endif
	if(int gerr=globus_nexus_startpoint_destroy(&sp))
	    throw Component::PIDL::GlobusError("startpoint_destroy", gerr);
}

static globus_nexus_handler_t handler_table[] =
{
    {GLOBUS_NEXUS_HANDLER_TYPE_THREADED, 0}, // isa???
    {GLOBUS_NEXUS_HANDLER_TYPE_THREADED, fn1}
};

static Component::PIDL::Dispatch dispatch(handler_table, sizeof(handler_table)/sizeof(globus_nexus_handler_t));

const Component::PIDL::TypeSignature& PingPong::PingPong::type_signature()
{
    static std::string mytypename="PingPong::PingPong";
    return mytypename;
}

PingPong::PingPong_interface::PingPong_interface()
    : Component::PIDL::Object_interface(&typeinfo, &dispatch, this)
{
}

PingPong::PingPong_interface::~PingPong_interface()
{
}

PingPong::PingPong_proxy::PingPong_proxy(const Component::PIDL::Reference& ref)
    : Component::PIDL::ProxyBase(ref)
{
}

PingPong::PingPong_proxy::~PingPong_proxy()
{
}

int PingPong::PingPong_proxy::pingpong(int i)
{
    Component::PIDL::ReplyEP* reply=Component::PIDL::ReplyEP::acquire();
    globus_nexus_startpoint_t sp;
    reply->get_startpoint(&sp);
    int size=globus_nexus_sizeof_int(1)
	+globus_nexus_sizeof_startpoint(&sp, 1);
    globus_nexus_buffer_t buffer;
    if(int gerr=globus_nexus_buffer_init(&buffer, size, 0))
	throw Component::PIDL::GlobusError("buffer_init", gerr);
    globus_nexus_put_int(&buffer, &i, 1);
    globus_nexus_put_startpoint_transfer(&buffer, &sp, 1);

    if(int gerr=globus_nexus_send_rsr(&buffer, &getStartpoint()->d_sp,
				      1, GLOBUS_TRUE, GLOBUS_FALSE))
	throw Component::PIDL::GlobusError("send_rsr", gerr);

    globus_nexus_buffer_t rbuffer=reply->wait();
    int flag;
    globus_nexus_get_int(&rbuffer, &flag, 1);
    if(flag != 0)
	NOT_FINISHED("Exceptions not implemented");
    int result;
    globus_nexus_get_int(&rbuffer, &result, 1);
    if(int gerr=globus_nexus_buffer_destroy(&rbuffer))
	throw Component::PIDL::GlobusError("buffer_destroy", gerr);
    Component::PIDL::ReplyEP::release(reply);

    return result;
}

