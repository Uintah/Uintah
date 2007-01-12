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



#include "PingPong_manual.h"
#include <Core/Exceptions/InternalError.h>
#include <Core/CCA/PIDL/Dispatch.h>
#include <Core/CCA/PIDL/GlobusError.h>
#include <Core/CCA/PIDL/ReplyEP.h>
#include <Core/CCA/PIDL/Startpoint.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/Util/NotFinished.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <globus_nexus.h>

static PIDL::TypeInfo typeinfo;

using namespace PingPong;

static void fn1(globus_nexus_endpoint_t* ep, globus_nexus_buffer_t* buffer,
		globus_bool_t)
{
    int arg1;
    globus_nexus_get_int(buffer, &arg1, 1);
    globus_nexus_startpoint_t sp;
    if(int gerr=globus_nexus_get_startpoint(buffer, &sp, 1))
	throw PIDL::GlobusError("get_startpoint", gerr);
    if(int gerr=globus_nexus_buffer_destroy(buffer))
	throw PIDL::GlobusError("buffer_destroy", gerr);
    void* v=globus_nexus_endpoint_get_user_pointer(ep);
    PingPong::PingPong_interface* obj=(PingPong::PingPong_interface*)v;
    if(!obj)
	throw InternalError("wrong object?", __FILE__, __LINE__);
    int ret=obj->pingpong(arg1);

    int size=globus_nexus_sizeof_int(2); // one for flag, one for data
    globus_nexus_buffer_t rbuff;
    if(int gerr=globus_nexus_buffer_init(&rbuff, size, 0))
	throw PIDL::GlobusError("buffer_init", gerr);
    int flag=0;
    globus_nexus_put_int(&rbuff, &flag, 1);
    globus_nexus_put_int(&rbuff, &ret, 1);
    if(int gerr=globus_nexus_send_rsr(&rbuff, &sp, 0,
				      GLOBUS_TRUE, GLOBUS_FALSE))
	throw PIDL::GlobusError("send_rsr", gerr);
#if 1
    static int hack=0;
    if(hack++ > 1)
#endif
	if(int gerr=globus_nexus_startpoint_destroy(&sp))
	    throw PIDL::GlobusError("startpoint_destroy", gerr);
}

static globus_nexus_handler_t handler_table[] =
{
    {GLOBUS_NEXUS_HANDLER_TYPE_THREADED, 0}, // isa???
    {GLOBUS_NEXUS_HANDLER_TYPE_THREADED, fn1}
};

static PIDL::Dispatch dispatch(handler_table, sizeof(handler_table)/sizeof(globus_nexus_handler_t));

const PIDL::TypeSignature& PingPong::PingPong::type_signature()
{
    static std::string mytypename="PingPong::PingPong";
    return mytypename;
}

PingPong::PingPong_interface::PingPong_interface()
    : PIDL::Object_interface(&typeinfo, &dispatch, this)
{
}

PingPong::PingPong_interface::~PingPong_interface()
{
}

PingPong::PingPong_proxy::PingPong_proxy(const PIDL::Reference& ref)
    : PIDL::ProxyBase(ref)
{
}

PingPong::PingPong_proxy::~PingPong_proxy()
{
}

int PingPong::PingPong_proxy::pingpong(int i)
{
    PIDL::ReplyEP* reply=PIDL::ReplyEP::acquire();
    globus_nexus_startpoint_t sp;
    reply->get_startpoint(&sp);
    int size=globus_nexus_sizeof_int(1)
	+globus_nexus_sizeof_startpoint(&sp, 1);
    globus_nexus_buffer_t buffer;
    if(int gerr=globus_nexus_buffer_init(&buffer, size, 0))
	throw PIDL::GlobusError("buffer_init", gerr);
    globus_nexus_put_int(&buffer, &i, 1);
    globus_nexus_put_startpoint_transfer(&buffer, &sp, 1);

    if(int gerr=globus_nexus_send_rsr(&buffer, &getStartpoint()->d_sp,
				      1, GLOBUS_TRUE, GLOBUS_FALSE))
	throw PIDL::GlobusError("send_rsr", gerr);

    globus_nexus_buffer_t rbuffer=reply->wait();
    int flag;
    globus_nexus_get_int(&rbuffer, &flag, 1);
    if(flag != 0)
	NOT_FINISHED("Exceptions not implemented");
    int result;
    globus_nexus_get_int(&rbuffer, &result, 1);
    if(int gerr=globus_nexus_buffer_destroy(&rbuffer))
	throw PIDL::GlobusError("buffer_destroy", gerr);
    PIDL::ReplyEP::release(reply);

    return result;
}

