
/*
 *  PIDL.h: Include a bunch of PIDL files for external clients
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/Component/PIDL/GlobusError.h>
#include <Core/CCA/Component/PIDL/Object_proxy.h>
#include <Core/CCA/Component/PIDL/Wharehouse.h>
#include <Core/Exceptions/InternalError.h>
#include <globus_nexus.h>
#include <iostream>
#include <sstream>

using Component::PIDL::GlobusError;
using Component::PIDL::Object;
using Component::PIDL::PIDL;
using Component::PIDL::Reference;
using Component::PIDL::Wharehouse;

static unsigned short port;
static char* host;

Wharehouse* PIDL::wharehouse;

static int approval_fn(void*, char* urlstring, globus_nexus_startpoint_t* sp)
{
    using namespace Component::PIDL;
    try {
	Wharehouse* wh=PIDL::getWharehouse();
	return wh->approval(urlstring, sp);
    } catch(const Exception& e) {
	std::cerr << "Caught exception (" << e.message() << "): " << urlstring
		  << ", rejecting client (code=1005)\n";
	return 1005;
    } catch(...) {
	std::cerr << "Caught unknown exception: " << urlstring
		  << ", rejecting client (code=1006)\n";
	return 1006;
    }
}

void PIDL::initialize(int, char*[])
{
    if(!wharehouse){
	wharehouse=new Wharehouse;
	if(int gerr=globus_module_activate(GLOBUS_NEXUS_MODULE))
	    throw GlobusError("Unable to initialize nexus", gerr);
 	if(int gerr=globus_nexus_allow_attach(&port, &host, approval_fn, 0))
	    throw GlobusError("globus_nexus_allow_attach failed", gerr);
	globus_nexus_enable_fault_tolerance(NULL, 0);
    }
}

Wharehouse* PIDL::getWharehouse()
{
    if(!wharehouse)
	throw InternalError("Wharehouse not initialized!\n");
    return wharehouse;
}


Object
PIDL::objectFrom(const URL& url)
{
    return new Object_proxy(url);
}

void PIDL::serveObjects()
{
    if(!wharehouse)
	throw InternalError("Wharehouse not initialized!\n");
    wharehouse->run();
}

std::string PIDL::getBaseURL()
{
    if(!wharehouse)
	throw InternalError("Wharehouse not initialized!\n");
    std::ostringstream o;
    o << "x-nexus://" << host << ":" << port << "/";
    return o.str();
}

