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
#include <Core/CCA/Component/PIDL/Warehouse.h>
#include <Core/Exceptions/InternalError.h>
#include <globus_nexus.h>
#include <iostream>
#include <sstream>


namespace PIDL {

static unsigned short port;
static char* host;

Warehouse* PIDL::PIDL::warehouse;

static int approval_fn(void*, char* urlstring, globus_nexus_startpoint_t* sp)
{
  try {
    Warehouse* wh=PIDL::getWarehouse();
    return wh->approval(urlstring, sp);
  } catch(const SCIRun::Exception& e) {
    std::cerr << "Caught exception (" << e.message() << "): " << urlstring
	      << ", rejecting client (code=1005)\n";
    return 1005;
  } catch(...) {
    std::cerr << "Caught unknown exception: " << urlstring
	      << ", rejecting client (code=1006)\n";
    return 1006;
  }
}

void
PIDL::initialize(int, char*[])
{
  if(!warehouse){
    warehouse=new Warehouse;
    if(int gerr=globus_module_activate(GLOBUS_NEXUS_MODULE))
      throw GlobusError("Unable to initialize nexus", gerr);
    if(int gerr=globus_nexus_allow_attach(&port, &host, approval_fn, 0))
      throw GlobusError("globus_nexus_allow_attach failed", gerr);
    globus_nexus_enable_fault_tolerance(NULL, 0);
  }
}

Warehouse*
PIDL::getWarehouse()
{
  if(!warehouse)
    throw SCIRun::InternalError("Warehouse not initialized!\n");
  return warehouse;
}


Object::pointer
PIDL::objectFrom(const URL& url)
{
  return Object::pointer(new Object_proxy(url));
}

void PIDL::serveObjects()
{
  if(!warehouse)
    throw SCIRun::InternalError("Warehouse not initialized!\n");
  warehouse->run();
}

std::string PIDL::getBaseURL()
{
  if(!warehouse)
    throw SCIRun::InternalError("Warehouse not initialized!\n");
  std::ostringstream o;
  o << "x-nexus://" << host << ":" << port << "/";
  return o.str();
}

}
