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

#include <Core/CCA/Comm/CommNexus.h>
#include <Core/CCA/Comm/CommError.h>

using namespace SCIRun;

void
CommNexus::initialize()
{
  if(int gerr=globus_module_activate(GLOBUS_NEXUS_MODULE))
    throw CommError("Unable to initialize nexus", gerr);
  globus_nexus_enable_fault_tolerance(NULL, 0);
}

void
CommNexus::finalize()
{
  if(int gerr=globus_module_deactivate(GLOBUS_NEXUS_MODULE))
    throw CommError("Unable to initialize nexus", gerr);
}

