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
 *  SCIRunUIPort.cc: CCA-style Interface to old TCL interfaces
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <SCIRun/Dataflow/SCIRunUIPort.h>
#include <SCIRun/Dataflow/SCIRunComponentInstance.h>
#include <Dataflow/Network/Module.h>
#include <iostream>
using namespace SCIRun;

SCIRunUIPort::SCIRunUIPort(SCIRunComponentInstance* component)
  : component(component)
{
}

SCIRunUIPort::~SCIRunUIPort()
{
}

int SCIRunUIPort::ui()
{
  Module* module = component->getModule();
  module->popupUI();
  cerr<<"Warning: need return correct value (0 success, -1 fatal error, other values for other errors !\n";
  return 0;
}


