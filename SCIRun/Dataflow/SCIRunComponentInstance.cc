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
 *  SCIRunComponentInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Dataflow/SCIRunComponentInstance.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

SCIRunComponentInstance::SCIRunComponentInstance(SCIRunFramework* framework,
						 const std::string& name,
						 Module* module)
  : ComponentInstance(framework, name), module(module)
{
}

SCIRunComponentInstance::~SCIRunComponentInstance()
{
}

PortInstance* SCIRunComponentInstance::getPortInstance(const std::string& name)
{
  cerr << "SCIRunComponentInstance::getPortInstance not finished!\n";
  return 0;
}
