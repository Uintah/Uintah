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
 *  InternalComponentInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Internal/InternalComponentInstance.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

InternalComponentInstance::InternalComponentInstance(SCIRunFramework* framework,
						     const std::string& instanceName,
						     const std::string& className)
  : ComponentInstance(framework, instanceName, className), useCount(0)
{
}

InternalComponentInstance::~InternalComponentInstance()
{
}

PortInstance*
InternalComponentInstance::getPortInstance(const std::string& /*name*/)
{
  cerr << "InternalComponentInstance::getPortInstance not finished\n";
  return 0;
}

PortInstanceIterator* InternalComponentInstance::getPorts()
{
  cerr << "SCIRunComponentInstance::getPorts not finished!\n";
  return 0;
}

void InternalComponentInstance::incrementUseCount()
{
  useCount++;
}

bool InternalComponentInstance::decrementUseCount()
{
  if(useCount<=0)
    return false;
  useCount--;
  return true;
}
