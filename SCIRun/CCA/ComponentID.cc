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
 *  ComponentID.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/SCIRunFramework.h>
#include <Core/CCA/PIDL/URL.h>
#include <iostream>
using namespace SCIRun;
using namespace std;

ComponentID::ComponentID(SCIRunFramework* framework, const std::string& name)
  : framework(framework), name(name)
{
}

ComponentID::~ComponentID()
{
}

string ComponentID::getInstanceName()
{
  return name;
}

string ComponentID::getSerialization()
{
  string s = framework->getURL().getString()+"/"+name;
  return s;
}
