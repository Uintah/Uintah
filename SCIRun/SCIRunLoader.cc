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
 *  SCIRunLoader.cc: An instance of the SCIRun CCA Component Loader
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#include <SCIRun/SCIRunLoader.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <iostream>
#include <sstream>

#include "CCACommunicator.h"

using namespace std;
using namespace SCIRun;

SCIRunLoader::SCIRunLoader()
{

}

SCIRunLoader::~SCIRunLoader()
{
  cerr << "SCIRun  Loader exits.\n";
  //abort();
}

int
SCIRunLoader::loadComponent(const std::string & componentType)
{
  cout<< "loadComponent not implemented.\n";
  return 0;
}

int
SCIRunLoader::getComonents(SSIDL::array1<std::string>& componentList)
{
  cout<< "getComponents not implemented.\n";
  return 0;

}


