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
 *  ComponentModel.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/ComponentModel.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

ComponentModel::ComponentModel(const std::string& prefixName)
  : prefixName(prefixName)
{
}

ComponentModel::~ComponentModel()
{
}

bool ComponentModel::haveComponent(const std::string& type)
{
  cerr << "Error: this component model does not implement haveComponent, name=" << type << "\n";
  return false;
}

ComponentInstance* ComponentModel::createInstance(const std::string& name,
						  const std::string& type)
{
  cerr << "Error: this component model does not implement createInstance\n";
  return 0;
}

bool  ComponentModel::destroyInstance(ComponentInstance* ic)
{
  cerr << "Error: this component model does not implement destroyInstance\n";
  return false;
}
