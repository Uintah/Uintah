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
 *  InternalComponentDescription.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Internal/InternalComponentDescription.h>
#include <SCIRun/Internal/InternalComponentInstance.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <iostream>
using namespace SCIRun;
using namespace std;

InternalComponentDescription::InternalComponentDescription(InternalComponentModel* model,
							   const std::string& serviceType,
							   InternalComponentInstance* (*create)(SCIRunFramework*, const std::string&),
							   bool isSingleton)
  : model(model), serviceType(serviceType), create(create), isSingleton(isSingleton)
{
  singleton_instance=0;
}

InternalComponentDescription::~InternalComponentDescription()
{
  cerr << "What if singleton_instance is refcounted?\n";
  if(singleton_instance)
    delete singleton_instance;
}

string InternalComponentDescription::getType() const
{
  return serviceType;
}

const ComponentModel* InternalComponentDescription::getModel() const
{
  return model;
}
