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
 *  CCAComponentDescription.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/CCA/CCAComponentDescription.h>
#include <SCIRun/CCA/CCAComponentModel.h>
using namespace SCIRun;
using namespace std;

CCAComponentDescription::CCAComponentDescription(CCAComponentModel* model)
  : model(model)
{
}

CCAComponentDescription::~CCAComponentDescription()
{
}

string CCAComponentDescription::getType() const
{
  return type;
}

const ComponentModel* CCAComponentDescription::getModel() const
{
  return model;
}

std::string CCAComponentDescription::getLoaderName() const
{
  return loaderName;
}

void CCAComponentDescription::setLoaderName(const std::string& loaderName)
{
  this->loaderName=loaderName;
}
