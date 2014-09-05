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
 *  BridgeComponentDescription.cc:
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#include <SCIRun/Bridge/BridgeComponentDescription.h>
#include <SCIRun/Bridge/BridgeComponentModel.h>
using namespace SCIRun;
using namespace std;

BridgeComponentDescription::BridgeComponentDescription(BridgeComponentModel* model)
  : model(model)
{
}

BridgeComponentDescription::~BridgeComponentDescription()
{
}

string BridgeComponentDescription::getType() const
{
  return type;
}

const ComponentModel* BridgeComponentDescription::getModel() const
{
  return model;
}

std::string BridgeComponentDescription::getLoaderName() const
{
  return loaderName;
}

void BridgeComponentDescription::setLoaderName(const std::string& loaderName)
{
  this->loaderName=loaderName;
}
