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
 *  BabelComponentDescription.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Babel/BabelComponentDescription.h>
#include <SCIRun/Babel/BabelComponentModel.h>
using namespace SCIRun;
using namespace std;

BabelComponentDescription::BabelComponentDescription(BabelComponentModel* model)
  : model(model)
{
}

BabelComponentDescription::~BabelComponentDescription()
{
}

string BabelComponentDescription::getType() const
{
  return type;
}

const ComponentModel* BabelComponentDescription::getModel() const
{
  return model;
}
