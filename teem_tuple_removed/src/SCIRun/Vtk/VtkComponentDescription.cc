/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  VtkComponentDescription.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <SCIRun/Vtk/VtkComponentDescription.h>
#include <SCIRun/Vtk/VtkComponentModel.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

VtkComponentDescription::VtkComponentDescription(VtkComponentModel* model, const string& type)
  : model(model), type(type)
{
}

VtkComponentDescription::~VtkComponentDescription()
{
}

string VtkComponentDescription::getType() const
{
  return type;
}

const ComponentModel* VtkComponentDescription::getModel() const
{
  return model;
}
