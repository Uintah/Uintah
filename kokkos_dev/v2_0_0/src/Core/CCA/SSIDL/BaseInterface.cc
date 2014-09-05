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
 *  BaseInterface: Implementation of SSIDL.BaseInterface for PIDL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/SSIDL/sidl_sidl.h>
#include <Core/Util/NotFinished.h>

//using SSIDL::Class;
using SSIDL::BaseInterface;
using SSIDL::Object;

void BaseInterface::addReference()
{
  SCIRun::Object::addReference();
}

void BaseInterface::deleteReference()
{
  SCIRun::Object::deleteReference();
}
/*
Class::pointer BaseInterface::getClass()
{
  NOT_FINISHED(".SSIDL.Class .SSIDL.BaseInterface.getClass()");
  return Class::pointer(0);
}
*/
bool BaseInterface::isSame(const BaseInterface::pointer& /*object*/)
{
  NOT_FINISHED("bool .SSIDL.BaseInterface.isSame(in .SSIDL.BaseInterface object)");
  return false;
}

/*
bool BaseInterface::isInstanceOf(const Class::pointer& type)
{
  NOT_FINISHED("bool .SSIDL.BaseInterface.isInstanceOf(in .SSIDL.Class type)");
  return false;
}

bool BaseInterface::supportsInterface(const Class::pointer& type)
{
  NOT_FINISHED("bool .SSIDL.BaseInterface.supportsInterface(in .SSIDL.Class type)");
  return false;
}
*/
BaseInterface::pointer BaseInterface::queryInterface(const std::string& /*type*/)
{
  NOT_FINISHED(".SSIDL.BaseInterface .SSIDL.BaseInterface.queryInterface(in .SSIDL.Class type)");
  return BaseInterface::pointer(0);
}

