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
 *  Interface: Implementation of SIDL.Interface for PIDL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/SIDL/sidl_sidl.h>
#include <Core/Util/NotFinished.h>

//using SIDL::Class;
using SIDL::Interface;
using SIDL::Object;

void Interface::addReference()
{
  SCIRun::Object::addReference();
}

void Interface::deleteReference()
{
  SCIRun::Object::deleteReference();
}
/*
Class::pointer Interface::getClass()
{
  NOT_FINISHED(".SIDL.Class .SIDL.Interface.getClass()");
  return Class::pointer(0);
}
*/
bool Interface::isSame(const Interface::pointer& /*object*/)
{
  NOT_FINISHED("bool .SIDL.Interface.isSame(in .SIDL.Interface object)");
  return false;
}

/*
bool Interface::isInstanceOf(const Class::pointer& type)
{
  NOT_FINISHED("bool .SIDL.Interface.isInstanceOf(in .SIDL.Class type)");
  return false;
}

bool Interface::supportsInterface(const Class::pointer& type)
{
  NOT_FINISHED("bool .SIDL.Interface.supportsInterface(in .SIDL.Class type)");
  return false;
}
*/
Interface::pointer Interface::queryInterface(const std::string& /*type*/)
{
  NOT_FINISHED(".SIDL.Interface .SIDL.Interface.queryInterface(in .SIDL.Class type)");
  return Interface::pointer(0);
}

