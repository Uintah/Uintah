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
 *  Interface: Implementation of CIA.Interface for PIDL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/CIA/CIA_sidl.h>
#include <Core/Util/NotFinished.h>

using CIA::Class;
using CIA::Interface;
using CIA::Object;

void Interface::addReference()
{
  Object::addReference();
}

void Interface::deleteReference()
{
  Object::deleteReference();
}

Class::pointer Interface::getClass()
{
  NOT_FINISHED(".CIA.Class .CIA.Interface.getClass()");
  return Class::pointer(0);
}

bool Interface::isSame(const Interface::pointer& object)
{
  NOT_FINISHED("bool .CIA.Interface.isSame(in .CIA.Interface object)");
  return false;
}

bool Interface::isInstanceOf(const Class::pointer& type)
{
  NOT_FINISHED("bool .CIA.Interface.isInstanceOf(in .CIA.Class type)");
  return false;
}

bool Interface::supportsInterface(const Class::pointer& type)
{
  NOT_FINISHED("bool .CIA.Interface.supportsInterface(in .CIA.Class type)");
  return false;
}

Interface::pointer Interface::queryInterface(const Class::pointer& type)
{
  NOT_FINISHED(".CIA.Interface .CIA.Interface.queryInterface(in .CIA.Class type)");
  return Interface::pointer(0);
}

