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
 *  Object: Implementation of SSIDL.Object for PIDL
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

//using SSIDL::Class;
using SSIDL::BaseInterface;
using SSIDL::Object;

/*
 * These are all implemented in SSIDL.interface, so these are just
 * up calls, since they will get generated from cia spec.
 */

void Object::addReference()
{
  BaseInterface::addReference();
}

void Object::deleteReference()
{
  BaseInterface::deleteReference();
}

/*Class::pointer Object::getClass()
{
  return  BaseInterface::getClass();
}*/

bool Object::isSame(const BaseInterface::pointer& i)
{
  return BaseInterface::isSame(i);
}
/*
bool Object::isInstanceOf(const Class::pointer& c)
{
  return BaseInterface::isInstanceOf(c);
}

bool Object::supportsInterface(const Class::pointer& c)
{
  return BaseInterface::supportsInterface(c);
}*/
/*
BaseInterface::pointer Object::queryInterface(const Class::pointer& c)
{
  return BaseInterface::queryInterface(c);
}
*/

BaseInterface::pointer Object::queryInterface(const std::string& c)
{
  return BaseInterface::queryInterface(c);
}

