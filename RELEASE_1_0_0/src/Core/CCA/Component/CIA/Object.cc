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
 *  Object: Implementation of CIA.Object for PIDL
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

using CIA::Class;
using CIA::Interface;
using CIA::Object_interface;
using CIA::Object;

/*
 * These are all implemented in CIA.interface, so these are just
 * up calls, since they will get generated from cia spec.
 */

Object Object_interface::addReference()
{
    return Interface_interface::addReference();
}

void Object_interface::deleteReference()
{
    Interface_interface::deleteReference();
}

Class Object_interface::getClass()
{
    return Interface_interface::getClass();
}

bool Object_interface::isSame(const Interface& i)
{
    return Interface_interface::isSame(i);
}

bool Object_interface::isInstanceOf(const Class& c)
{
    return Interface_interface::isInstanceOf(c);
}

bool Object_interface::supportsInterface(const Class& c)
{
    return Interface_interface::supportsInterface(c);
}

Interface Object_interface::queryInterface(const Class& c)
{
    return Interface_interface::queryInterface(c);
}

