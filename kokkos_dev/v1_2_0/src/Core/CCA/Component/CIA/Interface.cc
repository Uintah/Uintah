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
using CIA::Interface_interface;
using CIA::Object;

Object Interface_interface::addReference()
{
    NOT_FINISHED(".CIA.Object .CIA.Interface.addReference()");
    return 0;
}

void Interface_interface::deleteReference()
{
    NOT_FINISHED("void .CIA.Interface.deleteReference()");
}

Class Interface_interface::getClass()
{
    NOT_FINISHED(".CIA.Class .CIA.Interface.getClass()");
    return 0;
}

bool Interface_interface::isSame(const Interface& object)
{
    NOT_FINISHED("bool .CIA.Interface.isSame(in .CIA.Interface object)");
    return false;
}

bool Interface_interface::isInstanceOf(const Class& type)
{
    NOT_FINISHED("bool .CIA.Interface.isInstanceOf(in .CIA.Class type)");
    return false;
}

bool Interface_interface::supportsInterface(const Class& type)
{
    NOT_FINISHED("bool .CIA.Interface.supportsInterface(in .CIA.Class type)");
    return false;
}

Interface Interface_interface::queryInterface(const Class& type)
{
    NOT_FINISHED(".CIA.Interface .CIA.Interface.queryInterface(in .CIA.Class type)");
    return Interface();
}

