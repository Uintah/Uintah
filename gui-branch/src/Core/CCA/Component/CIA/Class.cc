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
 *  Class: Implementation of CIA.Class for PIDL
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
using CIA::Class_interface;
using CIA::Method;
using CIA::Object;

Object Class_interface::newInstance()
{
    NOT_FINISHED("final .CIA.Object .CIA.Class.newInstance()throws .CIA.InstantiationException");
    return 0;
}

bool Class_interface::isInterface()
{
    NOT_FINISHED("final bool .CIA.Class.isInterface()");
    return false;
}

bool Class_interface::isArray()
{
    NOT_FINISHED("final bool .CIA.Class.isArray()");
    return false;
}

bool Class_interface::isPrimitive()
{
    NOT_FINISHED("final bool .CIA.Class.isPrimitive()");
    return false;
}

::CIA::string Class_interface::getName()
{
    NOT_FINISHED("final string .CIA.Class.getName()");
    return "";
}

Class Class_interface::getSuperclass()
{
    NOT_FINISHED("final .CIA.Class .CIA.Class.getSuperclass()");
    return 0;
}

Class Class_interface::getComponentType()
{
    NOT_FINISHED("final .CIA.Class .CIA.Class.getComponentType()");
    return 0;
}

Method Class_interface::getMethod()
{
    NOT_FINISHED("final .CIA.Method .CIA.Class.getMethod()throws .CIA.NoSuchMethodException");
    return 0;
}


