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
 *  Method: Implementation of CIA.Method for PIDL
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
using CIA::Method_interface;

Class Method_interface::getDeclaringClass()
{
    NOT_FINISHED("Method_interface::getDeclaringClass");
    return 0;
}

::CIA::string Method_interface::getName()
{
    NOT_FINISHED("final string .CIA.Metho.getName()");
    return "";
}

Class Method_interface::getReturnType()
{
    NOT_FINISHED("Method_interface::getReturnType");
    return 0;
}

