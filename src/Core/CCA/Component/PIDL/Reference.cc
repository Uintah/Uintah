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
 *  Reference.h: A serializable "pointer" to an object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/Reference.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <globus_nexus.h>

using PIDL::Reference;

Reference::Reference()
{
    d_vtable_base=TypeInfo::vtable_invalid;
    globus_nexus_startpoint_set_null(&d_sp);
}

Reference::Reference(const Reference& copy)
    : d_sp(copy.d_sp), d_vtable_base(copy.d_vtable_base)
{
}

Reference::~Reference()
{
}

Reference& Reference::operator=(const Reference& copy)
{
    d_sp=copy.d_sp;
    d_vtable_base=copy.d_vtable_base;
    return *this;
}

int Reference::getVtableBase() const
{
    return d_vtable_base;
}

