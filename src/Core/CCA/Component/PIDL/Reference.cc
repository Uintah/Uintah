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

#include "Reference.h"
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <Core/CCA/Component/PIDL/PIDL.h>
using namespace SCIRun;

Reference::Reference()
{
  chan = PIDL::getSpChannel();
  d_vtable_base=TypeInfo::vtable_invalid;    
}

Reference::Reference(const Reference& copy)
  :d_vtable_base(copy.d_vtable_base) 
{
  chan = (copy.chan)->SPFactory(false);
}

Reference::~Reference(){
  if (chan != NULL) {
    delete chan;
    chan = NULL;
  }
}

Reference& Reference::operator=(const Reference& copy)
{
  d_vtable_base=copy.d_vtable_base;
  if(chan!=NULL) delete chan;  //k.z
  chan = (copy.chan)->SPFactory(false);
  return *this;
}

int Reference::getVtableBase() const
{
  return d_vtable_base;
}








