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
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <assert.h>
using namespace SCIRun;

Reference::Reference()
{
  chan = PIDL::getSpChannel();
  d_vtable_base=TypeInfo::vtable_invalid;
  primary=true;
}

Reference::Reference(SpChannel* n_chan)
{
  d_vtable_base=TypeInfo::vtable_invalid;
  chan = n_chan;
  primary=true;
}

Reference::Reference(const Reference& copy)
{
  assert(false);
  _copy(copy);
}

Reference *
Reference::clone()
{		    
  Reference * Clone=new Reference();
  Clone->d_vtable_base=d_vtable_base;
  Clone->primary=primary;
  //cannot clone non-primary Reference
  assert(primary); 
  
  //channel must not be null
  assert(chan!=NULL);
  delete Clone->chan;
  Clone->chan = chan->SPFactory(true);
  return Clone;
}

void
Reference::cloneTo(Reference &Clone)
{		    
  Clone.d_vtable_base=d_vtable_base;
  Clone.primary=primary;
  //cannot clone non-primary Reference
  assert(primary); 
  
  //channel must not be null
  assert(chan!=NULL);
  
  //this should be removed later
  delete Clone.chan;

  Clone.chan = chan->SPFactory(true);
}

Reference::~Reference()
{
#ifdef HAVE_GLOBUS
  if(primary)
#endif
    delete chan;

}

Reference& Reference::operator=(const Reference& copy)
{
  return _copy(copy);
}

Reference& Reference::_copy(const Reference& copy)
{
  d_vtable_base=copy.d_vtable_base;
  if(primary && chan!=NULL) delete chan; 

  //cannot copy Reference without a SP channel
  assert(copy.chan != NULL);
  chan = (copy.chan)->SPFactory(false);
  primary=false;
  return *this;
}

int Reference::getVtableBase() const
{
  return d_vtable_base;
}








