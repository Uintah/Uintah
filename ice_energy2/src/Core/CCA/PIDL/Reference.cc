/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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








