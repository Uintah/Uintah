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
 *  Datatype.cc: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Datatypes/Datatype.h>
#include <Core/Thread/AtomicCounter.h>

namespace SCIRun {

static AtomicCounter* current_generation = 0;
static Mutex init_lock("Datatypes generation counter initialization lock");

Datatype::Datatype()
: lock("Datatype ref_cnt lock")
{
    ref_cnt=0;
    if(!current_generation){
      init_lock.lock();
      if(!current_generation)
	current_generation = new AtomicCounter("Datatypes generation counter", 1);
      init_lock.unlock();
    }
    generation=(*current_generation)++;
}

Datatype::Datatype(const Datatype&)
    : lock("Datatype ref_cnt lock")
{
    ref_cnt=0;
    generation=(*current_generation)++;
}

Datatype& Datatype::operator=(const Datatype&)
{
    // XXX:
    // Should probably throw an exception if ref_cnt is > 0 or
    // something.
    generation=(*current_generation)++;
    return *this;
}

Datatype::~Datatype()
{
}

} // End namespace SCIRun

