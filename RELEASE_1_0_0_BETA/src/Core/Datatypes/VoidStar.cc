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
 *  VoidStar.cc: Just has a rep member -- other trivial classes can inherit
 *		 from this, rather than having a full-blown datatype and data-
 *		 port for every little thing that comes along...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Datatypes/VoidStar.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/String.h>

namespace SCIRun {

PersistentTypeID VoidStar::type_id("VoidStar", "Datatype", 0);

VoidStar::VoidStar()
{
}

VoidStar::VoidStar(const VoidStar& /*copy*/)
{
  NOT_FINISHED("VoidStar::VoidStar");
}

VoidStar::~VoidStar()
{
}

#define VoidStar_VERSION 2
void VoidStar::io(Piostream& stream) {
  int version=stream.begin_class("VoidStar", VoidStar_VERSION);
  if (version < 2) {
    if (stream.reading()) {
      int rep;
      Pio(stream, rep);
    }
  }
  stream.end_class();
}

} // End namespace SCIRun


