
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


