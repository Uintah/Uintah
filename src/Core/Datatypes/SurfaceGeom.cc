//  SurfaceGeom.cc - A group of Tets in 3 space
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/SurfaceGeom.h>

namespace SCIRun {

PersistentTypeID SurfaceGeom::type_id("SurfaceGeom", "Datatype", 0);

DebugStream SurfaceGeom::dbg("SurfaceGeom", true);

SurfaceGeom::SurfaceGeom()
{
}


string
SurfaceGeom::getInfo()
{
  ostringstream retval;
  retval << "name = " << d_name << endl;
  return retval.str();
}

void
SurfaceGeom::io(Piostream&)
{
}


} // End namespace SCIRun
