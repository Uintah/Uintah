//  SurfaceGeom.cc - A group of Tets in 3 space
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/SurfaceGeom.h>

namespace SCIRun {

//////////
// PIO support
static Persistent* maker(){
  return new SurfaceGeom();
}

string  SurfaceGeom::typeName(int){
  static string className = "SurfaceGeom";
  return className;
}

PersistentTypeID SurfaceGeom::type_id(SurfaceGeom::typeName(0), 
				      ContourGeom::typeName(0), 
				      maker);
#define SURFACEGEOM_VERSION 1
void
SurfaceGeom::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(), SURFACEGEOM_VERSION);
  ContourGeom::io(stream);
  Pio(stream, face_);
  stream.end_class();
}

DebugStream SurfaceGeom::dbg("SurfaceGeom", true);


SurfaceGeom::SurfaceGeom()
{
}


string
SurfaceGeom::getInfo()
{
  ostringstream retval;
  retval << "name = " << name_ << endl;
  return retval.str();
}

string
SurfaceGeom::getTypeName(int n){
  return typeName(n);
}


} // End namespace SCIRun
