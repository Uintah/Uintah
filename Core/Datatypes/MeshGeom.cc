//  MeshGeom.cc - A group of Tets in 3 space
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/MeshGeom.h>

namespace SCIRun {

//////////
// PIO support
static Persistent* maker(){
  return new MeshGeom();
}

string MeshGeom::typeName(int){
  static string className = "MeshGeom";
  return className;
}

PersistentTypeID MeshGeom::type_id(MeshGeom::typeName(0), 
				   SurfaceGeom::typeName(0), 
				   maker);

#define MESHGEOM_VERSION 1
void
MeshGeom::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(), MESHGEOM_VERSION);
  SurfaceGeom::io(stream);
  Pio(stream, cell_);
  stream.end_class();
}

DebugStream MeshGeom::dbg("MeshGeom", true);

//////////
// Constructors/Destructor
MeshGeom::MeshGeom()
{
}

string
MeshGeom::getInfo()
{
  ostringstream retval;
  retval << "name = " << name_ << endl;
  return retval.str();
}

string
MeshGeom::getTypeName(int n){
  return typeName(n);
}

} // End namespace SCIRun
