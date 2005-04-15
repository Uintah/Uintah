//  TriSurfaceGeom.cc - A group of Tets in 3 space
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/TriSurfaceGeom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID TriSurfaceGeom::type_id("TriSurfaceGeom", "Datatype", 0);

DebugStream TriSurfaceGeom::dbg("TriSurfaceGeom", true);

TriSurfaceGeom::TriSurfaceGeom(const vector<NodeSimp>& inodes, const vector<FaceSimp>& ifaces):
  has_neighbors(0)
{
  nodes = inodes;
  faces = ifaces;
}

TriSurfaceGeom::~TriSurfaceGeom(){
}

void TriSurfaceGeom::set_faces(const vector<FaceSimp>& ifaces){
  faces.clear();
  faces = ifaces;
}


void TriSurfaceGeom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
