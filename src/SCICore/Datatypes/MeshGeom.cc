//  MeshGeom.cc - A group of Tetrahedrals in 3 space
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/MeshGeom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID MeshGeom::type_id("MeshGeom", "Datatype", 0);

DebugStream MeshGeom::dbg("MeshGeom", true);

MeshGeom::MeshGeom(){
  has_bbox = false;
}

MeshGeom::~MeshGeom(){
}

string MeshGeom::get_info(){
  ostringstream retval;
  retval <<
    "name = " << name << endl;
  return retval.str();
}

bool MeshGeom::compute_bbox(){
  //compute diagnal and bbox
  dbg << "calling meshgeom::comput_bbox()" << endl;
  
  if(nodes.empty()){
    return false;
  }
  has_bbox = true;
  Point min, max;
  min = max = nodes[0].p;
  for(int i = 1; i < nodes.size(); i ++){
    min = Min(min, nodes[i].p);
    max = Max(max, nodes[i].p);
  }
  bbox.reset();
  bbox.extend(min);
  bbox.extend(max);
  diagonal = bbox.max()-bbox.min();
  return true;
}


    
void MeshGeom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
