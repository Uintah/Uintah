//  ContourGeom.cc - A group of Tets in 3 space
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/ContourGeom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID ContourGeom::type_id("ContourGeom", "Datatype", 0);

DebugStream ContourGeom::dbg("ContourGeom", true);

ContourGeom::ContourGeom(const vector<NodeSimp>& inodes, const vector<EdgeSimp>& iedges):
  has_bbox(0), has_neighbors(0)
{
  nodes = inodes;
  edges = iedges;
}

ContourGeom::~ContourGeom(){
}

string ContourGeom::get_info(){
  ostringstream retval;
  retval << "name = " << name << endl;
  return retval.str();
}

bool ContourGeom::compute_bbox(){
  //compute diagnal and bbox
  dbg << "calling Contourgeom::compute_bbox()" << endl;
  
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
  
void ContourGeom::set_nodes(const vector<NodeSimp>& inodes){
  nodes.clear();
  nodes = inodes;
}

void ContourGeom::set_edges(const vector<EdgeSimp>& iedges){
  edges.clear();
  edges = iedges;
}

void ContourGeom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
