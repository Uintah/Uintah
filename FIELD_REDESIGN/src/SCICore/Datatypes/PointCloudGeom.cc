//  PointCloudGeom.cc - A group of Nodes in 3 space
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/PointCloudGeom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID PointCloudGeom::type_id("PointCloudGeom", "Datatype", 0);

DebugStream PointCloudGeom::dbg("PointCloudGeom", true);

PointCloudGeom::PointCloudGeom(const vector<NodeSimp>& inodes):
  has_bbox(0)
{
  nodes = inodes;
}

PointCloudGeom::~PointCloudGeom(){
}

string PointCloudGeom::get_info(){
  ostringstream retval;
  retval << "name = " << name << endl;
  return retval.str();
}

bool PointCloudGeom::compute_bbox(){
  //compute diagnal and bbox
  dbg << "calling PointCloudgeom::compute_bbox()" << endl;
  
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
  
void PointCloudGeom::set_nodes(const vector<NodeSimp>& inodes){
  nodes.clear();
  nodes = inodes;
}

void PointCloudGeom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
