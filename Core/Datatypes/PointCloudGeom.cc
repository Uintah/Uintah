//  PointCloudGeom.cc - A group of Nodes in 3 space
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/PointCloudGeom.h>
#include <Core/Persistent/PersistentSTL.h>

namespace SCIRun {

//////////
// PIO support
static Persistent* maker(){
  return new PointCloudGeom();
}

string PointCloudGeom::typeName(int){
  static string className = "PointCloudGeom";
  return className;
}

PersistentTypeID PointCloudGeom::type_id(PointCloudGeom::typeName(0), 
					 UnstructuredGeom::typeName(0), 
					 maker);

#define POINTCLOUDGEOM_VERSION 1
void
PointCloudGeom::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(),  POINTCLOUDGEOM_VERSION);
  Pio(stream, d_node);
  stream.end_class();
}

DebugStream PointCloudGeom::dbg("PointCloudGeom", true);

//////////
// Constructors/Destructor
PointCloudGeom::PointCloudGeom()
{
}


PointCloudGeom::PointCloudGeom(const vector<NodeSimp>& inodes)
{
  d_node = inodes;
}


PointCloudGeom::~PointCloudGeom()
{
}

string
PointCloudGeom::getInfo()
{
  ostringstream retval;
  retval << "name = " << d_name << endl;
  return retval.str();
}

string
PointCloudGeom::getTypeName(int n){
  return typeName(n);
}

bool
PointCloudGeom::computeBoundingBox()
{
  //compute diagnal and bbox
  dbg << "calling PointCloudgeom::compute_bbox()" << endl;
  
  if (d_node.empty())
  {
    return false;
  }

  Point min, max;
  min = max = d_node[0].p;
  for (int i = 1; i < d_node.size(); i ++)
  {
    min = Min(min, d_node[i].p);
    max = Max(max, d_node[i].p);
  }

  d_bbox.reset();
  d_bbox.extend(min);
  d_bbox.extend(max);

  return true;
}

  
void
PointCloudGeom::setNodes(const vector<NodeSimp>& inodes)
{
  d_node.clear();
  d_node = inodes;
}

} // End namespace SCIRun
