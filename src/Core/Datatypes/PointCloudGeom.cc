//  PointCloudGeom.cc - A group of Nodes in 3 space
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/PointCloudGeom.h>
#include <Core/Persistent/PersistentSTL.h>
#include <sstream>


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
  UnstructuredGeom::io(stream);
  Pio(stream, node_);
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
  node_ = inodes;
}


PointCloudGeom::~PointCloudGeom()
{
}

string
PointCloudGeom::getInfo()
{
  std::ostringstream retval;
  retval << "name = " << name_ << endl;
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
  
  if (node_.empty())
  {
    return false;
  }

  Point min, max;
  min = max = node_[0].p;
  for (int i = 1; i < node_.size(); i ++)
  {
    min = Min(min, node_[i].p);
    max = Max(max, node_[i].p);
  }

  bbox_.reset();
  bbox_.extend(min);
  bbox_.extend(max);

  return true;
}

  
void
PointCloudGeom::setNodes(const vector<NodeSimp>& inodes)
{
  node_.clear();
  node_ = inodes;
}

} // End namespace SCIRun
