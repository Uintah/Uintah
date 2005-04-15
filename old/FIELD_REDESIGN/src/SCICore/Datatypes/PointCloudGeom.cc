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

namespace SCICore {
namespace Datatypes {

PersistentTypeID PointCloudGeom::type_id("PointCloudGeom", "Datatype", 0);

DebugStream PointCloudGeom::dbg("PointCloudGeom", true);

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

void
PointCloudGeom::io(Piostream&)
{
}


} // end Datatypes
} // end SCICore
