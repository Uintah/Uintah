//  SurfaceGeom.cc - A group of Tets in 3 space
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/SurfaceGeom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID SurfaceGeom::type_id("SurfaceGeom", "Datatype", 0);

DebugStream SurfaceGeom::dbg("SurfaceGeom", true);

SurfaceGeom::SurfaceGeom()
{
}


string
SurfaceGeom::get_info()
{
  ostringstream retval;
  retval << "name = " << name << endl;
  return retval.str();
}

bool
SurfaceGeom::compute_bbox()
{
  //compute diagnal and bbox
  dbg << "calling Surfacegeom::compute_bbox()" << endl;
  
  if (nodes.empty())
    {
      return false;
    }

  Point min, max;
  min = max = nodes[0].p;
  for (int i = 1; i < nodes.size(); i ++)
    {
      min = Min(min, nodes[i].p);
      max = Max(max, nodes[i].p);
    }

  bbox.reset();
  bbox.extend(min);
  bbox.extend(max);

  return true;
}
  
void
SurfaceGeom::set_nodes(const vector<NodeSimp>& inodes)
{
  nodes.clear();
  nodes = inodes;
}

void SurfaceGeom::io(Piostream&)
{
}


} // end Datatypes
} // end SCICore
