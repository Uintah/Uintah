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

namespace SCICore {
namespace Datatypes {

PersistentTypeID ContourGeom::type_id("ContourGeom", "Datatype", 0);

DebugStream ContourGeom::dbg("ContourGeom", true);


ContourGeom::ContourGeom()
{
}

ContourGeom::ContourGeom(const vector<NodeSimp>& inodes,
			 const vector<EdgeSimp>& iedges)
  : PointCloudGeom(inodes)
{
  d_edge = iedges;
}

ContourGeom::~ContourGeom()
{
}

string
ContourGeom::getInfo()
{
  ostringstream retval;
  retval << "name = " << d_name << endl;
  return retval.str();
}

void
ContourGeom::setEdges(const vector<EdgeSimp>& iedges)
{
  d_edge.clear();
  d_edge = iedges;
}

void ContourGeom::io(Piostream&)
{
}


} // end Datatypes
} // end SCICore
