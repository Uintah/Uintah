//  ContourGeom.cc - A group of Tets in 3 space
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/ContourGeom.h>
#include <Core/Persistent/PersistentSTL.h>
#include <sstream>

namespace SCIRun {

//////////
// PIO support
static Persistent* maker(){
  return new ContourGeom();
}

string ContourGeom::typeName(int){
  static string typeName = "ContourGeom";
  return typeName;
}

PersistentTypeID ContourGeom::type_id(ContourGeom::typeName(0),
				      UnstructuredGeom::typeName(0),
				      maker);

#define CONTOURGEOM_VERSION 1
void ContourGeom::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(), CONTOURGEOM_VERSION);
  PointCloudGeom::io(stream);
  Pio(stream, edge_);
  stream.end_class();
}

DebugStream ContourGeom::dbg("ContourGeom", true);

//////////
// Constructors/Destructor
ContourGeom::ContourGeom()
{
}

ContourGeom::ContourGeom(const vector<NodeSimp>& inodes,
			 const vector<EdgeSimp>& iedges)
  : PointCloudGeom(inodes)
{
  edge_ = iedges;
}

ContourGeom::~ContourGeom()
{
}

string
ContourGeom::getInfo()
{
  std::ostringstream retval;
  retval << "name = " << name_ << endl;
  return retval.str();
}

string
ContourGeom::getTypeName(int n){
  return typeName(n);
}


void
ContourGeom::setEdges(const vector<EdgeSimp>& iedges)
{
  edge_.clear();
  edge_ = iedges;
}

} // End namespace SCIRun
