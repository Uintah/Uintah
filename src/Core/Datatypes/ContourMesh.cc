/*
 *  ContourMesh.cc: contour mesh
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/ContourMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID ContourMesh::type_id("ContourMesh", "MeshBase", maker);

BBox 
ContourMesh::get_bounding_box() const
{
  BBox result;

  for (node_iterator i = node_begin();
       i!=node_end();
       ++i) 
    result.extend(nodes_[*i]);

  return result;
}

bool
ContourMesh::locate(node_index &node, const Point &p) const
{
  return false;
}

bool
ContourMesh::locate(edge_index &node, const Point &p) const
{
  return false;
}

#define CONTOURMESH_VERSION 1

void
ContourMesh::io(Piostream& stream)
{
  stream.begin_class(type_name().c_str(), CONTOURMESH_VERSION);

  // IO data members, in order
  Pio(stream,nodes_);
  Pio(stream,edges_);

  stream.end_class();
}

const string 
ContourMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "ContourMesh";
  return name;
}


} // namespace SCIRun
