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
ContourMesh::locate(node_index &idx, const Point &p) const
{
  node_iterator ni = node_begin();
  node_iterator found = node_begin();

  if (ni==node_end())
    return false;

  double closest = (p-nodes_[*ni]).length2();

  ++ni;
  for (; ni != node_end(); ++ni) {
    if ( (p-nodes_[*ni]).length2() < closest ) {
      closest = (p-nodes_[*ni]).length2();
      found = ni;
    }
  }

  idx = *found;

  return true;
}

bool
ContourMesh::locate(edge_index &, const Point &) const
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
