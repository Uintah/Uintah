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
#include <Core/Geometry/Vector.h>
#include <float.h>  // for DBL_MAX
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
  idx = *node_begin();

  if (ni==node_end())
    return false;

  double closest = (p-nodes_[*ni]).length2();

  ++ni;
  for (; ni != node_end(); ++ni) {
    if ( (p-nodes_[*ni]).length2() < closest ) {
      closest = (p-nodes_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}

bool
ContourMesh::locate(edge_index &idx, const Point &p) const
{
  edge_iterator ei;
  double cosa, closest=DBL_MAX;
  node_array nra;
  double dist1, dist2, dist3, dist4;
  Point n1,n2,q;

  if (ei==edge_end())
    return false;
  
  for (ei = edge_begin(); ei != edge_end(); ++ei) {
    get_nodes(nra,*ei);

    n1 = nodes_[nra[0]];
    n2 = nodes_[nra[1]];

    dist1 = (p-n1).length();
    dist2 = (p-n2).length();
    dist3 = (n1-n2).length();

    cosa = Dot(n1-p,n1-n2)/((n1-p).length()*dist3);

    q = n1 + (n1-n2) * (n1-n2)/dist3;

    dist4 = (p-q).length();

    if ( (cosa > 0) && (cosa < dist3) && (dist4 < closest) ) {
      closest = dist4;
      idx = *ei;
    } else if ( (cosa < 0) && (dist1 < closest) ) {
      closest = dist1;
      idx = *ei;
    } else if ( (cosa > dist3) && (dist2 < closest) ) {
      closest = dist2;
      idx = *ei;
    }
  }

  return true;
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
