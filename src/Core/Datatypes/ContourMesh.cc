/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

  for (Node::iterator i = node_begin();
       i!=node_end();
       ++i) 
    result.extend(nodes_[*i]);

  return result;
}

bool
ContourMesh::locate(Node::index_type &idx, const Point &p) const
{
  Node::iterator ni = node_begin();
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
ContourMesh::locate(Edge::index_type &idx, const Point &p) const
{
  Edge::iterator ei;
  double cosa, closest=DBL_MAX;
  Node::array_type nra;
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
  stream.begin_class(type_name(), CONTOURMESH_VERSION);

  MeshBase::io(stream);

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
