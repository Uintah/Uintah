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
 *  ScanlineMesh.cc: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan &&
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID ScanlineMesh::type_id("ScanlineMesh", "MeshBase", maker);


BBox 
ScanlineMesh::get_bounding_box() const
{
  BBox result;
  result.extend(min_);
  result.extend(max_);
  return result;
}

void 
ScanlineMesh::get_nodes(node_array &array, edge_index idx) const
{
  array.resize(2);
  array[0] = node_index(idx);
  array[1] = node_index(idx + 1);
}

//! return all cell_indecies that overlap the BBox in arr.
void 
ScanlineMesh::get_edges(edge_array &/* arr */, const BBox &/*bbox*/) const
{
  // TODO: implement this
}


void 
ScanlineMesh::get_center(Point &result, node_index idx) const
{
  const double sx = (max_.x() - min_.x()) / (length_ - 1); 

  result.x(idx * sx + min_.x());
  result.y(0);
  result.z(0);
}


void 
ScanlineMesh::get_center(Point &result, edge_index idx) const
{
  const double sx = (max_.x() - min_.x()) / (length_ - 1); 

  result.x((idx + 0.5) * sx + min_.x());
  result.y(0);
  result.z(0);
}

// TODO: verify
bool
ScanlineMesh::locate(edge_index &cell, const Point &p) const
{
  double i = (p.x() - min_.x()) / (max_.x() - min_.x()) * (length_ - 1) + 0.5;

  cell = (unsigned int)i;

  if (cell >= (length_ - 1))
  {
    return false;
  }
  else
  {
    return true;
  }
}


// TODO: verify
bool
ScanlineMesh::locate(node_index &node, const Point &p) const
{
  node_array nodes;     // storage for node_indeces
  cell_index cell;
  double max;
  int loop;

  // locate the cell enclosing the point (including weights)
  if (!locate(cell,p)) return false;
  weight_array w;
  calc_weights(this, cell, p, w);

  // get the node_indeces in this cell
  get_nodes(nodes,cell);

  // find, and return, the "heaviest" node
  max = w[0];
  loop=1;
  while (loop<8) {
    if (w[loop]>max) {
      max=w[loop];
      node=nodes[loop];
    }
  }
  return true;
}


#define LATVOLMESH_VERSION 1

void
ScanlineMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), LATVOLMESH_VERSION);

  MeshBase::io(stream);

  // IO data members, in order
  Pio(stream, length_);
  Pio(stream, min_);
  Pio(stream, max_);

  stream.end_class();
}

const string 
ScanlineMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "ScanlineMesh";
  return name;
}


} // namespace SCIRun
