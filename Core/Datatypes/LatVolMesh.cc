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
 *  LatVolMesh.cc: Templated Mesh defined on a 3D Regular Grid
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

#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID LatVolMesh::type_id("LatVolMesh", "MeshBase", maker);


BBox 
LatVolMesh::get_bounding_box() const
{
  BBox result;
  result.extend(min_);
  result.extend(max_);
  return result;
}

void 
LatVolMesh::get_nodes(node_array &array, cell_index idx) const
{
  array.resize(8);
  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;   array[0].k_ = idx.k_; 
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;   array[1].k_ = idx.k_; 
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1; array[2].k_ = idx.k_; 
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1; array[3].k_ = idx.k_; 
  array[4].i_ = idx.i_;   array[4].j_ = idx.j_;   array[4].k_ = idx.k_+1;
  array[5].i_ = idx.i_+1; array[5].j_ = idx.j_;   array[5].k_ = idx.k_+1;
  array[6].i_ = idx.i_+1; array[6].j_ = idx.j_+1; array[6].k_ = idx.k_+1;
  array[7].i_ = idx.i_;   array[7].j_ = idx.j_+1; array[7].k_ = idx.k_+1;
}

//! return all cell_indecies that overlap the BBox in arr.
void 
LatVolMesh::get_cells(cell_array &arr, const BBox &bbox) const
{
  arr.clear();
  cell_index min;
  locate(min, bbox.min());
  cell_index max;
  locate(max, bbox.max());
  
  if (max.i_ >= nx_ - 1) max.i_ = nx_ - 2;
  if (max.j_ >= ny_ - 1) max.j_ = ny_ - 2;
  if (max.k_ >= nz_ - 1) max.k_ = nz_ - 2;

  for (unsigned i = min.i_; i <= max.i_; i++) {
    for (unsigned j = min.j_; j <= max.j_; j++) {
      for (unsigned k = min.k_; k <= max.k_; k++) {
	arr.push_back(cell_index(i,j,k));
      }
    }
  }
}


void 
LatVolMesh::get_center(Point &result, node_index idx) const
{
  const double sx = (max_.x() - min_.x()) / (nx_ - 1); 
  const double sy = (max_.y() - min_.y()) / (ny_ - 1); 
  const double sz = (max_.z() - min_.z()) / (nz_ - 1);

  result.x(idx.i_ * sx + min_.x());
  result.y(idx.j_ * sy + min_.y());
  result.z(idx.k_ * sz + min_.z());
}


void 
LatVolMesh::get_center(Point &result, cell_index idx) const
{
  const double sx = (max_.x() - min_.x()) / (nx_ - 1); 
  const double sy = (max_.y() - min_.y()) / (ny_ - 1); 
  const double sz = (max_.z() - min_.z()) / (nz_ - 1);

  result.x((idx.i_ + 0.5) * sx + min_.x());
  result.y((idx.j_ + 0.5) * sy + min_.y());
  result.z((idx.k_ + 0.5) * sz + min_.z());
}

bool
LatVolMesh::locate(cell_index &cell, const Point &p) const
{
  double i = (p.x() - min_.x()) / (max_.x() - min_.x()) * (nx_ - 1) + 0.5;
  double j = (p.y() - min_.y()) / (max_.y() - min_.y()) * (ny_ - 1) + 0.5;
  double k = (p.z() - min_.z()) / (max_.z() - min_.z()) * (nz_ - 1) + 0.5;

  cell.i_ = (unsigned int)i;
  cell.j_ = (unsigned int)j;
  cell.k_ = (unsigned int)k;

  if (cell.i_ >= (nx_-1) ||
      cell.j_ >= (ny_-1) ||
      cell.k_ >= (nz_-1))
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool
LatVolMesh::locate(node_index &node, const Point &p) const
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
LatVolMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), LATVOLMESH_VERSION);

  MeshBase::io(stream);

  // IO data members, in order
  Pio(stream, nx_);
  Pio(stream, ny_);
  Pio(stream, nz_);
  Pio(stream, min_);
  Pio(stream, max_);

  stream.end_class();
}

const string 
LatVolMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "LatVolMesh";
  return name;
}


} // namespace SCIRun
