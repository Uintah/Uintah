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

void 
LatVolMesh::get_center(Point &result, node_index idx) const
{
  double xgap,ygap,zgap;

  // compute the distance between slices
  xgap = (max_.x()-min_.x())/(nx_-1);
  ygap = (max_.y()-min_.y())/(ny_-1);
  zgap = (max_.z()-min_.z())/(nz_-1);
  
  // return the node_index converted to object space
  result.x(min_.x()+idx.i_*xgap);
  result.y(min_.y()+idx.j_*ygap);
  result.z(min_.z()+idx.k_*zgap);
}

void
LatVolMesh::get_point(Point &result, node_index index) const
{ 
  get_center(result,index);
}

void 
LatVolMesh::get_center(Point &result, cell_index idx) const
{
  node_array nodes;
  Point min,max;

  // get the node_indeces inside of this cell
  get_nodes(nodes,idx);

  // convert the min and max nodes of the cell into object space points
  get_point(min,nodes[0]); // node 0 = min
  get_point(max,nodes[6]); // node 6 = max

  // return the point half way between min and max
  result.x(min.x()+(max.x()-min.x())*.5);
  result.y(min.y()+(max.y()-min.y())*.5);
  result.z(min.z()+(max.z()-min.z())*.5);
}

bool
LatVolMesh::locate(cell_index &cell, const Point &p) const
{
  double xgap,ygap,zgap;
  Point min,max;

  // compute the distance between slices
  xgap = (max_.x()-min_.x())/(nx_-1);
  ygap = (max_.y()-min_.y())/(ny_-1);
  zgap = (max_.z()-min_.z())/(nz_-1);

  // compute the cell_index (divide and truncate)
  cell.i_ = (unsigned)(p.x()/xgap);
  cell.j_ = (unsigned)(p.y()/ygap);
  cell.k_ = (unsigned)(p.z()/zgap);

  // compute the min and max Points of the cell
  min.x(cell.i_*xgap);
  min.y(cell.j_*ygap);
  min.z(cell.k_*zgap);
  max.x(min.x()+xgap);
  max.y(min.y()+ygap);
  max.z(min.z()+zgap);

  return true;
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
  stream.begin_class(type_name().c_str(), LATVOLMESH_VERSION);

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
