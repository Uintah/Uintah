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
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID LatVolMesh::type_id("LatVolMesh", "MeshBase", maker);


LatVolMesh::~LatVolMesh()
{
}


BBox 
LatVolMesh::get_bounding_box() const
{
  BBox result;
  result.extend(min_);
  result.extend(max_);
  return result;
}


inline void
LatVolMesh::unlocate(Point &result, const Point &p) const
{
  result = p;
}


inline void
LatVolMesh::get_point(Point &result, const node_index &index) const
{
  get_center(result,index);
}


inline LatVolMesh::node_iterator
LatVolMesh::node_begin() const
{
  return node_iterator(this, 0, 0, 0);
}


inline LatVolMesh::node_iterator
LatVolMesh::node_end() const
{
  return node_iterator(this, 0, 0, nz_);
}

inline LatVolMesh::edge_iterator
LatVolMesh::edge_begin() const
{
  return 0;
}


inline LatVolMesh::edge_iterator
LatVolMesh::edge_end() const
{
  return 0;
}


inline LatVolMesh::face_iterator
LatVolMesh::face_begin() const
{
  return 0;
}


inline LatVolMesh::face_iterator
LatVolMesh::face_end() const
{
  return 0;
}

inline LatVolMesh::cell_iterator
LatVolMesh::cell_begin() const
{
  return cell_iterator(this, 0, 0, 0);
}


inline LatVolMesh::cell_iterator
LatVolMesh::cell_end() const
{
  return cell_iterator(this, 0, 0, nz_-1);
}

inline void
LatVolMesh::get_nodes(node_array &, edge_index) const
{
}

inline void 
LatVolMesh::get_nodes(node_array &, face_index) const
{
}

inline void 
LatVolMesh::get_nodes(node_array &array, cell_index idx) const
{
  node_index a;

  // return the node_indexex  in this cell
  a.i_ = idx.i_; a.j_ = idx.j_; a.k_ = idx.k_;
  array[0] = a;
  array[1] = array[0]; array[1].i_+=1;
  array[2] = array[0]; array[2].j_+=1;
  array[3] = array[0]; array[3].i_+=1; array[3].j_+=1;

  array[4] = array[0]; array[4].k_+=1;
  array[5] = array[1]; array[5].k_+=1;
  array[6] = array[2]; array[6].k_+=1;
  array[7] = array[3]; array[7].k_+=1;
}

inline void 
LatVolMesh::get_edges(edge_array &, face_index) const
{
}

inline void 
LatVolMesh::get_edges(edge_array &, cell_index) const
{
}

inline void 
LatVolMesh::get_faces(face_array &, cell_index) const
{
}

inline unsigned
LatVolMesh::get_edges(edge_array &, node_index) const
{
  return 0;
}

inline unsigned
LatVolMesh::get_faces(face_array &, node_index) const
{
  return 0;
}

inline unsigned
LatVolMesh::get_faces(face_array &, edge_index) const
{
  return 0;
}

inline unsigned
LatVolMesh::get_cells(cell_array &, node_index) const
{
  return 0;
}

inline unsigned
LatVolMesh::get_cells(cell_array &, edge_index) const
{
  return 0;
}

inline unsigned
LatVolMesh::get_cells(cell_array &, face_index) const
{
  return 0;
}

inline void 
LatVolMesh::get_neighbor(cell_index &, face_index) const
{
}

inline void 
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

inline void 
LatVolMesh::get_center(Point &, edge_index) const
{
}

inline void 
LatVolMesh::get_center(Point &, face_index) const
{
}

inline void 
LatVolMesh::get_center(Point &result, cell_index idx) const
{
  node_array nodes;
  Point min,max;

  // get the node_indeces inside of this cell
  get_nodes(nodes,idx);

  // convert the min and max nodes of the cell into object space points
  get_point(min,nodes[0]);
  get_point(max,nodes[7]);

  // return the point half way between min and max
  result.x(min.x()+(max.x()-min.x())*.5);
  result.y(min.y()+(max.y()-min.y())*.5);
  result.z(min.z()+(max.z()-min.z())*.5);
}

inline void 
LatVolMesh::locate_node(node_index &node, const Point &p) const
{
  double w[8];          // storage for weights
  node_array nodes;     // storage for node_indeces
  cell_index cell;
  double max;
  int loop;

  // locate the cell enclosing the point (including weights)
  locate_cell(cell,p,w);

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
}

inline void 
LatVolMesh::locate_edge(edge_index &, const Point &, double[2]) const
{
}

inline void 
LatVolMesh::locate_face(face_index &, const Point &, double[4]) const 
{
}

inline void 
LatVolMesh::locate_cell(cell_index &cell, const Point &p, double w[8]) const
{
  double xgap,ygap,zgap;
  Point min,max;
  double fx,fy,fz;

  // compute the distance between slices
  xgap = (max_.x()-min_.x())/(nx_-1);
  ygap = (max_.y()-min_.y())/(ny_-1);
  zgap = (max_.z()-min_.z())/(nz_-1);

  // compute normalization factors for a milliunit cube 
  fx = 1000./xgap;
  fy = 1000./ygap;
  fz = 1000./zgap;

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

  // use opposite corner volumes as weights, renormalized to unit
  w[0] = (max.x()-p.x())*fx*(max.y()-p.y())*fy*(max.z()-p.z())*fz*.001;
  w[1] = (p.x()-min.x())*fx*(max.y()-p.y())*fy*(max.z()-p.z())*fz*.001;
  w[2] = (max.x()-p.x())*fx*(p.y()-min.y())*fy*(max.z()-p.z())*fz*.001; 
  w[3] = (p.x()-min.x())*fx*(p.y()-min.y())*fy*(max.z()-p.z())*fz*.001;
  w[4] = (max.x()-p.x())*fx*(max.y()-p.y())*fy*(p.z()-min.z())*fz*.001;
  w[5] = (p.x()-min.x())*fx*(max.y()-p.y())*fy*(p.z()-min.z())*fz*.001;
  w[6] = (max.x()-p.x())*fx*(p.y()-min.y())*fy*(p.z()-min.z())*fz*.001;
  w[7] = (p.x()-min.x())*fx*(p.y()-min.y())*fy*(p.z()-min.z())*fz*.001;
}

#define LATVOLMESH_VERSION 1

void
LatVolMesh::io(Piostream& stream)
{
  stream.begin_class(type_id.type.c_str(), LATVOLMESH_VERSION);

  // IO data members, in order
  Pio(stream, nx_);
  Pio(stream, ny_);
  Pio(stream, nz_);
  Pio(stream, min_);
  Pio(stream, max_);

  stream.end_class();
}

const string 
LatVolMesh::type_name(int)
{
  static const string name = "LatVolMesh";
  return name;
}


} // namespace SCIRun
