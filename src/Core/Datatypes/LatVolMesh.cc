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
#include <Core/Math/MinMax.h>


namespace SCIRun {


PersistentTypeID LatVolMesh::type_id("LatVolMesh", "MeshBase", maker);


Persistent *
LatVolMesh::maker()
{
  return new LatVolMesh(0, 0, 0, Point(0, 0, 0), Point(1, 1, 1));
}


LatVolMesh::LatVolMesh(int x, int y, int z, Point &min, Point &max)
  : nx_(x),
    ny_(y),
    nz_(z),
    min_(min),
    max_(max)
{
}


LatVolMesh::LatVolMesh(const LatVolMesh &copy)
  : nx_(copy.nx_),
    ny_(copy.ny_),
    nz_(copy.nz_),
    min_(copy.min_),
    max_(copy.max_)
{
}


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


void
LatVolMesh::unlocate(Point &result, const Point &p) const
{
  result = p;
}


inline void
LatVolMesh::get_point(Point &result, const node_index &index) const
{
  get_center(result,index);
}


LatVolMesh::node_iterator
LatVolMesh::node_begin() const
{
  return node_iterator(0, 0, 0);
}


LatVolMesh::node_iterator
LatVolMesh::node_end() const
{
  return node_iterator(nx_, ny_, nz_);
}


LatVolMesh::edge_iterator
LatVolMesh::edge_begin() const
{
  return NULL;
}


LatVolMesh::edge_iterator
LatVolMesh::edge_end() const
{
  return NULL;
}


LatVolMesh::face_iterator
LatVolMesh::face_begin() const
{
  return NULL;
}


LatVolMesh::face_iterator
LatVolMesh::face_end() const
{
  return NULL;
}


LatVolMesh::cell_iterator
LatVolMesh::cell_begin() const
{
  return cell_iterator(0, 0, 0);
}


LatVolMesh::cell_iterator
LatVolMesh::cell_end() const
{
  return cell_iterator(nx_-1, ny_-1, nz_-1);
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
  // NOTE: this code assumes that index_type is actually unsigned

  index_type x,y,z;
  index_type nxnya = (nx_-1)*(ny_-1);
  index_type nxnyb = nx_*ny_;
  index_type div;
  index_type mod;
  index_type a,b,c,d;
 
  // decompose the cell_index into it's xyz components
  mod = idx%nxnya;
  x = mod%(nx_-1);
  y = mod/(nx_-1);
  z = idx/nxnya;

  // convert from cell_index space to node_index space
  a = x+y*nx_+z*nxnyb;

  // compute some neighbor node_indeces
  b = a+nx_;
  c = a+nxnyb;
  d = a+nxnyb+nx_;

  // return the node_indeces 
  array[0] = a;
  array[1] = a+1;
  array[2] = b;
  array[3] = b+1;
  array[4] = c;
  array[5] = c+1;
  array[6] = d;
  array[7] = d+1;
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

inline int 
LatVolMesh::get_edges(edge_array &, node_index) const
{
}

inline int 
LatVolMesh::get_faces(face_array &, node_index) const
{
}

inline int 
LatVolMesh::get_faces(face_array &, edge_index) const
{
}

inline int 
LatVolMesh::get_cells(cell_array &, node_index) const
{
}

inline int 
LatVolMesh::get_cells(cell_array &, edge_index) const
{
}

inline int 
LatVolMesh::get_cells(cell_array &, face_index) const
{
}

inline void 
LatVolMesh::get_neighbor(cell_index &, face_index) const
{
}

inline void 
LatVolMesh::get_center(Point &result, node_index idx) const
{
  // NOTE: this code assumes that index_type is actually unsigned

  index_type x,y,z;
  index_type nxny = nx_*ny_;
  index_type div;
  index_type mod;
  double xgap,ygap,zgap;
 
  // decompose the node_index into it's xyz components
  mod = idx%nxny;
  x = (mod)%nx_;
  y = (mod)/nx_;
  z = idx/nxny;

  // compute the distance between slices
  xgap = (max_.x()-min_.x())/(nx_-1);
  ygap = (max_.y()-min_.y())/(ny_-1);
  zgap = (max_.z()-min_.z())/(nz_-1);
  
  // return the node_index converted to object space
  result.x(min_.x()+x*xgap);
  result.y(min_.y()+y*ygap);
  result.z(min_.z()+z*zgap);
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
  get_center(min,nodes[0]);
  get_center(max,nodes[7]);

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
LatVolMesh::locate_cell(cell_index &cell, const Point &p, double[8] w) const
{
  // NOTE: this code assumes index_type is actually unsigned
  
  double xgap,ygap,zgap;
  Point min,max;
  index_type x,y,z;
  double fx,fy,fz;

  // compute the distance between slices
  xgap = (max_.x()-min_.x())/(nx_-1);
  ygap = (max_.y()-min_.y())/(ny_-1);
  zgap = (max_.z()-min_.z())/(nz_-1);

  // compute normalization factors for a milliunit cube 
  fx = 1000./xgap;
  fy = 1000./ygap;
  fz = 1000./zgap;

  // compute the xyz components of the cell_index
  x = (index_type)(p.x()/xgap);
  y = (index_type)(p.y()/ygap);
  z = (index_type)(p.z()/zgap);

  // compute the min and max Points of the cell
  min.x(x*xgap);
  min.y(y*ygap);
  min.z(y*zgap);
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
  
  // return the cell_index
  cell = x + y*(nx_-1) + z*(nx_-1)*(ny_-1);
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

  stream.end_class();
}


} // namespace SCIRun
