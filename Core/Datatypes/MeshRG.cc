/*
 *  MeshRG.cc: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/MeshRG.h>


namespace SCIRun {


PersistentTypeID MeshRG::type_id("MeshRG", "Datatype", maker);


Persistent *
MeshRG::maker()
{
  return new MeshRG(0, 0, 0);
}


MeshRG::MeshRG(int x, int y, int z)
  : nx_(x),
    ny_(y),
    nz_(z)
{
}


MeshRG::MeshRG(const MeshRG &copy)
  : nx_(copy.nx_),
    ny_(copy.ny_),
    nz_(copy.nz_)
{
}


MeshRG::~MeshRG()
{
}


BBox 
MeshRG::get_bounding_box() const
{
  BBox result(Point(0, 0, 0), Point(nx_, ny_, nz_));
  return result;
}


void
MeshRG::locate_node(node_index &node, const Point &p)
{
  node.i_ = (int)(p.x() + 0.5);
  node.j_ = (int)(p.y() + 0.5);
  node.k_ = (int)(p.z() + 0.5);
}

void
MeshRG::locate_edge(edge_index & /* edge */, const Point & /* p */)
{
  // NOT IMPLEMENTED
}

void
MeshRG::locate_face(face_index & /* face */, const Point & /* p */)
{
  // NOT IMPLEMENTED
}


void
MeshRG::locate_cell(cell_index &cell, const Point &p)
{
  cell.i_ = (int)p.x();
  cell.j_ = (int)p.y();
  cell.k_ = (int)p.z();
}


void
MeshRG::unlocate(Point &result, const Point &p)
{
  result = p;
}


void
MeshRG::get_point(Point &result, const node_index &index) const
{
  result.x((double)index.i_);
  result.y((double)index.j_);
  result.z((double)index.k_);
}


MeshRG::node_iterator
MeshRG::node_begin() const
{
  return node_iterator(this, 0, 0, 0);
}


MeshRG::node_iterator
MeshRG::node_end() const
{
  return node_iterator(this, nx_, ny_, nz_);
}


MeshRG::edge_iterator
MeshRG::edge_begin() const
{
  return NULL;
}


MeshRG::edge_iterator
MeshRG::edge_end() const
{
  return NULL;
}


MeshRG::face_iterator
MeshRG::face_begin() const
{
  return NULL;
}


MeshRG::face_iterator
MeshRG::face_end() const
{
  return NULL;
}


MeshRG::cell_iterator
MeshRG::cell_begin() const
{
  return cell_iterator(this, 0, 0, 0);
}


MeshRG::cell_iterator
MeshRG::cell_end() const
{
  return cell_iterator(this, nx_-1, ny_-1, nz_-1);
}


#define MESHRG_VERSION 1

void
MeshRG::io(Piostream& stream)
{
  stream.begin_class(type_id.type.c_str(), MESHRG_VERSION);

  // IO data members, in order
  Pio(stream, nx_);
  Pio(stream, ny_);
  Pio(stream, nz_);

  stream.end_class();
}


} // namespace SCIRun
