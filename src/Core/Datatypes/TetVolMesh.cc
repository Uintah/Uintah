/*
 *  TetVolMesh.cc: Tetrahedral mesh with new design.
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

#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

Persistent* make_TetVolMesh() {
  return scinew TetVolMesh;
}

PersistentTypeID TetVolMesh::type_id("TetVolMesh", "MeshBase", 
				     make_TetVolMesh);

const string
TetVolMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name("TetVolMesh");
  return name;
}


TetVolMesh::TetVolMesh()
{
}

TetVolMesh::TetVolMesh(const TetVolMesh &copy)
  : points_(copy.points_),
    cells_(copy.cells_),
    neighbors_(copy.neighbors_)
{
}

TetVolMesh::~TetVolMesh()
{
}


BBox
TetVolMesh::get_bounding_box() const
{
  BBox result;

  node_iterator ni = node_begin();
  while (ni != node_end())
  {
    Point p;
    get_point(p, *ni);
    result.extend(p);
    ++ni;
  }
  return result;
}

void 
TetVolMesh::hash_face(node_index n1, node_index n2, node_index n3,
		      cell_index ci, hash_set<Face, FaceHash> &table) const {
  Face f;
  f.nodes_[0] = n1;
  f.nodes_[1] = n2;
  f.nodes_[2] = n3;
  hash_set<Face, FaceHash>::iterator iter = table.find(f);
  if (iter == table.end()) {
    f.cells_[0] = ci;
    table.insert(f); // insert for the first time
  } else {
    Face f = *iter;
    f.cells_[1] = ci; // add this cell
    table.erase(iter);
    table.insert(f);
  }
}

void 
TetVolMesh::compute_faces()
{
  hash_set<Face, FaceHash> table;

  cell_iterator ci = cell_begin();
  while (ci != cell_end()) {
    node_array arr;
    get_nodes(arr, *ci); ++ci;
    // 4 faces
    hash_face(arr[0], arr[1], arr[2], *ci, table);
    hash_face(arr[0], arr[1], arr[3], *ci, table);
    hash_face(arr[0], arr[2], arr[3], *ci, table);
    hash_face(arr[1], arr[2], arr[3], *ci, table);
  }
  // dump edges into the edges_ container.
  faces_.resize(table.size());
  vector<Face>::iterator              f_iter = faces_.begin();
  hash_set<Face, FaceHash>::iterator ht_iter = table.begin();
  while (ht_iter != table.end()) {
    *f_iter = *ht_iter;
    ++f_iter; ++ht_iter;
  }
}

void 
TetVolMesh::hash_edge(node_index n1, node_index n2, 
		      cell_index ci, hash_set<Edge, EdgeHash> &table) const {
  Edge e;
  e.nodes_[0] = n1;
  e.nodes_[1] = n2;
  hash_set<Edge, EdgeHash>::iterator iter = table.find(e);
  if (iter == table.end()) {
    table.insert(e); // insert for the first time
  } else {
    Edge e = *iter;
    e.cells_.push_back(ci); // add this cell
    table.erase(iter);
    table.insert(e);
  }
}

void 
TetVolMesh::compute_edges()
{
  hash_set<Edge, EdgeHash> table;

  cell_iterator ci = cell_begin();
  while (ci != cell_end()) {
    node_array arr;
    get_nodes(arr, *ci); ++ci;
    // 6 edges
    hash_edge(arr[0], arr[1], *ci, table);
    hash_edge(arr[0], arr[2], *ci, table);
    hash_edge(arr[0], arr[3], *ci, table);
    hash_edge(arr[1], arr[2], *ci, table);
    hash_edge(arr[1], arr[3], *ci, table);
    hash_edge(arr[2], arr[3], *ci, table);
  }
  // dump edges into the edges_ container.
  edges_.resize(table.size());
  vector<Edge>::iterator              e_iter = edges_.begin();
  hash_set<Edge, EdgeHash>::iterator ht_iter = table.begin();
  while (ht_iter != table.end()) {
    *e_iter = *ht_iter;
    ++e_iter; ++ht_iter;
  }
  
}

void 
TetVolMesh::finish() {
  compute_edges();
  compute_faces();
}


TetVolMesh::node_iterator
TetVolMesh::node_begin() const
{
  return 0;
}

TetVolMesh::node_iterator
TetVolMesh::node_end() const
{
  return points_.size();
}

TetVolMesh::edge_iterator
TetVolMesh::edge_begin() const
{
  return 0;
}

TetVolMesh::edge_iterator
TetVolMesh::edge_end() const
{
  return edges_.size();
}

TetVolMesh::face_iterator
TetVolMesh::face_begin() const
{
  return 0;
}

TetVolMesh::face_iterator
TetVolMesh::face_end() const
{
  return faces_.size();
}

TetVolMesh::cell_iterator
TetVolMesh::cell_begin() const
{
  return 0;
}

TetVolMesh::cell_iterator
TetVolMesh::cell_end() const
{
  return cells_.size() >> 2;
}


void
TetVolMesh::get_nodes(node_array &array, edge_index idx) const
{
  Edge e = edges_[idx];
  array.push_back(e.nodes_[0]); 
  array.push_back(e.nodes_[1]);
}


void
TetVolMesh::get_nodes(node_array &array, face_index idx) const
{
  Face f = faces_[idx];
  array.push_back(f.nodes_[0]); 
  array.push_back(f.nodes_[1]);
  array.push_back(f.nodes_[2]);
}


void
TetVolMesh::get_nodes(node_array &array, cell_index idx) const
{
  array.push_back(cells_[idx * 4 + 0]);
  array.push_back(cells_[idx * 4 + 1]);
  array.push_back(cells_[idx * 4 + 2]);
  array.push_back(cells_[idx * 4 + 3]);
}


void
TetVolMesh::get_edges(edge_array &array, face_index idx) const
{
  static int table[4][3] =
  {
    {3, 4, 5},
    {1, 2, 5},
    {0, 2, 4},
    {0, 1, 3}
  };

  int base = idx / 4 * 6;
  int off = idx % 4;

  array.push_back(base + table[off][0]);
  array.push_back(base + table[off][1]);
  array.push_back(base + table[off][2]);
  array.push_back(base + table[off][3]);
}


void
TetVolMesh::get_edges(edge_array &array, cell_index index) const
{
  cell_index::value_type idx = index;
  idx *= 6;
  array.push_back(idx);
  array.push_back(++idx);
  array.push_back(++idx);
  array.push_back(++idx);
  array.push_back(++idx);
  array.push_back(++idx);
}


void
TetVolMesh::get_faces(face_array &array, cell_index index) const
{
  cell_index::value_type idx = index;
  idx *= 4;
  array.push_back(idx);
  array.push_back(++idx);
  array.push_back(++idx);
  array.push_back(++idx);
}

bool
TetVolMesh::get_neighbor(cell_index &neighbor, cell_index from,
			 face_index idx) const
{
  const Face &f= faces_[idx];
  if (from == f.cells_[0]) {
    neighbor = f.cells_[1];
  } else { 
    neighbor = f.cells_[0];
  }
  if (neighbor == -1) return false;
  return true;
}

void 
TetVolMesh::get_center(Point &p, node_index idx) const
{
  get_point(p, idx);
}

void 
TetVolMesh::get_center(Point &p, edge_index idx) const
{
  const double s = 1./2.;
  node_array arr;
  get_nodes(arr, idx);
  Point p1;
  get_point(p, arr[0]);
  get_point(p1, arr[1]);

  p = ((Vector(p) + Vector(p1)) * s).asPoint();
}

void 
TetVolMesh::get_center(Point &p, face_index idx) const
{
  const double s = 1./3.;
  node_array arr;
  get_nodes(arr, idx);
  Point p1, p2;
  get_point(p, arr[0]);
  get_point(p1, arr[1]);
  get_point(p2, arr[2]);

  p = ((Vector(p) + Vector(p1) + Vector(p2)) * s).asPoint();
}

void 
TetVolMesh::get_center(Point &p, cell_index idx) const
{
  const double s = 1./5.;
  node_array arr;
  get_nodes(arr, idx);
  Point p1, p2, p3, p4;
  get_point(p, arr[0]);
  get_point(p1, arr[1]);
  get_point(p2, arr[2]);
  get_point(p3, arr[3]);
  get_point(p4, arr[4]);

  p = ((Vector(p) + Vector(p1) + Vector(p2) + 
	Vector(p3) + Vector(p4)) * s).asPoint();
}

static double
distance2(const Point &p0, const Point &p1)
{
  const double dx = p0.x() - p1.x();
  const double dy = p0.y() - p1.y();
  const double dz = p0.z() - p1.z();
  return dx * dx + dy * dy + dz * dz;
}


bool
TetVolMesh::locate(node_index &loc, const Point &p) const
{
  node_iterator ni = node_begin();
  if (ni == node_end()) { return false; }

  double min_dist = distance2(p, points_[*ni]);
  loc = *ni;
  ++ni;

  while (ni != node_end()) {
    const double dist = distance2(p, points_[*ni]);
    if (dist < min_dist) {
      loc = *ni;
    }
    ++ni;
  }
  return true;
}


bool
TetVolMesh::locate(edge_index &/*edge*/, const Point & /* p */) const
{
  //FIX_ME
  return false;
}


bool
TetVolMesh::locate(face_index &/*face*/, const Point & /* p */) const
{
  //FIX_ME
  return false;
}


bool
TetVolMesh::locate(cell_index &cell, const Point &p) const
{
  bool found_p = false;
  cell_iterator ci = cell_begin();
  while (ci != cell_end()) {
    if (inside4_p((*ci) * 4, p)) {
      found_p = true;
      break;     
    }
  }
  cell = *ci;
  return found_p;
}

void
TetVolMesh::unlocate(Point &result, const Point &p)
{
  result = p;
}


void
TetVolMesh::get_point(Point &result, node_index index) const
{
  result = points_[index];
}


bool
TetVolMesh::inside4_p(int i, const Point &p) const
{
  // TODO: This has not been tested.
  // TODO: Looks like too much code to check sign of 4 plane/point tests.

  const Point &p0 = points_[cells_[i+0]];
  const Point &p1 = points_[cells_[i+1]];
  const Point &p2 = points_[cells_[i+2]];
  const Point &p3 = points_[cells_[i+3]];
  const double x0 = p0.x();
  const double y0 = p0.y();
  const double z0 = p0.z();
  const double x1 = p1.x();
  const double y1 = p1.y();
  const double z1 = p1.z();
  const double x2 = p2.x();
  const double y2 = p2.y();
  const double z2 = p2.z();
  const double x3 = p3.x();
  const double y3 = p3.y();
  const double z3 = p3.z();

  const double a0 = + x1*(y2*z3-y3*z2) + x2*(y3*z1-y1*z3) + x3*(y1*z2-y2*z1);
  const double a1 = - x2*(y3*z0-y0*z3) - x3*(y0*z2-y2*z0) - x0*(y2*z3-y3*z2);
  const double a2 = + x3*(y0*z1-y1*z0) + x0*(y1*z3-y3*z1) + x1*(y3*z0-y0*z3);
  const double a3 = - x0*(y1*z2-y2*z1) - x1*(y2*z0-y0*z2) - x2*(y0*z1-y1*z0);
  const double iV6 = 1.0 / (a0+a1+a2+a3);

  const double b0 = - (y2*z3-y3*z2) - (y3*z1-y1*z3) - (y1*z2-y2*z1);
  const double c0 = + (x2*z3-x3*z2) + (x3*z1-x1*z3) + (x1*z2-x2*z1);
  const double d0 = - (x2*y3-x3*y2) - (x3*y1-x1*y3) - (x1*y2-x2*y1);
  const double s0 = iV6 * (a0 + b0*p.x() + c0*p.y() + d0*p.z());
  if (s0 < -1.e-6)
    return false;

  const double b1 = + (y3*z0-y0*z3) + (y0*z2-y2*z0) + (y2*z3-y3*z2);
  const double c1 = - (x3*z0-x0*z3) - (x0*z2-x2*z0) - (x2*z3-x3*z2);
  const double d1 = + (x3*y0-x0*y3) + (x0*y2-x2*y0) + (x2*y3-x3*y2);
  const double s1 = iV6 * (a1 + b1*p.x() + c1*p.y() + d1*p.z());
  if (s1 < -1.e-6)
    return false;

  const double b2 = - (y0*z1-y1*z0) - (y1*z3-y3*z1) - (y3*z0-y0*z3);
  const double c2 = + (x0*z1-x1*z0) + (x1*z3-x3*z1) + (x3*z0-x0*z3);
  const double d2 = - (x0*y1-x1*y0) - (x1*y3-x3*y1) - (x3*y0-x0*y3);
  const double s2 = iV6 * (a2 + b2*p.x() + c2*p.y() + d2*p.z());
  if (s2 < -1.e-6)
    return false;

  const double b3 = +(y1*z2-y2*z1) + (y2*z0-y0*z2) + (y0*z1-y1*z0);
  const double c3 = -(x1*z2-x2*z1) - (x2*z0-x0*z2) - (x0*z1-x1*z0);
  const double d3 = +(x1*y2-x2*y1) + (x2*y0-x0*y2) + (x0*y1-x1*y0);
  const double s3 = iV6 * (a3 + b3*p.x() + c3*p.y() + d3*p.z());
  if (s3 < -1.e-6)
    return false;

  return true;
}


#define TETVOLMESH_VERSION 1

void
TetVolMesh::io(Piostream &stream)
{
  stream.begin_class(type_id.type.c_str(), TETVOLMESH_VERSION);

  Pio(stream, points_);
  Pio(stream, cells_);
  Pio(stream, neighbors_);

  stream.end_class();
}



} // namespace SCIRun
