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
#include <iostream>

namespace SCIRun {

using std::for_each;
using std::cerr;
using std::endl;

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


TetVolMesh::TetVolMesh() :
  points_(0),
  points_lock_("TetVolMesh points_ fill lock"),
  cells_(0),
  cells_lock_("TetVolMesh cells_ fill lock"),
  neighbors_(0),
  nbors_lock_("TetVolMesh neighbors_ fill lock"),
  faces_(0),
  face_table_(0),
  face_table_lock_("TetVolMesh faces_ fill lock"),
  edges_(0),
  edge_table_(0),
  edge_table_lock_("TetVolMesh edge_ fill lock"),
  node_nbor_lock_("TetVolMesh node_neighbors__ fill lock")
{
}

TetVolMesh::TetVolMesh(const TetVolMesh &copy): 
  points_(copy.points_),
  points_lock_("TetVolMesh points_ fill lock"),
  cells_(copy.cells_),
  cells_lock_("TetVolMesh cells_ fill lock"),
  neighbors_(copy.neighbors_),
  nbors_lock_("TetVolMesh neighbors_ fill lock"),
  faces_(copy.faces_),
  face_table_(copy.face_table_),
  face_table_lock_("TetVolMesh faces_ fill lock"),
  edges_(copy.edges_),
  edge_table_(copy.edge_table_),
  edge_table_lock_("TetVolMesh edge_ fill lock"),
  node_nbor_lock_("TetVolMesh node_neighbors__ fill lock")
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
		      cell_index ci, face_ht &table) const {
  Face f(n1, n2, n3);
  
  face_ht::iterator iter = table.find(f);
  if (iter == table.end()) {
    f.cells_[0] = ci;
    table[f] = 0; // insert for the first time
  } else {
    Face f = (*iter).first;
    if (f.cells_[1] != -1) {
      cerr << "This Mesh has problems." << endl;
      return;
    }
    f.cells_[1] = ci; // add this cell
    table.erase(iter);
    table[f] = 0;
  }
}

void 
TetVolMesh::compute_faces()
{
  if (faces_.size() > 0) return;

  face_table_lock_.lock();
  cell_iterator ci = cell_begin();
  while (ci != cell_end()) {
    node_array arr;
    get_nodes(arr, *ci); 
    // 4 faces
    hash_face(arr[0], arr[1], arr[2], *ci, face_table_);
    hash_face(arr[0], arr[1], arr[3], *ci, face_table_);
    hash_face(arr[0], arr[2], arr[3], *ci, face_table_);
    hash_face(arr[1], arr[2], arr[3], *ci, face_table_);
    ++ci;
  }
  // dump edges into the edges_ container.
  faces_.resize(face_table_.size());
  vector<Face>::iterator f_iter = faces_.begin();
  face_ht::iterator ht_iter = face_table_.begin();
  int i = 0;
  while (ht_iter != face_table_.end()) {
    *f_iter = (*ht_iter).first;
    (*ht_iter).second = i;
    ++f_iter; ++ht_iter; i++;
  }
  face_table_lock_.unlock();
}

void 
TetVolMesh::hash_edge(node_index n1, node_index n2, 
		      cell_index ci, edge_ht &table) const {
  Edge e(n1, n2);
  edge_ht::iterator iter = table.find(e);
  if (iter == table.end()) {
    table[e] = 0; // insert for the first time
  } else {
    Edge e = (*iter).first;
    e.cells_.push_back(ci); // add this cell
    table.erase(iter);
    table[e] = 0;
  }
}

void 
TetVolMesh::compute_edges()
{
  if (edges_.size() > 0) return;

  edge_table_lock_.lock();
  cell_iterator ci = cell_begin();
  while (ci != cell_end()) {
    node_array arr;
    get_nodes(arr, *ci); 
    // 6 edges
    hash_edge(arr[0], arr[1], *ci, edge_table_);
    hash_edge(arr[0], arr[2], *ci, edge_table_);
    hash_edge(arr[0], arr[3], *ci, edge_table_);
    hash_edge(arr[1], arr[2], *ci, edge_table_);
    hash_edge(arr[1], arr[3], *ci, edge_table_);
    hash_edge(arr[2], arr[3], *ci, edge_table_);
    ++ci;
  }
  // dump edges into the edges_ container.
  edges_.resize(edge_table_.size());
  vector<Edge>::iterator              e_iter = edges_.begin();
  edge_ht::iterator ht_iter = edge_table_.begin();
  while (ht_iter != edge_table_.end()) {
    *e_iter = (*ht_iter).first;
    (*ht_iter).second = e_iter - edges_.begin();
    ++e_iter; ++ht_iter;
  }
  edge_table_lock_.unlock();
}

void 
TetVolMesh::finish_mesh() {
  compute_edges();
  compute_faces();
  compute_node_neighbors();
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
  array.clear();
  Edge e = edges_[idx];
  array.push_back(e.nodes_[0]); 
  array.push_back(e.nodes_[1]);
}


void
TetVolMesh::get_nodes(node_array &array, face_index idx) const
{
  array.clear();
  Face f = faces_[idx];
  array.push_back(f.nodes_[0]); 
  array.push_back(f.nodes_[1]);
  array.push_back(f.nodes_[2]);
}


void
TetVolMesh::get_nodes(node_array &array, cell_index idx) const
{
  array.clear();
  array.push_back(cells_[idx * 4 + 0]);
  array.push_back(cells_[idx * 4 + 1]);
  array.push_back(cells_[idx * 4 + 2]);
  array.push_back(cells_[idx * 4 + 3]);
}


void
TetVolMesh::get_edges(edge_array &array, face_index idx) const
{
  array.clear();
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
TetVolMesh::get_edges(edge_array &array, cell_index idx) const
{
  array.clear();

  const int off = idx * 4;
  Edge e0(cells_[off + 0], cells_[off + 1]);
  Edge e1(cells_[off + 0], cells_[off + 2]);
  Edge e2(cells_[off + 0], cells_[off + 3]);
  Edge e3(cells_[off + 1], cells_[off + 2]);
  Edge e4(cells_[off + 1], cells_[off + 3]);
  Edge e5(cells_[off + 2], cells_[off + 3]);

  array.push_back((*(edge_table_.find(e0))).second);
  array.push_back((*(edge_table_.find(e1))).second);
  array.push_back((*(edge_table_.find(e2))).second);
  array.push_back((*(edge_table_.find(e3))).second);
  array.push_back((*(edge_table_.find(e4))).second);
  array.push_back((*(edge_table_.find(e5))).second);
}


void
TetVolMesh::get_faces(face_array &array, cell_index idx) const
{
  array.clear();

  const int off = idx * 4;
  Face f0(cells_[off + 0], cells_[off + 1], cells_[off + 2]);
  Face f1(cells_[off + 0], cells_[off + 1], cells_[off + 3]);
  Face f2(cells_[off + 0], cells_[off + 2], cells_[off + 3]);
  Face f3(cells_[off + 1], cells_[off + 2], cells_[off + 3]);
  
  // operator[] not const safe...
  array.push_back((*(face_table_.find(f0))).second);
  array.push_back((*(face_table_.find(f1))).second);
  array.push_back((*(face_table_.find(f2))).second);
  array.push_back((*(face_table_.find(f3))).second);
}

bool
TetVolMesh::get_neighbor(cell_index &neighbor, cell_index from,
			 face_index idx) const
{
  const Face &f = faces_[idx];

  if (from == f.cells_[0]) {
    neighbor = f.cells_[1];
  } else { 
    neighbor = f.cells_[0];
  }
  if (neighbor == -1) return false;
  return true;
}

void 
TetVolMesh::get_neighbors(node_array &array, node_index idx) const
{
  array.clear();
  array.insert(array.end(), node_neighbors_[idx].begin(), 
	       node_neighbors_[idx].end());
}

void 
TetVolMesh::compute_node_neighbors()
{
  if (node_neighbors_.size() > 0) return;
  node_nbor_lock_.lock();
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  for_each(edge_begin(), edge_end(), FillNodeNeighbors(node_neighbors_, 
						       *this));
  node_nbor_lock_.unlock();
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
    ++ci;
  }
  cell = *ci;
  return found_p;
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

//! return the volume of the tet.
double 
TetVolMesh::get_gradient_basis(cell_index ci, Vector& g0, Vector& g1, 
			       Vector& g2, Vector& g3)
{
  Point p1, p2, p3, p4;
  node_array nodes;
  get_nodes(nodes, ci);
  get_point(p1, nodes[0]);
  get_point(p2, nodes[1]);
  get_point(p3, nodes[2]);
  get_point(p4, nodes[3]);

  double x1=p1.x();
  double y1=p1.y();
  double z1=p1.z();
  double x2=p2.x();
  double y2=p2.y();
  double z2=p2.z();
  double x3=p3.x();
  double y3=p3.y();
  double z3=p3.z();
  double x4=p4.x();
  double y4=p4.y();
  double z4=p4.z();
  double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
  double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
  double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
  double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
  double iV6=1./(a1+a2+a3+a4);

  double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
  double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
  double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
  g0=Vector(b1*iV6, c1*iV6, d1*iV6);
  double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
  double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
  double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
  g1=Vector(b2*iV6, c2*iV6, d2*iV6);
  double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
  double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
  double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
  g2=Vector(b3*iV6, c3*iV6, d3*iV6);
  double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
  double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
  double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
  g3=Vector(b4*iV6, c4*iV6, d4*iV6);

  double vol=(1./iV6)/6.0;
  return(vol);
}

TetVolMesh::node_index
TetVolMesh::add_find_point(const Point &p, double err)
{
  node_index i;
  if (locate(i, p) && distance2(points_[i], p) < err)
  {
    return i;
  }
  else
  {
    points_.push_back(p);
    return points_.size() - 1;
  }
}


void
TetVolMesh::add_tet(node_index a, node_index b, node_index c, node_index d)
{
  cells_.push_back(a);
  cells_.push_back(b);
  cells_.push_back(c);
  cells_.push_back(d);
}


void
TetVolMesh::connect(double err)
{
  // Collapse point set by err.
  // TODO: average in stead of first found for new point?
  vector<Point> points(points_);
  vector<int> mapping(points_.size());
  vector<Point>::size_type i; 
  points_.clear();
  for (i = 0; i < points.size(); i++)
  {
    mapping[i] = add_find_point(points[i], err);
  }

  // Repair faces.
  for (i=0; i < cells_.size(); i++)
  {
    cells_[i] = mapping[i];
  }
  
  // TODO: Remove all degenerate cells here.

  // TODO: fix forward/backward facing problems.

  // TODO: Find neighbors
  vector<list<int> > edgemap(points_.size());
  for (i=0; i< cells_.size(); i++)
  {
    edgemap[cells_[i]].push_back(i);
  }

#if 0
  for (i=0; i<edgemap.size(); i++)
  {
    list<int>::iterator li1 = edgemap[i].begin();

    while (li1 != edgemap[i].end())
    {
      int e1 = *li1;
      li1++;

      list<int>::iterator li2 = li1;
      while (li2 != edgemap[i].end())
      {
	int e2 = *li2;
	li2++;
	
	if ( faces_[next(e1)] == faces_[prev(e2)])
	{
	  neighbors_[e1] = e2;
	  neighbors_[e2] = e1;
	}
      }
    }
  }
#endif

  // Remove unused points.
  // Reuse mapping array, edgemap array.
  vector<Point> dups(points_);
  points_.clear();
  
  for (i=0; i<dups.size(); i++)
  {
    if(edgemap[i].begin() != edgemap[i].end())
    {
      points_.push_back(dups[i]);
      mapping[i] = points_.size() - 1;
    }
  }

  // Repair faces.
  for (i=0; i < cells_.size(); i++)
  {
    cells_[i] = mapping[i];
  }
}


TetVolMesh::node_index
TetVolMesh::add_point(const Point &p)
{
  points_.push_back(p);
  return points_.size() - 1;
}


void
TetVolMesh::add_tet(const Point &p0, const Point &p1, const Point &p2,
		    const Point &p3)
{
  add_tet(add_find_point(p0), add_find_point(p1), add_find_point(p2),
	  add_find_point(p3));
}

void
TetVolMesh::add_tet_unconnected(const Point &p0,
				const Point &p1,
				const Point &p2,
				const Point &p3)
{
  add_tet(add_point(p0), add_point(p1), add_point(p2), add_point(p3));
}


#define TETVOLMESH_VERSION 1

void
TetVolMesh::io(Piostream &stream)
{
  stream.begin_class(type_id.type.c_str(), TETVOLMESH_VERSION);

  MeshBase::io(stream);

  Pio(stream, points_);
  Pio(stream, cells_);
  Pio(stream, neighbors_);

  stream.end_class();
}
} // namespace SCIRun
