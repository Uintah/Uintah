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
#include <algorithm>

namespace SCIRun {

using std::for_each;
using std::cerr;
using std::endl;
using std::copy;

Persistent* make_TetVolMesh() {
  return scinew TetVolMesh;
}

PersistentTypeID TetVolMesh::type_id("TetVolMesh", "Mesh",
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
  node_nbor_lock_("TetVolMesh node_neighbors_ fill lock"),
  grid_(0),
  grid_lock_("TetVolMesh grid_ fill lock")
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
  node_nbor_lock_("TetVolMesh node_neighbors_ fill lock"),
  grid_(copy.grid_),
  grid_lock_("TetVolMesh grid_ fill lock")
{
}

TetVolMesh::~TetVolMesh()
{
}

/* To generate a random point inside of a tetrahedron, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void TetVolMesh::get_random_point(Point &p, const Cell::index_type &ei,
				  int seed) const
{
  static MusilRNG rng;

  // get positions of the vertices
  Node::array_type ra;
  get_nodes(ra,ei);
  Point p0,p1,p2,p3;
  get_point(p0,ra[0]);
  get_point(p1,ra[1]);
  get_point(p2,ra[2]);
  get_point(p3,ra[3]);

  // generate barrycentric coordinates
  double t,u,v,w;
  if (seed) {
    MusilRNG rng1(seed);
    t = rng1();
    u = rng1();
    v = rng1();
    w = rng1();
  } else {
    t = rng();
    u = rng();
    v = rng();
    w = rng();
  }
  double sum = t+u+v+w;
  t/=sum;
  u/=sum;
  v/=sum;
  w/=sum;

  // compute the position of the random point
  p = (p0.vector()*t+p1.vector()*u+p2.vector()*v+p3.vector()*w).point();
}

BBox
TetVolMesh::get_bounding_box() const
{
  BBox result;

  Node::iterator ni, nie;
  begin(ni);
  end(nie);
  while (ni != nie)
  {
    Point p;
    get_point(p, *ni);
    result.extend(p);
    ++ni;
  }
  return result;
}


void
TetVolMesh::transform(Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  
  // Recompute grid.
  grid_=0;
  compute_grid();
}


void
TetVolMesh::hash_face(Node::index_type n1, Node::index_type n2,
		      Node::index_type n3,
		      Cell::index_type ci, face_ht &table) const {
  PFace f(n1, n2, n3);

  face_ht::iterator iter = table.find(f);
  if (iter == table.end()) {
    f.cells_[0] = ci;
    table[f] = 0; // insert for the first time
  } else {
    PFace f = (*iter).first;
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
  face_table_lock_.lock();
  if (faces_.size() > 0) {face_table_lock_.unlock(); return;}
  cerr << "TetVolMesh::computing faces...\n";

  Cell::iterator ci, cie;
  begin(ci); end(cie);
  Node::array_type arr(4);
  while (ci != cie)
  {
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
  vector<PFace>::iterator f_iter = faces_.begin();
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
TetVolMesh::hash_edge(Node::index_type n1, Node::index_type n2,
		      Cell::index_type ci, edge_ht &table) const {
  PEdge e(n1, n2);
  edge_ht::iterator iter = table.find(e);
  if (iter == table.end()) {
    table[e] = 0; // insert for the first time
  } else {
    PEdge e = (*iter).first;
    e.cells_.push_back(ci); // add this cell
    table.erase(iter);
    table[e] = 0;
  }
}

void
TetVolMesh::compute_edges()
{
  edge_table_lock_.lock();
  if (edges_.size() > 0) {edge_table_lock_.unlock(); return;}
  cerr << "TetVolMesh::computing edges...\n";

  Cell::iterator ci, cie;
  begin(ci); end(cie);
  Node::array_type arr(4);
  while (ci != cie)
  {
    get_nodes(arr, *ci);
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
  vector<PEdge>::iterator              e_iter = edges_.begin();
  edge_ht::iterator ht_iter = edge_table_.begin();
  while (ht_iter != edge_table_.end()) {
    *e_iter = (*ht_iter).first;
    (*ht_iter).second = e_iter - edges_.begin();
    ++e_iter; ++ht_iter;
  }
  edge_table_lock_.unlock();
}

void
TetVolMesh::flush_changes() {
  compute_edges();
  compute_faces();
  compute_node_neighbors();
  compute_grid();
}


void
TetVolMesh::begin(TetVolMesh::Node::iterator &itr) const
{
  itr = 0;
}

void
TetVolMesh::end(TetVolMesh::Node::iterator &itr) const
{
  itr = points_.size();
}

void
TetVolMesh::size(TetVolMesh::Node::size_type &s) const
{
  s = points_.size();
}

void
TetVolMesh::begin(TetVolMesh::Edge::iterator &itr) const
{
  itr = 0;
}

void
TetVolMesh::end(TetVolMesh::Edge::iterator &itr) const
{
  itr = edges_.size();
}

void
TetVolMesh::size(TetVolMesh::Edge::size_type &s) const
{
  s = edges_.size();
}

void
TetVolMesh::begin(TetVolMesh::Face::iterator &itr) const
{
  itr = 0;
}

void
TetVolMesh::end(TetVolMesh::Face::iterator &itr) const
{
  itr = faces_.size();
}

void
TetVolMesh::size(TetVolMesh::Face::size_type &s) const
{
  s = faces_.size();
}

void
TetVolMesh::begin(TetVolMesh::Cell::iterator &itr) const
{
  itr = 0;
}

void
TetVolMesh::end(TetVolMesh::Cell::iterator &itr) const
{
  itr = cells_.size() >> 2;
}

void
TetVolMesh::size(TetVolMesh::Cell::size_type &s) const
{
  s = cells_.size() >> 2;
}

void
TetVolMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.clear();
  PEdge e = edges_[idx];
  array.push_back(e.nodes_[0]);
  array.push_back(e.nodes_[1]);
}


void
TetVolMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.clear();
  PFace f = faces_[idx];
  array.push_back(f.nodes_[0]);
  array.push_back(f.nodes_[1]);
  array.push_back(f.nodes_[2]);
}


void
TetVolMesh::get_nodes(Node::array_type &array, Cell::index_type idx) const
{
  array.clear();
  array.push_back(cells_[idx * 4 + 0]);
  array.push_back(cells_[idx * 4 + 1]);
  array.push_back(cells_[idx * 4 + 2]);
  array.push_back(cells_[idx * 4 + 3]);
}


void
TetVolMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
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
TetVolMesh::get_edges(Edge::array_type &array, Cell::index_type idx) const
{
  array.clear();

  const int off = idx * 4;
  PEdge e0(cells_[off + 0], cells_[off + 1]);
  PEdge e1(cells_[off + 0], cells_[off + 2]);
  PEdge e2(cells_[off + 0], cells_[off + 3]);
  PEdge e3(cells_[off + 1], cells_[off + 2]);
  PEdge e4(cells_[off + 1], cells_[off + 3]);
  PEdge e5(cells_[off + 2], cells_[off + 3]);

  array.push_back((*(edge_table_.find(e0))).second);
  array.push_back((*(edge_table_.find(e1))).second);
  array.push_back((*(edge_table_.find(e2))).second);
  array.push_back((*(edge_table_.find(e3))).second);
  array.push_back((*(edge_table_.find(e4))).second);
  array.push_back((*(edge_table_.find(e5))).second);
}


void
TetVolMesh::get_faces(Face::array_type &array, Cell::index_type idx) const
{
  array.clear();

  const int off = idx * 4;
  PFace f0(cells_[off + 0], cells_[off + 1], cells_[off + 2]);
  PFace f1(cells_[off + 0], cells_[off + 1], cells_[off + 3]);
  PFace f2(cells_[off + 0], cells_[off + 2], cells_[off + 3]);
  PFace f3(cells_[off + 1], cells_[off + 2], cells_[off + 3]);

  // operator[] not const safe...
  array.push_back((*(face_table_.find(f0))).second);
  array.push_back((*(face_table_.find(f1))).second);
  array.push_back((*(face_table_.find(f2))).second);
  array.push_back((*(face_table_.find(f3))).second);
}

bool
TetVolMesh::get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
			 Face::index_type idx) const
{
  const PFace &f = faces_[idx];

  if (from == f.cells_[0]) {
    neighbor = f.cells_[1];
  } else {
    neighbor = f.cells_[0];
  }
  if (neighbor == -1) return false;
  return true;
}

void
TetVolMesh::get_neighbors(Cell::array_type &array, Cell::index_type idx) const
{
  Face::array_type faces;
  get_faces(faces, idx);
  array.clear();
  Face::array_type::iterator iter = faces.begin();
  while(iter != faces.end()) {
    Cell::index_type nbor;
    if (get_neighbor(nbor, idx, *iter)) {
      array.push_back(nbor);
    }
    ++iter;
  }
}

void
TetVolMesh::get_neighbors(Node::array_type &array, Node::index_type idx) const
{
  array.clear();
  array.insert(array.end(), node_neighbors_[idx].begin(),
	       node_neighbors_[idx].end());
}

void
TetVolMesh::compute_node_neighbors()
{
  node_nbor_lock_.lock();
  if (node_neighbors_.size() > 0) {node_nbor_lock_.unlock(); return;}
  cerr << "TetVolMesh::computing node neighbors...\n";
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  Edge::iterator ei, eie;
  begin(ei); end(eie);
  for_each(ei, eie, FillNodeNeighbors(node_neighbors_, *this));
  node_nbor_lock_.unlock();
}

void
TetVolMesh::get_center(Point &p, Node::index_type idx) const
{
  get_point(p, idx);
}

void
TetVolMesh::get_center(Point &p, Edge::index_type idx) const
{
  const double s = 1./2.;
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_point(p, arr[0]);
  get_point(p1, arr[1]);

  p = ((Vector(p) + Vector(p1)) * s).asPoint();
}

void
TetVolMesh::get_center(Point &p, Face::index_type idx) const
{
  const double s = 1./3.;
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p1, p2;
  get_point(p, arr[0]);
  get_point(p1, arr[1]);
  get_point(p2, arr[2]);

  p = ((Vector(p) + Vector(p1) + Vector(p2)) * s).asPoint();
}

void
TetVolMesh::get_center(Point &p, Cell::index_type idx) const
{
  const double s = .25L;
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p1, p2, p3;
  get_point(p, arr[0]);
  get_point(p1, arr[1]);
  get_point(p2, arr[2]);
  get_point(p3, arr[3]);


  p = ((Vector(p) + Vector(p1) + Vector(p2) +
	Vector(p3)) * s).asPoint();
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
TetVolMesh::locate(Node::index_type &loc, const Point &p)
{
  Cell::index_type ci;
  if (locate(ci, p)) { // first try the fast way.
    Node::array_type nodes;
    get_nodes(nodes, ci);

    double d0 = distance2(p, points_[nodes[0]]);
    double d = d0;
    loc = nodes[0];
    double d1 = distance2(p, points_[nodes[1]]);
    if (d1 < d) {
      d = d1;
      loc = nodes[1];
    }
    double d2 = distance2(p, points_[nodes[2]]);
    if (d2 < d) {
      d = d2;
      loc = nodes[2];
    }
    double d3 = distance2(p, points_[nodes[3]]);
    if (d3 < d)  {
       loc = nodes[3];
    }
    return true;
  }
  else
  {  // do exhaustive search.
    Node::iterator ni, nie;
    begin(ni);
    end(nie);
    if (ni == nie) { return false; }

    double min_dist = distance2(p, points_[*ni]);
    loc = *ni;
    ++ni;

    while (ni != nie)
    {
      const double dist = distance2(p, points_[*ni]);
      if (dist < min_dist)
      {
	loc = *ni;
      }
      ++ni;
    }
    return true;
  }
}


bool
TetVolMesh::locate(Edge::index_type &/*edge*/, const Point & /* p */)
{
  //FIX_ME
  ASSERTFAIL("TetVolMesh::locate(Edge::index_type &) not implemented!");
  //return false;
}


bool
TetVolMesh::locate(Face::index_type &/*face*/, const Point & /* p */)
{
  //FIX_ME
  ASSERTFAIL("TetVolMesh::locate(Face::index_type&) not implemented!");
  //return false;
}


bool
TetVolMesh::locate(Cell::index_type &cell, const Point &p)
{
  if (grid_.get_rep() == 0) {
    compute_grid();
    //ASSERTFAIL("Call compute_grid before calling locate!");
  }
  LatVolMeshHandle mesh = grid_->get_typed_mesh();
  LatVolMesh::Cell::index_type ci;
  if (!mesh->locate(ci, p)) { return false; }
  bool found_p = false;
  vector<Cell::index_type> v = grid_->value(ci);
  vector<Cell::index_type>::iterator iter = v.begin();
  while (iter != v.end()) {
    if (inside4_p((*iter) * 4, p)) {
      found_p = true;
      break;
    }
    ++iter;
  }

  if (found_p)
    cell = *iter;

  return found_p;
}


void
TetVolMesh::get_weights(const Point &p,
			Cell::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}

void
TetVolMesh::get_weights(const Point &p,
			Node::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    Node::array_type ra(4);
    get_nodes(ra,idx);
    Point p0,p1,p2,p3;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    get_point(p3,ra[3]);
    double vol0, vol1, vol2, vol3, vol_sum;
    vol0 = (Cross(Cross(p1-p,p2-p),p3-p)).length();
    vol1 = (Cross(Cross(p0-p,p2-p),p3-p)).length();
    vol2 = (Cross(Cross(p0-p,p1-p),p3-p)).length();
    vol3 = (Cross(Cross(p0-p,p1-p),p2-p)).length();
    vol_sum = vol0+vol1+vol2+vol3;
    l.push_back(ra[0]);
    l.push_back(ra[1]);
    l.push_back(ra[2]);
    l.push_back(ra[3]);
    w.push_back(vol0/vol_sum);
    w.push_back(vol1/vol_sum);
    w.push_back(vol2/vol_sum);
    w.push_back(vol3/vol_sum);
  }
}


void
TetVolMesh::compute_grid()
{
  grid_lock_.lock();
  if (grid_.get_rep() != 0) {grid_lock_.unlock(); return;} // only create once.

  cerr << "TetVolMesh::compute_grid starting" << endl;
  BBox bb = get_bounding_box();
  // cubed root of number of cells to get a subdivision ballpark
  const double one_third = 1.L/3.L;
  Cell::size_type csize;  size(csize);
  int s = (int)ceil(pow((double)csize , one_third));
  const double cell_epsilon = bb.diagonal().length() * 0.1 / s;

  LatVolMeshHandle mesh(scinew LatVolMesh(s, s, s, bb.min(), bb.max()));
  grid_ = scinew LatticeVol<vector<Cell::index_type> >(mesh, Field::CELL);
  grid_->resize_fdata();
  LatticeVol<vector<Cell::index_type> >::fdata_type &fd = grid_->fdata();

  BBox box;
  Node::array_type nodes;
  Cell::iterator ci, cie;
  begin(ci); end(cie);
  while(ci != cie)
  {
    get_nodes(nodes, *ci);

    box.reset();
    box.extend(points_[nodes[0]]);
    box.extend(points_[nodes[1]]);
    box.extend(points_[nodes[2]]);
    box.extend(points_[nodes[3]]);
    const Point padmin(box.min().x() - cell_epsilon,
		       box.min().y() - cell_epsilon,
		       box.min().z() - cell_epsilon);
    const Point padmax(box.max().x() + cell_epsilon,
		       box.max().y() + cell_epsilon,
		       box.max().z() + cell_epsilon);
    box.extend(padmin);
    box.extend(padmax);

    // add this cell index to all overlapping cells in grid_
    LatVolMesh::Cell::array_type carr;
    mesh->get_cells(carr, box);
    LatVolMesh::Cell::array_type::iterator giter = carr.begin();
    while (giter != carr.end()) {
      // Would like to just get a reference to the vector at the cell
      // but can't from value. Bypass the interface.
      vector<Cell::index_type> &v = fd[*giter];
      v.push_back(*ci);
      ++giter;
    }
    ++ci;
  }
  cerr << "TetVolMesh::compute_grid done." << endl << endl;
  grid_lock_.unlock();
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
TetVolMesh::get_gradient_basis(Cell::index_type ci, Vector& g0, Vector& g1,
			       Vector& g2, Vector& g3)
{
  Point& p1 = points_[cells_[ci * 4]];
  Point& p2 = points_[cells_[ci * 4+1]];
  Point& p3 = points_[cells_[ci * 4+2]];
  Point& p4 = points_[cells_[ci * 4+3]];

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

TetVolMesh::Node::index_type
TetVolMesh::add_find_point(const Point &p, double err)
{
  Node::index_type i;
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
TetVolMesh::add_tet(Node::index_type a, Node::index_type b, Node::index_type c, Node::index_type d)
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


TetVolMesh::Node::index_type
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


MeshHandle
TetVolMesh::clip(ClipperHandle clipper)
{
  TetVolMesh *clipped = scinew TetVolMesh();

  hash_map<under_type, under_type, hash<under_type>,
    equal_to<under_type> > nodemap;

  Elem::iterator bi, ei;
  begin(bi); end(ei);
  while (bi != ei)
  {
    Point p;
    get_center(p, *bi);
    if (clipper->inside_p(p))
    {
      // Add this element to the new mesh.
      Node::array_type onodes;
      get_nodes(onodes, *bi);
      Node::array_type nnodes(onodes.size());

      for (unsigned int i=0; i<onodes.size(); i++)
      {
	if (nodemap.find(onodes[i]) == nodemap.end())
	{
	  Point np;
	  get_center(np, onodes[i]);
	  nodemap[onodes[i]] = clipped->add_point(np);
	}
	nnodes[i] = nodemap[onodes[i]];
      }

      clipped->add_tet(nnodes[0], nnodes[1], nnodes[2], nnodes[3]);
    }
    
    ++bi;
  }

  clipped->flush_changes();  // Really should copy normals
  return clipped;
}


#define TETVOLMESH_VERSION 1

void
TetVolMesh::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), TETVOLMESH_VERSION);
  Mesh::io(stream);

  SCIRun::Pio(stream, points_);
  SCIRun::Pio(stream, cells_);
  SCIRun::Pio(stream, neighbors_);

  stream.end_class();

  if (stream.reading())
  {
    flush_changes();
  }
}

const TypeDescription*
TetVolMesh::get_type_description() const
{
  return SCIRun::get_type_description((TetVolMesh *)0);
}


const TypeDescription*
get_type_description(TetVolMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TetVolMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(TetVolMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TetVolMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(TetVolMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TetVolMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(TetVolMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TetVolMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(TetVolMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TetVolMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


} // namespace SCIRun
