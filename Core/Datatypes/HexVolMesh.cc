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
 *  HexVolMesh.cc: Hexrahedral mesh with new design.
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

#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <algorithm>
#include <sci_hash_map.h>


namespace SCIRun {

using std::for_each;
using std::cerr;
using std::endl;
using std::copy;

Persistent* make_HexVolMesh() {
  return scinew HexVolMesh;
}

PersistentTypeID HexVolMesh::type_id("HexVolMesh", "Mesh",
				     make_HexVolMesh);

const string
HexVolMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name("HexVolMesh");
  return name;
}


HexVolMesh::HexVolMesh() :
  points_(0),
  points_lock_("HexVolMesh points_ fill lock"),
  cells_(0),
  cells_lock_("HexVolMesh cells_ fill lock"),
  neighbors_(0),
  nbors_lock_("HexVolMesh neighbors_ fill lock"),
  faces_(0),
  face_table_(0),
  face_table_lock_("HexVolMesh faces_ fill lock"),
  edges_(0),
  edge_table_(0),
  edge_table_lock_("HexVolMesh edge_ fill lock"),
  node_nbor_lock_("HexVolMesh node_neighbors_ fill lock"),
  grid_(0),
  grid_lock_("HexVolMesh grid_ fill lock")
{
}

HexVolMesh::HexVolMesh(const HexVolMesh &copy):
  points_(copy.points_),
  points_lock_("HexVolMesh points_ fill lock"),
  cells_(copy.cells_),
  cells_lock_("HexVolMesh cells_ fill lock"),
  neighbors_(copy.neighbors_),
  nbors_lock_("HexVolMesh neighbors_ fill lock"),
  faces_(copy.faces_),
  face_table_(copy.face_table_),
  face_table_lock_("HexVolMesh faces_ fill lock"),
  edges_(copy.edges_),
  edge_table_(copy.edge_table_),
  edge_table_lock_("HexVolMesh edge_ fill lock"),
  node_nbor_lock_("HexVolMesh node_neighbors_ fill lock"),
  grid_(copy.grid_),
  grid_lock_("HexVolMesh grid_ fill lock")
{
}

HexVolMesh::~HexVolMesh()
{
}

/* To generate a random point inside of a hexrahedron, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void HexVolMesh::get_random_point(Point &/*p*/, const Cell::index_type &/*ei*/,
				  int /*seed*/) const
{
  ASSERTFAIL("don't know how to pick a random point in a hex");
}

BBox
HexVolMesh::get_bounding_box() const
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
HexVolMesh::transform(Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  
  // Recompute grid.
  grid_.detach();
  compute_grid();
}


void
HexVolMesh::hash_face(Node::index_type n1, Node::index_type n2,
		      Node::index_type n3, Node::index_type n4,
		      Cell::index_type ci, face_ht &table) const {
  PFace f(n1, n2, n3, n4);

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
HexVolMesh::compute_faces()
{
  face_table_lock_.lock();
  if (faces_.size() > 0) {face_table_lock_.unlock(); return;}
  cerr << "HexVolMesh::computing faces...\n";

  Cell::iterator ci, cie;
  begin(ci); end(cie);
  Node::array_type arr(8);
  while (ci != cie)
  {
    get_nodes(arr, *ci);
    // 6 faces -- each is entered CCW from outside looking in
    hash_face(arr[0], arr[1], arr[2], arr[3], *ci, face_table_);
    hash_face(arr[7], arr[6], arr[5], arr[4], *ci, face_table_);
    hash_face(arr[0], arr[4], arr[5], arr[1], *ci, face_table_);
    hash_face(arr[2], arr[6], arr[7], arr[3], *ci, face_table_);
    hash_face(arr[3], arr[7], arr[4], arr[0], *ci, face_table_);
    hash_face(arr[1], arr[5], arr[6], arr[2], *ci, face_table_);
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
HexVolMesh::hash_edge(Node::index_type n1, Node::index_type n2,
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
HexVolMesh::compute_edges()
{
  edge_table_lock_.lock();
  if (edges_.size() > 0) {edge_table_lock_.unlock(); return;}
  cerr << "HexVolMesh::computing edges...\n";

  Cell::iterator ci, cie;
  begin(ci); end(cie);
  Node::array_type arr;
  while (ci != cie)
  {
    get_nodes(arr, *ci);
    hash_edge(arr[0], arr[1], *ci, edge_table_);
    hash_edge(arr[1], arr[2], *ci, edge_table_);
    hash_edge(arr[2], arr[3], *ci, edge_table_);
    hash_edge(arr[3], arr[0], *ci, edge_table_);

    hash_edge(arr[4], arr[5], *ci, edge_table_);
    hash_edge(arr[5], arr[6], *ci, edge_table_);
    hash_edge(arr[6], arr[7], *ci, edge_table_);
    hash_edge(arr[7], arr[4], *ci, edge_table_);

    hash_edge(arr[0], arr[4], *ci, edge_table_);
    hash_edge(arr[5], arr[1], *ci, edge_table_);

    hash_edge(arr[2], arr[6], *ci, edge_table_);
    hash_edge(arr[7], arr[3], *ci, edge_table_);
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
HexVolMesh::flush_changes()
{
  compute_edges();
  compute_faces();
//  compute_node_neighbors();
  compute_grid();
}


void
HexVolMesh::begin(HexVolMesh::Node::iterator &itr) const
{
  itr = 0;
}

void
HexVolMesh::end(HexVolMesh::Node::iterator &itr) const
{
  itr = points_.size();
}

void
HexVolMesh::size(HexVolMesh::Node::size_type &s) const
{
  s = points_.size();
}

void
HexVolMesh::begin(HexVolMesh::Edge::iterator &itr) const
{
  itr = 0;
}

void
HexVolMesh::end(HexVolMesh::Edge::iterator &itr) const
{
  itr = edges_.size();
}

void
HexVolMesh::size(HexVolMesh::Edge::size_type &s) const
{
  s = edges_.size();
}

void
HexVolMesh::begin(HexVolMesh::Face::iterator &itr) const
{
  itr = 0;
}

void
HexVolMesh::end(HexVolMesh::Face::iterator &itr) const
{
  itr = faces_.size();
}

void
HexVolMesh::size(HexVolMesh::Face::size_type &s) const
{
  s = faces_.size();
}

void
HexVolMesh::begin(HexVolMesh::Cell::iterator &itr) const
{
  itr = 0;
}

void
HexVolMesh::end(HexVolMesh::Cell::iterator &itr) const
{
  itr = cells_.size() >> 3;
}

void
HexVolMesh::size(HexVolMesh::Cell::size_type &s) const
{
  s = cells_.size() >> 3;
}

void
HexVolMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.clear();
  PEdge e = edges_[idx];
  array.push_back(e.nodes_[0]);
  array.push_back(e.nodes_[1]);
}


void
HexVolMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.clear();
  const PFace &f = faces_[idx];
  array.push_back(f.nodes_[0]);
  array.push_back(f.nodes_[1]);
  array.push_back(f.nodes_[2]);
  array.push_back(f.nodes_[3]);
}


void
HexVolMesh::get_nodes(Node::array_type &array, Cell::index_type idx) const
{
  array.clear();
  array.push_back(cells_[idx * 8 + 0]);
  array.push_back(cells_[idx * 8 + 1]);
  array.push_back(cells_[idx * 8 + 2]);
  array.push_back(cells_[idx * 8 + 3]);
  array.push_back(cells_[idx * 8 + 4]);
  array.push_back(cells_[idx * 8 + 5]);
  array.push_back(cells_[idx * 8 + 6]);
  array.push_back(cells_[idx * 8 + 7]);
}

void
HexVolMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
{
  array.clear();
  const PFace &f = faces_[idx];
  PEdge e0(f.nodes_[0], f.nodes_[1]);
  PEdge e1(f.nodes_[1], f.nodes_[2]);
  PEdge e2(f.nodes_[2], f.nodes_[3]);
  PEdge e3(f.nodes_[3], f.nodes_[0]);

  array.push_back((*(edge_table_.find(e0))).second);
  array.push_back((*(edge_table_.find(e1))).second);
  array.push_back((*(edge_table_.find(e2))).second);
  array.push_back((*(edge_table_.find(e3))).second);
}


void
HexVolMesh::get_edges(Edge::array_type &array, Cell::index_type idx) const
{
  array.clear();
  const int off = idx * 8;
  PEdge e00(cells_[off + 0], cells_[off + 1]);
  PEdge e01(cells_[off + 1], cells_[off + 2]);
  PEdge e02(cells_[off + 2], cells_[off + 3]);
  PEdge e03(cells_[off + 3], cells_[off + 0]);
  PEdge e04(cells_[off + 4], cells_[off + 5]);
  PEdge e05(cells_[off + 5], cells_[off + 6]);
  PEdge e06(cells_[off + 6], cells_[off + 7]);
  PEdge e07(cells_[off + 7], cells_[off + 4]);
  PEdge e08(cells_[off + 0], cells_[off + 4]);
  PEdge e09(cells_[off + 5], cells_[off + 1]);
  PEdge e10(cells_[off + 2], cells_[off + 6]);
  PEdge e11(cells_[off + 7], cells_[off + 3]);

  array.push_back((*(edge_table_.find(e00))).second);
  array.push_back((*(edge_table_.find(e01))).second);
  array.push_back((*(edge_table_.find(e02))).second);
  array.push_back((*(edge_table_.find(e03))).second);
  array.push_back((*(edge_table_.find(e04))).second);
  array.push_back((*(edge_table_.find(e05))).second);
  array.push_back((*(edge_table_.find(e06))).second);
  array.push_back((*(edge_table_.find(e07))).second);
  array.push_back((*(edge_table_.find(e08))).second);
  array.push_back((*(edge_table_.find(e09))).second);
  array.push_back((*(edge_table_.find(e10))).second);
  array.push_back((*(edge_table_.find(e11))).second);
}


void
HexVolMesh::get_faces(Face::array_type &array, Cell::index_type idx) const
{
  array.clear();

  const int off = idx * 8;
  PFace f0(cells_[off + 0], cells_[off + 1], cells_[off + 2], cells_[off + 3]);
  PFace f1(cells_[off + 4], cells_[off + 5], cells_[off + 6], cells_[off + 7]);
  PFace f2(cells_[off + 0], cells_[off + 4], cells_[off + 5], cells_[off + 1]);
  PFace f3(cells_[off + 2], cells_[off + 6], cells_[off + 7], cells_[off + 3]);
  PFace f4(cells_[off + 3], cells_[off + 7], cells_[off + 4], cells_[off + 0]);
  PFace f5(cells_[off + 1], cells_[off + 5], cells_[off + 6], cells_[off + 2]);

  // operator[] not const safe...
  array.push_back((*(face_table_.find(f0))).second);
  array.push_back((*(face_table_.find(f1))).second);
  array.push_back((*(face_table_.find(f2))).second);
  array.push_back((*(face_table_.find(f3))).second);
  array.push_back((*(face_table_.find(f4))).second);
  array.push_back((*(face_table_.find(f5))).second);
}

bool
HexVolMesh::get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
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
HexVolMesh::get_neighbors(Cell::array_type &array, Cell::index_type idx) const
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
HexVolMesh::get_neighbors(Node::array_type &array, Node::index_type idx) const
{
  array.clear();
  array.insert(array.end(), node_neighbors_[idx].begin(),
	       node_neighbors_[idx].end());
}

void
HexVolMesh::compute_node_neighbors()
{
  node_nbor_lock_.lock();
  if (node_neighbors_.size() > 0) {node_nbor_lock_.unlock(); return;}
  cerr << "HexVolMesh::computing node neighbors...\n";
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  Edge::iterator ei, eie;
  begin(ei); end(eie);
  for_each(ei, eie, FillNodeNeighbors(node_neighbors_, *this));
  node_nbor_lock_.unlock();
}

void
HexVolMesh::get_center(Point &p, Node::index_type idx) const
{
  get_point(p, idx);
}

void
HexVolMesh::get_center(Point &p, Edge::index_type idx) const
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
HexVolMesh::get_center(Point &p, Face::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  Node::array_type::iterator nai = nodes.begin();
  Vector v(0.0, 0.0, 0.0);
  while (nai != nodes.end())
  {
    Point pp;
    get_point(pp, *nai);
    v += pp.asVector();
    ++nai;
  }
  v *= 1.0 / nodes.size();
  p = v.asPoint();
}

void
HexVolMesh::get_center(Point &p, Cell::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  Node::array_type::iterator nai = nodes.begin();
  Vector v(0.0, 0.0, 0.0);
  while (nai != nodes.end())
  {
    Point pp;
    get_point(pp, *nai);
    v += pp.asVector();
    ++nai;
  }
  v *= 1.0 / nodes.size();
  p = v.asPoint();
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
HexVolMesh::locate(Node::index_type &loc, const Point &p)
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
HexVolMesh::locate(Edge::index_type &/*edge*/, const Point & /* p */)
{
  //FIX_ME
  ASSERTFAIL("HexVolMesh::locate(Edge::index_type &) not implemented!");
  //return false;
}


bool
HexVolMesh::locate(Face::index_type &/*face*/, const Point & /* p */)
{
  //FIX_ME
  ASSERTFAIL("HexVolMesh::locate(Face::index_type&) not implemented!");
  //return false;
}


bool
HexVolMesh::locate(Cell::index_type &cell, const Point &p)
{
  if (grid_.get_rep() == 0)
  {
    compute_grid();
  }
  LatVolMeshHandle mesh = grid_->get_typed_mesh();
  LatVolMesh::Cell::index_type ci;
  if (!mesh->locate(ci, p)) { return false; }
  vector<Cell::index_type> v = grid_->value(ci);
  vector<Cell::index_type>::iterator iter = v.begin();
  while (iter != v.end()) {
    if (inside8_p(*iter, p))
    {
      cell = *iter;
      return true;
    }
    ++iter;
  }
  return false;
}


void
HexVolMesh::get_weights(const Point &p,
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
HexVolMesh::get_weights(const Point &p,
			Node::array_type &nodes, vector<double> &w)
{
  Cell::index_type cell;
  if (locate(cell, p))
  {
    get_nodes(nodes, cell);
    Vector v[8];
    double vol[8];
    int i;
    for (i = 0; i < 8; i++)
    {
      Point np;
      get_point(np, nodes[i]);
      v[i] = np - p;
    }
    vol[0] =
      Cross(Cross(v[1], v[2]), v[6]).length() +
      Cross(Cross(v[1], v[3]), v[2]).length() +
      Cross(Cross(v[1], v[4]), v[5]).length() +
      Cross(Cross(v[1], v[6]), v[5]).length() +
      Cross(Cross(v[2], v[5]), v[6]).length() +
      Cross(Cross(v[3], v[4]), v[7]).length() +
      Cross(Cross(v[3], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[6]).length() +
      Cross(Cross(v[4], v[6]), v[7]).length();

    vol[1] =
      Cross(Cross(v[0], v[2]), v[3]).length() +
      Cross(Cross(v[0], v[3]), v[7]).length() +
      Cross(Cross(v[0], v[4]), v[5]).length() +
      Cross(Cross(v[0], v[4]), v[7]).length() +
      Cross(Cross(v[2], v[3]), v[7]).length() +
      Cross(Cross(v[2], v[5]), v[6]).length() +
      Cross(Cross(v[2], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[7]).length() +
      Cross(Cross(v[5], v[6]), v[7]).length();

    vol[2] =
      Cross(Cross(v[0], v[1]), v[3]).length() +
      Cross(Cross(v[0], v[1]), v[4]).length() +
      Cross(Cross(v[0], v[3]), v[4]).length() +
      Cross(Cross(v[1], v[4]), v[5]).length() +
      Cross(Cross(v[1], v[5]), v[6]).length() +
      Cross(Cross(v[3], v[4]), v[7]).length() +
      Cross(Cross(v[3], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[6]).length() +
      Cross(Cross(v[4], v[6]), v[7]).length();

    vol[3] =
      Cross(Cross(v[0], v[1]), v[2]).length() +
      Cross(Cross(v[0], v[1]), v[4]).length() +
      Cross(Cross(v[0], v[4]), v[7]).length() +
      Cross(Cross(v[1], v[2]), v[5]).length() +
      Cross(Cross(v[1], v[4]), v[5]).length() +
      Cross(Cross(v[2], v[5]), v[6]).length() +
      Cross(Cross(v[2], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[7]).length() +
      Cross(Cross(v[5], v[6]), v[7]).length();

    vol[4] =
      Cross(Cross(v[0], v[1]), v[2]).length() +
      Cross(Cross(v[0], v[1]), v[5]).length() +
      Cross(Cross(v[0], v[2]), v[3]).length() +
      Cross(Cross(v[0], v[3]), v[7]).length() +
      Cross(Cross(v[1], v[2]), v[5]).length() +
      Cross(Cross(v[2], v[3]), v[6]).length() +
      Cross(Cross(v[2], v[5]), v[6]).length() +
      Cross(Cross(v[3], v[6]), v[7]).length() +
      Cross(Cross(v[5], v[6]), v[7]).length();

    vol[5] =
      Cross(Cross(v[0], v[1]), v[3]).length() +
      Cross(Cross(v[0], v[1]), v[4]).length() +
      Cross(Cross(v[0], v[3]), v[4]).length() +
      Cross(Cross(v[1], v[2]), v[3]).length() +
      Cross(Cross(v[1], v[2]), v[6]).length() +
      Cross(Cross(v[2], v[3]), v[6]).length() +
      Cross(Cross(v[3], v[4]), v[7]).length() +
      Cross(Cross(v[3], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[6]), v[7]).length();

    vol[6] =
      Cross(Cross(v[0], v[1]), v[2]).length() +
      Cross(Cross(v[0], v[1]), v[5]).length() +
      Cross(Cross(v[0], v[2]), v[3]).length() +
      Cross(Cross(v[0], v[3]), v[4]).length() +
      Cross(Cross(v[0], v[4]), v[5]).length() +
      Cross(Cross(v[1], v[2]), v[5]).length() +
      Cross(Cross(v[2], v[3]), v[7]).length() +
      Cross(Cross(v[3], v[4]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[7]).length();

    vol[7] =
      Cross(Cross(v[0], v[1]), v[3]).length() +
      Cross(Cross(v[0], v[1]), v[4]).length() +
      Cross(Cross(v[0], v[3]), v[4]).length() +
      Cross(Cross(v[1], v[2]), v[3]).length() +
      Cross(Cross(v[1], v[2]), v[6]).length() +
      Cross(Cross(v[1], v[4]), v[5]).length() +
      Cross(Cross(v[1], v[5]), v[6]).length() +
      Cross(Cross(v[2], v[3]), v[6]).length() +
      Cross(Cross(v[4], v[5]), v[6]).length();

    const double suminv = 1.0 / (vol[0] + vol[1] + vol[2] + vol[3] +
				 vol[4] + vol[5] + vol[6] + vol[7]);

    for (i = 0; i < 8; i++)
    {
      w.push_back(vol[i] * suminv);
    }
  }
}


void
HexVolMesh::compute_grid()
{
  grid_lock_.lock();
  if (grid_.get_rep() != 0) {grid_lock_.unlock(); return;} // only create once.

  cerr << "HexVolMesh::compute_grid starting" << endl;
  BBox bb = get_bounding_box();
  if (!bb.valid()) { grid_lock_.unlock(); return; }

  // Cubed root of number of cells to get a subdivision ballpark.
  const double one_third = 1.L/3.L;
  Cell::size_type csize;  size(csize);
  const int s = (int)ceil(pow((double)csize , one_third));
  const double cell_epsilon = bb.diagonal().length() * 0.1 / s;

  LatVolMeshHandle mesh(scinew LatVolMesh(s, s, s, bb.min(), bb.max()));
  grid_ = scinew LatticeVol<vector<Cell::index_type> >(mesh, Field::CELL);
  LatticeVol<vector<Cell::index_type> >::fdata_type &fd = grid_->fdata();

  BBox box;
  Node::array_type nodes;
  Cell::iterator ci, cie;
  begin(ci); end(cie);
  while(ci != cie)
  {
    get_nodes(nodes, *ci);

    box.reset();
    for (unsigned int i = 0; i < nodes.size(); i++)
    {
      box.extend(points_[nodes[i]]);
    }
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
  cerr << "HexVolMesh::compute_grid done." << endl << endl;
  grid_lock_.unlock();
}



bool
HexVolMesh::inside8_p(Cell::index_type i, const Point &p) const
{
  Face::array_type faces;
  get_faces(faces, i);

  Point center;
  get_center(center, i);

  for (unsigned int i = 0; i < faces.size(); i++)
  {
    Node::array_type nodes;
    get_nodes(nodes, faces[i]);
    Point p0, p1, p2;
    get_center(p0, nodes[0]);
    get_center(p1, nodes[1]);
    get_center(p2, nodes[2]);

    const Vector v0(p1 - p0), v1(p2 - p0);
    const Vector normal = Cross(v0, v1);
    const Vector off0(p - p0);
    const Vector off1(center - p0);
    if (Dot(off0, normal) * Dot(off1, normal) < 0)
    {
      return false;
    }
  }
  return true;
}
    


//! return the volume of the hex.
double
HexVolMesh::get_gradient_basis(Cell::index_type /*ci*/, 
			       Vector& /*g0*/, Vector& /*g1*/,
			       Vector& /*g2*/, Vector& /*g3*/,
			       Vector& /*g4*/, Vector& /*g5*/,
			       Vector& /*g6*/, Vector& /*g7*/)
{
  ASSERTFAIL("get_gradient_basis not implemented for hexes");
  return 1;
}

HexVolMesh::Node::index_type
HexVolMesh::add_find_point(const Point &p, double err)
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
HexVolMesh::add_hex(Node::index_type a, Node::index_type b, 
		    Node::index_type c, Node::index_type d,
		    Node::index_type e, Node::index_type f,
		    Node::index_type g, Node::index_type h)
{
  cells_.push_back(a);
  cells_.push_back(b);
  cells_.push_back(c);
  cells_.push_back(d);
  cells_.push_back(e);
  cells_.push_back(f);
  cells_.push_back(g);
  cells_.push_back(h);
}


void
HexVolMesh::connect(double err)
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


HexVolMesh::Node::index_type
HexVolMesh::add_point(const Point &p)
{
  points_.push_back(p);
  return points_.size() - 1;
}


void
HexVolMesh::add_hex(const Point &p0, const Point &p1, 
		    const Point &p2, const Point &p3,
		    const Point &p4, const Point &p5,
		    const Point &p6, const Point &p7)
{
  add_hex(add_find_point(p0), add_find_point(p1),
	  add_find_point(p2), add_find_point(p3),
	  add_find_point(p4), add_find_point(p5),
	  add_find_point(p6), add_find_point(p7));
}

void
HexVolMesh::add_hex_unconnected(const Point &p0,
				const Point &p1,
				const Point &p2,
				const Point &p3,
				const Point &p4,
				const Point &p5,
				const Point &p6,
				const Point &p7)
{
  add_hex(add_point(p0), add_point(p1), 
	  add_point(p2), add_point(p3),
	  add_point(p4), add_point(p5),
	  add_point(p6), add_point(p7));
}


HexVolMesh::Elem::index_type
HexVolMesh::add_elem(Node::array_type a)
{
  cells_.push_back(a[0]);
  cells_.push_back(a[1]);
  cells_.push_back(a[2]);
  cells_.push_back(a[3]);
  cells_.push_back(a[4]);
  cells_.push_back(a[5]);
  cells_.push_back(a[6]);
  cells_.push_back(a[7]);
  return (cells_.size() - 1) >> 3;
}


#define HEXVOLMESH_VERSION 1

void
HexVolMesh::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), HEXVOLMESH_VERSION);
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
HexVolMesh::get_type_description() const
{
  return SCIRun::get_type_description((HexVolMesh *)0);
}


const TypeDescription*
get_type_description(HexVolMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("HexVolMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(HexVolMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("HexVolMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(HexVolMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("HexVolMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(HexVolMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("HexVolMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(HexVolMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("HexVolMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


} // namespace SCIRun
