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
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Plane.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <algorithm>
#include <set>
#include <sci_hash_map.h>
#include <Core/Exceptions/InternalError.h>

namespace SCIRun {

using std::for_each;
using std::cerr;
using std::endl;
using std::copy;

const size_t HexVolMesh::FaceHash::bucket_size = 4;
const size_t HexVolMesh::FaceHash::min_buckets = 8;

const int HexVolMesh::FaceHash::sz_quarter_int = (int)(HexVolMesh::sz_int * .25); // in bits
const int HexVolMesh::FaceHash::top4_mask = ((~((int)0)) << sz_quarter_int << sz_quarter_int << sz_quarter_int);
const int HexVolMesh::FaceHash::up4_mask = top4_mask ^ (~((int)0) << sz_quarter_int << sz_quarter_int);
const int HexVolMesh::FaceHash::mid4_mask =  top4_mask ^ (~((int)0) << sz_quarter_int);
const int HexVolMesh::FaceHash::low4_mask = ~(top4_mask | mid4_mask);

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
  faces_(0),
  face_table_(),
  face_table_lock_("HexVolMesh faces_ fill lock"),
  edges_(0),
  edge_table_(),
  edge_table_lock_("HexVolMesh edge_ fill lock"),
  node_nbor_lock_("HexVolMesh node_neighbors_ fill lock"),
  grid_(0),
  grid_lock_("HexVolMesh grid_ fill lock"),
  synchronized_(NODES_E | CELLS_E)
{
}

HexVolMesh::HexVolMesh(const HexVolMesh &copy):
  points_(copy.points_),
  points_lock_("HexVolMesh points_ fill lock"),
  cells_(copy.cells_),
  cells_lock_("HexVolMesh cells_ fill lock"),
  faces_(copy.faces_),
  face_table_(copy.face_table_),
  face_table_lock_("HexVolMesh faces_ fill lock"),
  edges_(copy.edges_),
  edge_table_(copy.edge_table_),
  edge_table_lock_("HexVolMesh edge_ fill lock"),
  node_nbor_lock_("HexVolMesh node_neighbors_ fill lock"),
  grid_(copy.grid_),
  grid_lock_("HexVolMesh grid_ fill lock"),
  synchronized_(copy.synchronized_)
{
}

HexVolMesh::~HexVolMesh()
{
}

/* To generate a random point inside of a hexrahedron, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void HexVolMesh::get_random_point(Point &/*p*/, Cell::index_type /*ei*/,
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
HexVolMesh::transform(const Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  synchronized_ &= ~LOCATE_E;
  grid_ = 0;
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
      cerr << "This Mesh has problems: Cells #" 
	   << f.cells_[0] << ", #" << f.cells_[1] << ", and #" << ci 
	   << " are illegally adjacent." << endl; 
      SCI_THROW(InternalError("Corrupt HexVolMesh"));      
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
  if (synchronized_ & FACES_E) {
    face_table_lock_.unlock();
    return;
  }
  face_table_.clear();

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

  synchronized_ |= FACES_E;
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
  if (synchronized_ & EDGES_E) {
    edge_table_lock_.unlock();
    return;
  }
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
    (*ht_iter).second = static_cast<Edge::index_type>(e_iter - edges_.begin());
    ++e_iter; ++ht_iter;
  }
  
  synchronized_ |= EDGES_E;
  edge_table_lock_.unlock();
}




bool
HexVolMesh::synchronize(unsigned int tosync)
{
  if (tosync & EDGES_E && !(synchronized_ & EDGES_E)) compute_edges();
  if (tosync & FACES_E && !(synchronized_ & FACES_E)) compute_faces();
  if (tosync & LOCATE_E && !(synchronized_ & LOCATE_E)) {
    compute_grid();
    if (!(synchronized_ & FACES_E)) compute_faces();
  }
  if (tosync & NODE_NEIGHBORS_E && !(synchronized_ & NODE_NEIGHBORS_E)) 
    compute_node_neighbors();
  return true;
}


void
HexVolMesh::begin(HexVolMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on HexVolMesh first");
  itr = 0;
}

void
HexVolMesh::end(HexVolMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on HexVolMesh first");
  itr = static_cast<Node::iterator>(points_.size());
}

void
HexVolMesh::size(HexVolMesh::Node::size_type &s) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on HexVolMesh first");
  s = static_cast<Node::size_type>(points_.size());
}

void
HexVolMesh::begin(HexVolMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  itr = 0;
}

void
HexVolMesh::end(HexVolMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  itr = static_cast<Edge::iterator>(edges_.size());
}

void
HexVolMesh::size(HexVolMesh::Edge::size_type &s) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  s = static_cast<Edge::size_type>(edges_.size());
}

void
HexVolMesh::begin(HexVolMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  itr = 0;
}

void
HexVolMesh::end(HexVolMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  itr = static_cast<Face::iterator>(faces_.size());
}

void
HexVolMesh::size(HexVolMesh::Face::size_type &s) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  s = static_cast<Face::size_type>(faces_.size());
}

void
HexVolMesh::begin(HexVolMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on HexVolMesh first");
  itr = 0;
}

void
HexVolMesh::end(HexVolMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on HexVolMesh first");
  itr = static_cast<Cell::iterator>(cells_.size() >> 3);
}

void
HexVolMesh::size(HexVolMesh::Cell::size_type &s) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on HexVolMesh first");
  s = static_cast<Cell::size_type>(cells_.size() >> 3);
}

void
HexVolMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
  array.clear();
  PEdge e = edges_[idx];
  array.push_back(e.nodes_[0]);
  array.push_back(e.nodes_[1]);
}


void
HexVolMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
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
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
  array.clear();
  const PFace &f = faces_[idx];
  PEdge e0(f.nodes_[0], f.nodes_[1]);
  PEdge e1(f.nodes_[1], f.nodes_[2]);
  PEdge e2(f.nodes_[2], f.nodes_[3]);
  PEdge e3(f.nodes_[3], f.nodes_[0]);

  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
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

  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on HexVolMesh first");
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
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
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
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on HexVolMesh first");
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
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E,
	    "Must call synchronize NODE_NEIGHBORS_E on HexVolMesh first");
  array.clear();
  array.insert(array.end(), node_neighbors_[idx].begin(),
	       node_neighbors_[idx].end());
}

void
HexVolMesh::compute_node_neighbors()
{
  if (!(synchronized_ & EDGES_E)) synchronize(EDGES_E);
  node_nbor_lock_.lock();
  if (synchronized_ & NODE_NEIGHBORS_E) {
    node_nbor_lock_.unlock();
    return;
  }
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  Edge::iterator ei, eie;
  begin(ei); end(eie);
  for_each(ei, eie, FillNodeNeighbors(node_neighbors_, *this));
  synchronized_ |= NODE_NEIGHBORS_E;
  node_nbor_lock_.unlock();
}

void
HexVolMesh::get_center(Point &p, Node::index_type idx) const
{
  get_point(p, idx);
}


void
HexVolMesh::get_center(Point &result, Edge::index_type idx) const
{
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);
  
  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


void
HexVolMesh::get_center(Point &p, Face::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  Node::array_type::iterator nai = nodes.begin();
  get_point(p, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 4.0);
}

void
HexVolMesh::get_center(Point &p, Cell::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 8);
  Node::array_type::iterator nai = nodes.begin();
  get_point(p, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 8.0);
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
    
    double dmin = distance2(p, points_[nodes[0]]);
    loc = nodes[0];
    for (unsigned int i = 1; i < nodes.size(); i++)
    {
      const double d = distance2(p, points_[nodes[i]]);
      if (d < dmin)
      {
	dmin = d;
	loc = nodes[i];
      }
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
	min_dist = dist;
      }
      ++ni;
    }
    return true;
  }
}


bool
HexVolMesh::locate(Edge::index_type &edge, const Point &p)
{
  Cell::index_type cell;
  if (locate(cell, p))
  {
    Edge::array_type edges;
    get_edges(edges, cell);

    if (edges.size() == 0) { return false; }

    edge = edges[0];
    Point loc;
    get_center(loc, edges[0]);
    double mindist = distance2(p, loc);
    for (unsigned int i = 0; i < edges.size(); i++)
    {
      get_center(loc, edges[i]);
      const double dist = distance2(p, loc);
      if (dist < mindist)
      {
	edge = edges[i];
	mindist = dist;
      }
    }
    return true;
  }
  return false;
}


bool
HexVolMesh::locate(Face::index_type &face, const Point &p)
{
  Cell::index_type cell;
  if (locate(cell, p))
  {
    Face::array_type faces;
    get_faces(faces, cell);

    if (faces.size() == 0) { return false; }

    face = faces[0];
    Point loc;
    get_center(loc, faces[0]);
    double mindist = distance2(p, loc);
    for (unsigned int i = 0; i < faces.size(); i++)
    {
      get_center(loc, faces[i]);
      const double dist = distance2(p, loc);
      if (dist < mindist)
      {
	face = faces[i];
	mindist = dist;
      }
    }
    return true;
  }
  return false;
}


bool
HexVolMesh::locate(Cell::index_type &cell, const Point &p)
{
  // Check last cell found first.  Copy cache to cell first so that we
  // don't care about thread safeness, such that worst case on
  // context switch is that cache is not found.
  static Cell::index_type cache(0);
  cell = cache;
  if (cell > Cell::index_type(0) &&
      cell < Cell::index_type(cells_.size()/8) &&
      inside8_p(cell, p))
  {
    return true;
  }

  if ( (!synchronized_) & LOCATE_E) // I hope I got the () right.
    synchronize(LOCATE_E);
  ASSERT(grid_.get_rep());

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


// The volume x 6, used by get_weights to compute barycentric coordinates.
static double
tet_vol6(const Point &p1, const Point &p2, const Point &p3, const Point &p4)
{
  const double x1=p1.x();
  const double y1=p1.y();
  const double z1=p1.z();
  const double x2=p2.x();
  const double y2=p2.y();
  const double z2=p2.z();
  const double x3=p3.x();
  const double y3=p3.y();
  const double z3=p3.z();
  const double x4=p4.x();
  const double y4=p4.y();
  const double z4=p4.z();
  const double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
  const double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
  const double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
  const double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
  return fabs(a1+a2+a3+a4);
}

#if 0
// Tet inside test, cut and pasted from TetVolMesh.cc
static bool
tet_inside_p(const Point &p, const Point &p0, const Point &p1,
	  const Point &p2, const Point &p3)
{
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
  if (s0 < -1.e-12)
    return false;

  const double b1 = + (y3*z0-y0*z3) + (y0*z2-y2*z0) + (y2*z3-y3*z2);
  const double c1 = - (x3*z0-x0*z3) - (x0*z2-x2*z0) - (x2*z3-x3*z2);
  const double d1 = + (x3*y0-x0*y3) + (x0*y2-x2*y0) + (x2*y3-x3*y2);
  const double s1 = iV6 * (a1 + b1*p.x() + c1*p.y() + d1*p.z());
  if (s1 < -1.e-12)
    return false;

  const double b2 = - (y0*z1-y1*z0) - (y1*z3-y3*z1) - (y3*z0-y0*z3);
  const double c2 = + (x0*z1-x1*z0) + (x1*z3-x3*z1) + (x3*z0-x0*z3);
  const double d2 = - (x0*y1-x1*y0) - (x1*y3-x3*y1) - (x3*y0-x0*y3);
  const double s2 = iV6 * (a2 + b2*p.x() + c2*p.y() + d2*p.z());
  if (s2 < -1.e-12)
    return false;

  const double b3 = +(y1*z2-y2*z1) + (y2*z0-y0*z2) + (y0*z1-y1*z0);
  const double c3 = -(x1*z2-x2*z1) - (x2*z0-x0*z2) - (x0*z1-x1*z0);
  const double d3 = +(x1*y2-x2*y1) + (x2*y0-x0*y2) + (x0*y1-x1*y0);
  const double s3 = iV6 * (a3 + b3*p.x() + c3*p.y() + d3*p.z());
  if (s3 < -1.e-12)
    return false;

  return true;
}


static void
tetinterp(const Point &p, Point nodes[8],
	  vector<double> &w, int a, int b, int c, int d)
{
  int i;
  w.resize(8);
  for (i=0; i < 8; i++)
  {
    w[i] = 0.0;
  }
  
  const double wa = tet_vol6(p, nodes[b], nodes[c], nodes[d]); 
  const double wb = tet_vol6(p, nodes[a], nodes[c], nodes[d]); 
  const double wc = tet_vol6(p, nodes[a], nodes[b], nodes[d]); 
  const double wd = tet_vol6(p, nodes[a], nodes[b], nodes[c]); 

  const double sum = 1.0 / (wa + wb + wc + wd);
  
  w[a] = wa * sum;
  w[b] = wb * sum;
  w[c] = wc * sum;
  w[d] = wd * sum;
}
#endif


//===================================================================

// area3D_Polygon(): computes the area of a 3D planar polygon
//    Input:  int n = the number of vertices in the polygon
//            Point* V = an array of n+2 vertices in a plane
//                       with V[n]=V[0] and V[n+1]=V[1]
//            Point N = unit normal vector of the polygon's plane
//    Return: the (float) area of the polygon

// Copyright 2000, softSurfer (www.softsurfer.com)
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

double
HexVolMesh::polygon_area(const Node::array_type &ni, const Vector N) const
{
    double area = 0;
    double an, ax, ay, az;  // abs value of normal and its coords
    int   coord;           // coord to ignore: 1=x, 2=y, 3=z
    unsigned int   i, j, k;         // loop indices
    const unsigned int n = ni.size();

    // select largest abs coordinate to ignore for projection
    ax = (N.x()>0 ? N.x() : -N.x());     // abs x-coord
    ay = (N.y()>0 ? N.y() : -N.y());     // abs y-coord
    az = (N.z()>0 ? N.z() : -N.z());     // abs z-coord

    coord = 3;                     // ignore z-coord
    if (ax > ay) {
        if (ax > az) coord = 1;    // ignore x-coord
    }
    else if (ay > az) coord = 2;   // ignore y-coord

    // compute area of the 2D projection
    for (i=1, j=2, k=0; i<=n; i++, j++, k++)
        switch (coord) {
        case 1:
            area += (points_[ni[i%n]].y() *
		     (points_[ni[j%n]].z() - points_[ni[k%n]].z()));
            continue;
        case 2:
            area += (points_[ni[i%n]].x() * 
		     (points_[ni[j%n]].z() - points_[ni[k%n]].z()));
            continue;
        case 3:
            area += (points_[ni[i%n]].x() * 
		     (points_[ni[j%n]].y() - points_[ni[k%n]].y()));
            continue;
        }

    // scale to get area before projection
    an = sqrt( ax*ax + ay*ay + az*az);  // length of normal vector
    switch (coord) {
    case 1:
        area *= (an / (2*ax));
        break;
    case 2:
        area *= (an / (2*ay));
        break;
    case 3:
        area *= (an / (2*az));
    }
    return area;
}

double
HexVolMesh::pyramid_volume(const Node::array_type &face, const Point &p) const
{
  Vector e1(points_[face[1]]-points_[face[0]]);
  Vector e2(points_[face[1]]-points_[face[2]]);
  if (Cross(e1,e2).length2()>0.0) {
    Plane plane(points_[face[0]], points_[face[1]], points_[face[2]]);
    //double dist = plane.eval_point(p);
    return fabs(plane.eval_point(p)*polygon_area(face,plane.normal())*0.25);
  }
  Vector e3(points_[face[3]]-points_[face[2]]);
  if (Cross(e2,e3).length2()>0.0) {
    Plane plane(points_[face[1]], points_[face[2]], points_[face[3]]);
    //double dist = plane.eval_point(p);
    return fabs(plane.eval_point(p)*polygon_area(face,plane.normal())*0.25);
  }
  return 0.0;
}


static unsigned int wtable[8][6] =
  {{1, 3, 4,  2, 7, 5},
   {2, 0, 5,  3, 4, 6},
   {3, 1, 6,  0, 5, 7},
   {0, 2, 7,  1, 6, 4},
   {5, 7, 0,  6, 3, 1},
   {6, 4, 1,  5, 0, 2},
   {7, 5, 2,  4, 1, 3},
   {4, 6, 3,  5, 2, 4}};


static double
tri_area(const Point &a, const Point &b, const Point &c)
{
  return Cross(b-a, c-a).length();
}


void
HexVolMesh::get_face_weights(vector<double> &w, const Node::array_type &nodes,
			     const Point &p, int i0, int i1, int i2, int i3)
{
  for (unsigned int j = 0; j < 8; j++)
  {
    w[j] = 0.0;
  }

  const Point &p0 = point(nodes[i0]);
  const Point &p1 = point(nodes[i1]);
  const Point &p2 = point(nodes[i2]);
  const Point &p3 = point(nodes[i3]);

  const double a0 = tri_area(p, p0, p1);
  if (a0 < 1.0e-6)
  {
    const Vector v0 = p0 - p1;
    const Vector v1 = p - p1;
    w[i0] = Dot(v0, v1) / Dot(v0, v0);
    w[i1] = 1.0 - w[i0];
    return;
  }
  const double a1 = tri_area(p, p1, p2);
  if (a1 < 1.0e-6)
  {
    const Vector v0 = p1 - p2;
    const Vector v1 = p - p2;
    w[i1] = Dot(v0, v1) / Dot(v0, v0);
    w[i2] = 1.0 - w[i1];
    return;
  }
  const double a2 = tri_area(p, p2, p3);
  if (a2 < 1.0e-6)
  {
    const Vector v0 = p2 - p3;
    const Vector v1 = p - p3;
    w[i2] = Dot(v0, v1) / Dot(v0, v0);
    w[i3] = 1.0 - w[i2];
    return;
  }
  const double a3 = tri_area(p, p3, p0);
  if (a3 < 1.0e-6)
  {
    const Vector v0 = p3 - p0;
    const Vector v1 = p - p0;
    w[i3] = Dot(v0, v1) / Dot(v0, v0);
    w[i0] = 1.0 - w[i3];
    return;
  }

  w[i0] = tri_area(p0, p1, p2) / (a0 * a3);
  w[i1] = tri_area(p1, p2, p0) / (a1 * a0);
  w[i2] = tri_area(p2, p3, p1) / (a2 * a1);
  w[i3] = tri_area(p3, p0, p2) / (a3 * a2);

  const double suminv = 1.0 / (w[i0] + w[i1] + w[i2] + w[i3]);
  w[i0] *= suminv;
  w[i1] *= suminv;
  w[i2] *= suminv;
  w[i3] *= suminv;
}
  

void
HexVolMesh::get_weights(const Point &p,
			Node::array_type &nodes, vector<double> &w)
{
  synchronize (Mesh::FACES_E);
  Cell::index_type cell;
  if (locate(cell, p))
  {
    get_nodes(nodes,cell);
    const unsigned int nnodes = nodes.size();
    ASSERT(nnodes == 8);
    w.resize(nnodes);
      
    double sum = 0.0;
    unsigned int i;
    for (i=0; i < nnodes; i++)
    {
      const double a0 =
	tet_vol6(p, point(nodes[i]), point(nodes[wtable[i][0]]),
		 point(nodes[wtable[i][1]]));
      if (a0 < 1.0e-6)
      {
	get_face_weights(w, nodes, p, i, wtable[i][0],
			 wtable[i][3], wtable[i][1]);
	return;
      }
      const double a1 =
	tet_vol6(p, point(nodes[i]), point(nodes[wtable[i][1]]),
		 point(nodes[wtable[i][2]]));
      if (a1 < 1.0e-6)
      {
	get_face_weights(w, nodes, p, i, wtable[i][1],
			 wtable[i][4], wtable[i][2]);
	return;
      }
      const double a2 =
	tet_vol6(p, point(nodes[i]), point(nodes[wtable[i][2]]),
		 point(nodes[wtable[i][0]]));
      if (a2 < 1.0e-6)
      {
	get_face_weights(w, nodes, p, i, wtable[i][2],
			 wtable[i][5], wtable[i][0]);
	return;
      }
      w[i] = tet_vol6(point(nodes[i]), point(nodes[wtable[i][0]]),
		      point(nodes[wtable[i][1]]), point(nodes[wtable[i][2]]))
	/ (a0 * a1 * a2);
      sum += w[i];
    }
    const double suminv = 1.0 / sum;
    for (i = 0; i < nnodes; i++)
    {
      w[i] *= suminv;
    }
  }
}


void
HexVolMesh::compute_grid()
{
  grid_lock_.lock();
  if (synchronized_ & LOCATE_E) {
    grid_lock_.unlock();
    return;
  }
  if (grid_.get_rep() != 0) {grid_lock_.unlock(); return;} // only create once.

  BBox bb = get_bounding_box();
  if (!bb.valid()) { grid_lock_.unlock(); return; }

  // Cubed root of number of cells to get a subdivision ballpark.
  const double one_third = 1.L/3.L;
  Cell::size_type csize;  size(csize);
  const int s = ((int)ceil(pow((double)csize , one_third))) / 2 + 2;
  const Vector cell_epsilon = bb.diagonal() * (0.01 / s);
  bb.extend(bb.min() - cell_epsilon*2);
  bb.extend(bb.max() + cell_epsilon*2);

  LatVolMeshHandle mesh(scinew LatVolMesh(s, s, s, bb.min(), bb.max()));
  grid_ = scinew LatVolField<vector<Cell::index_type> >(mesh, Field::CELL);
  LatVolField<vector<Cell::index_type> >::fdata_type &fd = grid_->fdata();

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
    const Point padmin(box.min() - cell_epsilon);
    const Point padmax(box.max() + cell_epsilon);
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

  synchronized_ |= LOCATE_E;
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
    if (Dot(off0, normal) * Dot(off1, normal) < 0.0)
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
    return static_cast<Node::index_type>(points_.size() - 1);
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



HexVolMesh::Node::index_type
HexVolMesh::add_point(const Point &p)
{
  points_.push_back(p);
  return static_cast<Node::index_type>(points_.size() - 1);
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
  return static_cast<Elem::index_type>((cells_.size() - 1) >> 3);
}


#define HEXVOLMESH_VERSION 2

void
HexVolMesh::io(Piostream &stream)
{
  const int version = stream.begin_class(type_name(-1), HEXVOLMESH_VERSION);
  Mesh::io(stream);

  SCIRun::Pio(stream, points_);
  SCIRun::Pio(stream, cells_);
  if (version == 1)
  {
    vector<under_type>  face_neighbors;
    SCIRun::Pio(stream, face_neighbors);
  }

  stream.end_class();

  if (stream.reading())
  {
    synchronized_ = NODES_E | CELLS_E;
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



#if 0
void
HexVolMesh::compute_face_neighbors(double err)
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
  vector<list<unsigned long> > edgemap(points_.size());
  for (i=0; i< cells_.size(); i++)
  {
    edgemap[cells_[i]].push_back(i);
  }


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
#endif



} // namespace SCIRun
