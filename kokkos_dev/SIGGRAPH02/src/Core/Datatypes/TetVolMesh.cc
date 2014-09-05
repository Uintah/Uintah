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
#include <Core/Geometry/BBox.h>
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

  //! Unique Edges
#ifdef HAVE_HASH_SET
  edge_hasher_(cells_),
#endif
  edge_eq_(cells_),
#ifdef HAVE_HASH_SET
  all_edges_(100,edge_hasher_,edge_eq_),
  edges_(100,edge_hasher_,edge_eq_),
#else
  all_edges_(edge_eq_),
  edges_(edge_eq_),
#endif
  edge_lock_("TetVolMesh edges_ fill lock"),

  //! Unique Faces
#ifdef HAVE_HASH_SET
  face_hasher_(cells_),
#endif
  face_eq_(cells_),
#ifdef HAVE_HASH_SET
  all_faces_(100,face_hasher_,face_eq_),
  faces_(100,face_hasher_,face_eq_),
#else
  all_faces_(face_eq_),
  faces_(face_eq_),
#endif
  face_lock_("TetVolMesh faces_ fill lock"),

  node_neighbors_(0),
  node_neighbor_lock_("TetVolMesh node_neighbors_ fill lock"),
  grid_(0),
  grid_lock_("TetVolMesh grid_ fill lock"),
  synchronized_(CELLS_E | NODES_E)
{
}

TetVolMesh::TetVolMesh(const TetVolMesh &copy):
  points_(copy.points_),
  points_lock_("TetVolMesh points_ fill lock"),
  cells_(copy.cells_),
  cells_lock_("TetVolMesh cells_ fill lock"),
#ifdef HAVE_HASH_SET
  edge_hasher_(cells_),
#endif
  edge_eq_(cells_),
#ifdef HAVE_HASH_SET
  all_edges_(100,edge_hasher_,edge_eq_),
  edges_(100,edge_hasher_,edge_eq_),
#else
  all_edges_(edge_eq_),
  edges_(edge_eq_),
#endif
  edge_lock_("TetVolMesh edges_ fill lock"),
#ifdef HAVE_HASH_SET
  face_hasher_(cells_),
#endif
  face_eq_(cells_),
#ifdef HAVE_HASH_SET
  all_faces_(100,face_hasher_,face_eq_),
  faces_(100,face_hasher_,face_eq_),
#else
  all_faces_(face_eq_),
  faces_(face_eq_),
#endif
  face_lock_("TetVolMesh edges_ fill lock"),
  node_neighbors_(0),
  node_neighbor_lock_("TetVolMesh node_neighbors_ fill lock"),
  grid_(copy.grid_),
  grid_lock_("TetVolMesh grid_ fill lock"),
  synchronized_(copy.synchronized_)
{
}

TetVolMesh::~TetVolMesh()
{
}


/* To generate a random point inside of a tetrahedron, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void
TetVolMesh::get_random_point(Point &p, const Cell::index_type &ei,
			     int seed) const
{
  static MusilRNG rng;

  // get positions of the vertices
  Node::array_type ra(4);
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
  //! TODO: This could be included in the synchronize scheme
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
TetVolMesh::compute_faces()
{  
  face_lock_.lock();
  faces_.clear();
  all_faces_.clear();
  unsigned int i, num_cells = cells_.size();
  for (i = 0; i < num_cells; i++)
  {
    faces_.insert(i);
    all_faces_.insert(i);
  }
  synchronized_ |= FACES_E;
  synchronized_ |= FACE_NEIGHBORS_E;
  face_lock_.unlock();
}


void
TetVolMesh::compute_edges()
{
  edge_lock_.lock();
  edges_.clear();
  all_edges_.clear();
  unsigned int i, num_cells = (cells_.size()) / 4 * 6;
  for (i = 0; i < num_cells; i++)
  {
    edges_.insert(i);
    all_edges_.insert(i);
  }
  synchronized_ |= EDGES_E;
  synchronized_ |= EDGE_NEIGHBORS_E;
  edge_lock_.unlock();
}

void
TetVolMesh::compute_node_neighbors()
{
  node_neighbor_lock_.lock();
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  unsigned int i, num_cells = cells_.size();
  for (i = 0; i < num_cells; i++)
  {
    node_neighbors_[cells_[i]].push_back(i);
  }
  synchronized_ |= NODE_NEIGHBORS_E;
  node_neighbor_lock_.unlock();
}




bool
TetVolMesh::synchronize(unsigned int tosync)
{
  if (tosync & NODE_NEIGHBORS_E && !(synchronized_ & NODE_NEIGHBORS_E))
    compute_node_neighbors();
  if (tosync & EDGES_E && !(synchronized_ & EDGES_E) ||
      tosync & EDGE_NEIGHBORS_E && !(synchronized_ & EDGE_NEIGHBORS_E))
    compute_edges();
  if (tosync & FACES_E && !(synchronized_ & FACES_E) || 
      tosync & FACE_NEIGHBORS_E && !(synchronized_ & FACE_NEIGHBORS_E))
    compute_faces();
  if (tosync & GRID_E && !(synchronized_ & GRID_E))
    compute_grid();
  return true;
}


void
TetVolMesh::begin(TetVolMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E, "Must call synchronize on mesh first");
  itr = 0;
}

void
TetVolMesh::end(TetVolMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E, "Must call synchronize on mesh first");
  itr = points_.size();
}

void
TetVolMesh::size(TetVolMesh::Node::size_type &s) const
{
  ASSERTMSG(synchronized_ & NODES_E, "Must call synchronize on mesh first");
  s = points_.size();
}

void
TetVolMesh::begin(TetVolMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E, "Must call synchronize on mesh first");
  itr = edges_.begin();
}

void
TetVolMesh::end(TetVolMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E, "Must call synchronize on mesh first");
  itr = edges_.end();
}

void
TetVolMesh::size(TetVolMesh::Edge::size_type &s) const
{
  ASSERTMSG(synchronized_ & EDGES_E, "Must call synchronize on mesh first");
  s = edges_.size();
}

void
TetVolMesh::begin(TetVolMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E, "Must call synchronize on mesh first");
  itr = faces_.begin();
}

void
TetVolMesh::end(TetVolMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E, "Must call synchronize on mesh first");
  itr = faces_.end();
}

void
TetVolMesh::size(TetVolMesh::Face::size_type &s) const
{
  ASSERTMSG(synchronized_ & FACES_E, "Must call synchronize on mesh first");
  s = faces_.size();
}

void
TetVolMesh::begin(TetVolMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & ELEMENTS_E, "Must call synchronize on mesh first");
  itr = 0;
}

void
TetVolMesh::end(TetVolMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & ELEMENTS_E, "Must call synchronize on mesh first");
  itr = cells_.size() >> 2;
}

void
TetVolMesh::size(TetVolMesh::Cell::size_type &s) const
{
  ASSERTMSG(synchronized_ & ELEMENTS_E, "Must call synchronize on mesh first");
  s = cells_.size() >> 2;
}



void
TetVolMesh::create_cell_edges(Cell::index_type c)
{
  //ASSERT(!is_frozen());
  edge_lock_.lock();
  for (int i = c*6; i < c*6+6; ++i)
  {
    edges_.insert(i);
    all_edges_.insert(i);
  }
  edge_lock_.unlock();
}
      

void
TetVolMesh::delete_cell_edges(Cell::index_type c)
{
  //ASSERT(!is_frozen());
  edge_lock_.lock();
  for (int i = c*6; i < c*6+6; ++i)
  {
    //! If the Shared Edge Set is represented by the particular
    //! cell/edge index that is being recomputed, then
    //! remove it (and insert a non-recomputed edge if any left)
    bool shared_edge_exists = true;
    Edge::iterator shared_edge = edges_.find(i);
    // ASSERT guarantees edges were computed correctly for this cell
    ASSERT(shared_edge != edges_.end());
    if ((*shared_edge).index_ == i) 
    {
      edges_.erase(shared_edge);
      shared_edge_exists = false;
    }
    
    Edge::HalfEdgeSet::iterator half_edge_to_delete = all_edges_.end();
    pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =
      all_edges_.equal_range(i);
    for (Edge::HalfEdgeSet::iterator e = range.first; e != range.second; ++e)
    {
      if ((*e).index_ == i)
      {
	half_edge_to_delete = e;
      }
      else if (!shared_edge_exists)
      {
	edges_.insert((*e).index_);
	shared_edge_exists = true;
      }
      //! At this point, the edges_ set has the new index for this 
      //! shared edge and we know what half-edge is getting deleted below
      if (half_edge_to_delete != all_edges_.end() && shared_edge_exists) break;
    }
    //! ASSERT guarantees edges were computed correctly for this cell
    ASSERT(half_edge_to_delete != all_edges_.end());
    all_edges_.erase(half_edge_to_delete);
  }
  edge_lock_.unlock();
}

void
TetVolMesh::create_cell_faces(Cell::index_type c)
{
  //ASSERT(!is_frozen());
  face_lock_.lock();
  for (int i = c*4; i < c*4+4; ++i)
  {
    faces_.insert(i);
    all_faces_.insert(i);
  }
  face_lock_.unlock();
}

void
TetVolMesh::delete_cell_faces(Cell::index_type c)
{
  //ASSERT(!is_frozen());
  face_lock_.lock();
  for (int i = c*4; i < c*4+4; ++i)
  {
    // If the Shared Face Set is represented by the particular
    // cell/face index that is being recomputed, then
    // remove it (and insert a non-recomputed shared face if any exist)
    bool shared_face_exists = true;
    Face::FaceSet::iterator shared_face = faces_.find(i);
    ASSERT(shared_face != faces_.end());
    if ((*shared_face).index_ == i) 
    {
      faces_.erase(shared_face);
      shared_face_exists = false;
    }
    
    Face::HalfFaceSet::iterator half_face_to_delete = all_faces_.end();
    pair<Face::HalfFaceSet::iterator, Face::HalfFaceSet::iterator> range =
      all_faces_.equal_range(i);
    for (Face::HalfFaceSet::iterator e = range.first; e != range.second; ++e)
    {
      if ((*e).index_ == i)
      {
	half_face_to_delete = e;
      }
      else if (!shared_face_exists)
      {
	faces_.insert((*e).index_);
	shared_face_exists = true;
      }
      if (half_face_to_delete != all_faces_.end() && shared_face_exists) break;
    }

    //! If this ASSERT is reached, it means that the faces
    //! were not computed correctlyfor this cell
    ASSERT(half_face_to_delete != all_faces_.end());
    all_faces_.erase(half_face_to_delete);
  }
  face_lock_.unlock();
}

void
TetVolMesh::create_cell_node_neighbors(Cell::index_type c)
{
  //ASSERT(!is_frozen());
  node_neighbor_lock_.lock();
  for (int i = c*4; i < c*4+4; ++i)
  {
    node_neighbors_[cells_[i]].push_back(i);
  }
  node_neighbor_lock_.unlock();
}


void
TetVolMesh::delete_cell_node_neighbors(Cell::index_type c)
{
  //ASSERT(!is_frozen());
  node_neighbor_lock_.lock();
  for (int i = c*4; i < c*4+4; ++i)
  {
    const int n = cells_[i];
    vector<Cell::index_type>::iterator node_cells_end = node_neighbors_[n].end();
    vector<Cell::index_type>::iterator cell = node_neighbors_[n].begin();
    while (cell != node_cells_end && (*cell) != i) ++cell;

    //! ASSERT that the node_neighbors_ structure contains this cell
    ASSERT(cell != node_cells_end);

    node_neighbors_[n].erase(cell);
  }
  node_neighbor_lock_.unlock();      
}

      

//! Given two nodes (n0, n1), return all edge indexes that
//! span those two nodes
bool
TetVolMesh::is_edge(Node::index_type n0, Node::index_type n1,
		    Edge::array_type *array)
{
  ASSERTMSG(synchronized_ & EDGES_E, "Must call synchronize EDGES_E on TetVolMesh first.");
  edge_lock_.lock();
  cells_lock_.lock();

  //! Create a phantom cell with edge 0 being the one we're searching for
  const int fake_edge = cells_.size() / 4 * 6;
  vector<under_type>::iterator c0 = cells_.insert(cells_.end(),n0);
  vector<under_type>::iterator c1 = cells_.insert(cells_.end(),n1);

  //! Search the all_edges_ multiset for edges matching our fake_edge
  pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =
    all_edges_.equal_range(fake_edge);

  if (array)
  {
    array->clear();
    copy(range.first, range.second, array->end());
  }

  //! Delete the pahntom cell
  cells_.erase(c0);
  cells_.erase(c1);

  edge_lock_.unlock();
  cells_lock_.unlock();

  return range.first != range.second;
}

//! Given three nodes (n0, n1, n2), return all facee indexes that
//! span those three nodes
bool
TetVolMesh::is_face(Node::index_type n0,Node::index_type n1, 
		    Node::index_type n2, Face::array_type *array)
{
  ASSERTMSG(synchronized_ & FACES_E, "Must call synchronize FACES_E on TetVolMesh first.");
  face_lock_.lock();
  cells_lock_.lock();

  //! Create a phantom cell with face 3 being the one we're searching for
  const int fake_face = cells_.size() + 3;
  vector<under_type>::iterator c0 = cells_.insert(cells_.end(),n0);
  vector<under_type>::iterator c1 = cells_.insert(cells_.end(),n1);
  vector<under_type>::iterator c2 = cells_.insert(cells_.end(),n2);

  //! Search the all_face_ multiset for edges matching our fake_edge
  pair<Face::HalfFaceSet::const_iterator, Face::HalfFaceSet::const_iterator>
     range = all_faces_.equal_range(fake_face);

  if (array)
  {
    array->clear();
    copy(range.first, range.second, array->end());
  }

  //! Delete the pahntom cell
  cells_.erase(c0);
  cells_.erase(c1);
  cells_.erase(c2);

  face_lock_.unlock();
  cells_lock_.unlock();
  return range.first != range.second;
}
  
  
  
void
TetVolMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.clear();
  pair<Edge::index_type, Edge::index_type> edge = Edge::edgei(idx);
  array.push_back(cells_[edge.first]);
  array.push_back(cells_[edge.second]);
}


void
TetVolMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.clear();
  const int base = idx/4*4;
  for (int i = base; i < base+4; i++)
    if (i != idx.index_)
      array.push_back(cells_[i]);
}


void
TetVolMesh::get_nodes(Node::array_type &array, Cell::index_type idx) const
{
  array.clear();
  for (int n = idx*4; n < idx*4+4; ++n)
    array.push_back(cells_[n]);
}

void
TetVolMesh::set_nodes(Node::array_type &array, Cell::index_type idx)
{
  ASSERT(array.size() == 4);

  if (synchronized_ & EDGES_E) delete_cell_edges(idx);
  if (synchronized_ & FACES_E) delete_cell_faces(idx);
  if (synchronized_ & NODE_NEIGHBORS_E) delete_cell_node_neighbors(idx);

  for (int n = 4; n < 4; ++n)
    cells_[idx * 4 + n] = array[n];
  
  synchronized_ &= ~GRID_E;
  if (synchronized_ & EDGES_E) create_cell_edges(idx);
  if (synchronized_ & FACES_E) create_cell_faces(idx);
  if (synchronized_ & NODE_NEIGHBORS_E) create_cell_node_neighbors(idx);

}

void
TetVolMesh::get_edges(Edge::array_type &/*array*/, Node::index_type /*idx*/) const
{
  ASSERTFAIL("Not implemented yet");
}


void
TetVolMesh::get_edges(Edge::array_type &/*array*/, Face::index_type /*idx*/) const
{
  ASSERTFAIL("Not implemented correctly");
#if 0
  array.clear();    
  static int table[4][3] =
  {
    {3, 4, 5},
    {1, 2, 5},
    {0, 2, 4},
    {0, 1, 3}
  };

  const int base = idx / 4 * 6;
  const int off = idx % 4;
  array.push_back(base + table[off][0]);
  array.push_back(base + table[off][1]);
  array.push_back(base + table[off][2]);
#endif
}


void
TetVolMesh::get_edges(Edge::array_type &array, Cell::index_type idx) const
{
  array.clear();
  for (int e = idx * 6; e < idx * 6 + 6; e++)
  {
    array.push_back(e);
  }
}



void
TetVolMesh::get_faces(Face::array_type &/*array*/, Node::index_type /*idx*/) const
{
  ASSERTFAIL("Not implemented yet");
}

void
TetVolMesh::get_faces(Face::array_type &/*array*/, Edge::index_type /*idx*/) const
{
  ASSERTFAIL("Not implemented yet");
}


void
TetVolMesh::get_faces(Face::array_type &array, Cell::index_type idx) const
{
  array.clear();
  for (int f = idx * 4; f < idx * 4 + 4; f++)
    array.push_back(f);
}

void
TetVolMesh::get_cells(Cell::array_type &array, Edge::index_type idx) const
{
  pair<Edge::HalfEdgeSet::const_iterator, Edge::HalfEdgeSet::const_iterator>
    range = all_edges_.equal_range(idx);

  //! ASSERT that this cell's edges have been computed
  ASSERT(range.first != range.second);

  array.clear();
  while (range.first != range.second)
  {
    array.push_back((*range.first)/4);
    ++range.first;
  }
}


void
TetVolMesh::get_cells(Cell::array_type &array, Face::index_type idx) const
{
  pair<Face::HalfFaceSet::const_iterator, 
       Face::HalfFaceSet::const_iterator> range = all_faces_.equal_range(idx);

  //! ASSERT that this cell's faces have been computed
  ASSERT(range.first != range.second);

  array.clear();
  while (range.first != range.second)
  {
    array.push_back((*range.first)/4);
    ++range.first;
  }
}

  

//! this is a bad hack for existing code that calls this function
//! call the one below instead
bool
TetVolMesh::get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
			 Face::index_type idx) const
{
  ASSERT(idx/4 == from);
  Face::index_type neigh;
  bool ret_val = get_neighbor(neigh, idx);
  neighbor.index_ = neigh.index_;
  return ret_val;
}



//! given a face index, return the face index that spans the same 3 nodes
bool
TetVolMesh::get_neighbor(Face::index_type &neighbor, Face::index_type idx)const
{
  ASSERTMSG(synchronized_ & FACE_NEIGHBORS_E, "Must call synchronize FACE_NEIGHBORS_E on TetVolMesh first.");
  pair<Face::HalfFaceSet::const_iterator,
       Face::HalfFaceSet::const_iterator> range = all_faces_.equal_range(idx);

  // ASSERT that this face was computed
  ASSERT(range.first != range.second);

  // Cell has no neighbor
  Face::HalfFaceSet::const_iterator second = range.first;
  if (++second == range.second)
  {
    neighbor = -1;
    return false;
  }

  if ((*range.first).index_ == idx)
    neighbor = (*second).index_;
  else if ((*second).index_ == idx)
    neighbor = (*range.first).index_;
  else {ASSERTFAIL("Non-Manifold Face in all_faces_ structure.");}

  return true;
}  
  
  


void
TetVolMesh::get_neighbors(Cell::array_type &array, Cell::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACE_NEIGHBORS_E, "Must call synchronize FACE_NEIGHBORS_E on TetVolMesh first.");
  Face::index_type face;
  for (int i = idx*4; i < idx*4+4;i++)
  {
    face.index_ = i;
    pair<const Face::HalfFaceSet::const_iterator,
         const Face::HalfFaceSet::const_iterator> range =
      all_faces_.equal_range(face);
    for (Face::HalfFaceSet::const_iterator iter = range.first;
	 iter != range.second; ++iter)
      if (*iter != i)
	array.push_back(*iter/4);
  } 
}

void
TetVolMesh::get_neighbors(Node::array_type &array, Node::index_type idx) const
{
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E, 
	    "Must call synchronize NODE_NEIGHBORS_E on TetVolMesh first.");
  set<int> inserted;
  for (unsigned int i = 0; i < node_neighbors_[idx].size(); i++)
  {
    const int base = node_neighbors_[idx][i]/4*4;
    for (int c = base; c < base+4; c++)
      if (c != idx && inserted.find(cells_[c]) == inserted.end())
      {
	inserted.insert(cells_[c]);
	array.push_back(cells_[c]);
      }
  }
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
  Node::array_type arr(2);
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
  Node::array_type arr(3);
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
  const Point &p0 = points_[cells_[idx * 4 + 0]];
  const Point &p1 = points_[cells_[idx * 4 + 1]];
  const Point &p2 = points_[cells_[idx * 4 + 2]];
  const Point &p3 = points_[cells_[idx * 4 + 3]];

  p = ((p0.asVector() + p1.asVector() +
	p2.asVector() + p3.asVector()) * s).asPoint();
}


bool
TetVolMesh::locate(Node::index_type &loc, const Point &p)
{
  Cell::index_type ci;
  if (locate(ci, p)) // first try the fast way.
  {
    Node::array_type nodes;
    get_nodes(nodes, ci);

    Point ptmp;
    double mindist;
    for (int i=0; i<4; i++)
    {
      get_center(ptmp, nodes[i]);
      double dist = (p - ptmp).length2();
      if (i == 0 || dist < mindist)
      {
	mindist = dist;
	loc = nodes[i];
      }
    }
    return true;
  }
  else
  {  // do exhaustive search.
    bool found_p = false;
    double mindist;
    Node::iterator bi; begin(bi);
    Node::iterator ei; end(ei);
    while (bi != ei)
    {
      Point c;
      get_center(c, *bi);
      const double dist = (p - c).length2();
      if (!found_p || dist < mindist)
      {
	mindist = dist;
	loc = *bi;
	found_p = true;
      }
      ++bi;
    }
    return found_p;
  }
}


bool
TetVolMesh::locate(Edge::index_type &edge, const Point &p)
{
  bool found_p = false;
  double mindist;
  Edge::iterator bi; begin(bi);
  Edge::iterator ei; end(ei);
  while (bi != ei)
  {
    Point c;
    get_center(c, *bi);
    const double dist = (p - c).length2();
    if (!found_p || dist < mindist)
    {
      mindist = dist;
      edge = *bi;
      found_p = true;
    }
    ++bi;
  }
  return found_p;
}


bool
TetVolMesh::locate(Face::index_type &face, const Point &p)
{
  bool found_p = false;
  double mindist;
  Face::iterator bi; begin(bi);
  Face::iterator ei; end(ei);
  while (bi != ei)
  {
    Point c;
    get_center(c, *bi);
    const double dist = (p - c).length2();
    if (!found_p || dist < mindist)
    {
      mindist = dist;
      face = *bi;
      found_p = true;
    }
    ++bi;
  }
  return found_p;
}


bool
TetVolMesh::locate(Cell::index_type &cell, const Point &p)
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
  while (iter != v.end())
  {
    if (inside4_p((*iter) * 4, p))
    {
      cell = *iter;
      return true;
    }
    ++iter;
  }
  return false;
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
    const double vol0 = (Cross(Cross(p1-p,p2-p),p3-p)).length();
    const double vol1 = (Cross(Cross(p0-p,p2-p),p3-p)).length();
    const double vol2 = (Cross(Cross(p0-p,p1-p),p3-p)).length();
    const double vol3 = (Cross(Cross(p0-p,p1-p),p2-p)).length();
    const double vol_sum = vol0+vol1+vol2+vol3;
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

  BBox bb = get_bounding_box();
  if (!bb.valid()) { grid_lock_.unlock(); return; }
  // cubed root of number of cells to get a subdivision ballpark
  const double one_third = 1.L/3.L;
  Cell::size_type csize;  size(csize);
  const int s = ((int)ceil(pow((double)csize , one_third))) / 2 + 2;
  const Vector cell_epsilon = bb.diagonal() * (0.01 / s);

  LatVolMeshHandle mesh(scinew LatVolMesh(s, s, s, bb.min(), bb.max()));
  grid_ = scinew LatVolField<vector<Cell::index_type> >(mesh, Field::CELL);
  grid_->resize_fdata();
  LatVolField<vector<Cell::index_type> >::fdata_type &fd = grid_->fdata();

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
  synchronized_ |= GRID_E;
  grid_lock_.unlock();
}



bool
TetVolMesh::inside4_p(int i, const Point &p)
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



//! This code uses the robust geometric predicates 
//! in Core/Math/Predicates.h
//! for some reason they crash right now, so this code is not compiled in
#if 0
bool
TetVolMesh::inside4_p(int i, const Point &p)
{
  double *p0 = &points_[cells_[i*4+0]](0);
  double *p1 = &points_[cells_[i*4+1]](0);
  double *p2 = &points_[cells_[i*4+2]](0);
  double *p3 = &points_[cells_[i*4+3]](0);

  return (orient3d(p2, p1, p3, p0) < 0.0 &&
	  orient3d(p0, p2, p3, p1) < 0.0 &&
	  orient3d(p0, p3, p1, p2) < 0.0 &&
	  orient3d(p0, p1, p2, p3) < 0.0);
}

void
TetVolMesh::rewind_mesh()
{
  //! Fix Tetrahedron orientation.
  //! TetVolMesh tets are oriented as follows:
  //! Points 0, 1, & 2 map out face 3 in a counter-clockwise order
  //! Point 3 is above the plane of face 3 in a right handed coordinate system.
  //! Therefore, crossing edge #0(0-1) and edge #2(0-2) creates a normal that
  //! points in the (general) direction of Point 3.  
  vector<Point>::size_type i, num_cells = cells_.size();
  for (i = 0; i < num_cells/4; i++)
  {   
    //! This is the approximate tet volume * 6.  All we care about is sign.
    //! orient3d will return EXACTLY 0.0 if point d lies on plane made by a,b,c
    const double tet_vol = orient3d(&points_[cells_[i*4+0]](0), 
				    &points_[cells_[i*4+1]](0),
				    &points_[cells_[i*4+2]](0),
				    &points_[cells_[i*4+3]](0));
    //! Tet is oriented backwards.  Swap index #0 and #1 to re-orient tet.
    if (tet_vol > 0.) 
      flip(i);
    else if (tet_vol == 0.) // orient3d is exact, no need for epsilon
      // TODO: Degerate tetrahedron (all 4 nodes lie on a plane), mark to delete
      cerr << "Zero Volume Tetrahedron #" << i << ".  Need to delete\n";
    //! else means Tet is valid.  Do nothing.
  }
}

#endif

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
  if (locate(i, p) && (points_[i] - p).length2() < err)
  {
    return i;
  }
  else
  {
    points_.push_back(p);
    if (synchronized_ & NODE_NEIGHBORS_E)
      node_neighbors_.push_back(vector<Cell::index_type>());
    return points_.size() - 1;
  }
}


TetVolMesh::Elem::index_type
TetVolMesh::add_tet(Node::index_type a, Node::index_type b, 
		    Node::index_type c, Node::index_type d)
{
  const int tet = cells_.size() / 4;
  cells_.push_back(a);
  cells_.push_back(b);
  cells_.push_back(c);
  cells_.push_back(d);

  if (synchronized_ & NODE_NEIGHBORS_E) create_cell_node_neighbors(tet);
  if (synchronized_ & EDGES_E) create_cell_edges(tet);
  if (synchronized_ & FACES_E) create_cell_faces(tet);
  synchronized_ &= ~GRID_E;

  return tet; 
}



TetVolMesh::Node::index_type
TetVolMesh::add_point(const Point &p)
{
  points_.push_back(p);
  if (synchronized_ & NODE_NEIGHBORS_E)
    node_neighbors_.push_back(vector<Cell::index_type>());
  return points_.size() - 1;
}


TetVolMesh::Elem::index_type
TetVolMesh::add_tet(const Point &p0, const Point &p1, const Point &p2,
		    const Point &p3)
{
  return add_tet(add_find_point(p0), add_find_point(p1), 
		 add_find_point(p2), add_find_point(p3));
}


TetVolMesh::Elem::index_type
TetVolMesh::add_elem(Node::array_type a)
{
  ASSERT(a.size() == 4);

  const int tet = cells_.size() / 4;
 
  for (unsigned int n = 0; n < a.size(); n++)
    cells_.push_back(a[n]);

  if (synchronized_ & NODE_NEIGHBORS_E) create_cell_node_neighbors(tet);
  if (synchronized_ & EDGES_E) create_cell_edges(tet);
  if (synchronized_ & FACES_E) create_cell_faces(tet);
  synchronized_ &= ~GRID_E;

  return tet;
}



void
TetVolMesh::delete_cells(set<int> &to_delete)
{
  vector<under_type> old_cells = cells_;
  int i = 0, c;

  cells_.clear();
  cells_.reserve(old_cells.size() - to_delete.size()*4);

  for (set<int>::iterator deleted = to_delete.begin();
       deleted != to_delete.end(); ++deleted)
  {
    for (;i < *deleted; ++i)
      for (c = i*4; c < i*4+4; ++c)
	cells_.push_back(old_cells[c]);
    ++i;

  }

  for (; i < (int)(old_cells.size()/4); ++i)
    for (c = i*4; c < i*4+4; ++c)
      cells_.push_back(old_cells[c]);
  
}


double 
TetVolMesh::volume(TetVolMesh::Cell::index_type ci)
{
  TetVolMesh::Node::array_type n;
  get_nodes(n, ci);
  Point p1, p2, p3, p4;
  get_point(p1, n[0]);
  get_point(p2, n[1]);
  get_point(p3, n[2]);
  get_point(p4, n[3]);
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
  return (a1+a2+a3+a4)/6.;
}

void
TetVolMesh::orient(Cell::index_type ci) {
  double sgn=volume(ci);
  if(sgn < 0.0){
    // Switch two of the edges so that the volume is positive
    int tmp = cells_[ci * 4];
    cells_[ci * 4] = cells_[ci * 4 + 1];
    cells_[ci * 4 + 1] = tmp;
    sgn=-sgn;
  }
  if(sgn < 1.e-9){ // return 0; // Degenerate...
    cerr << "Warning - small element, volume=" << sgn << endl;
  }
}



#define TETVOLMESH_VERSION 2

void
TetVolMesh::io(Piostream &stream)
{
  const int version = stream.begin_class(type_name(-1), TETVOLMESH_VERSION);
  Mesh::io(stream);

  SCIRun::Pio(stream, points_);
  SCIRun::Pio(stream, cells_);
  if (version == 1)
  {
    vector<int> neighbors;
    SCIRun::Pio(stream, neighbors);
  }

  // orient the tets..
  Cell::iterator iter, endit;
  begin(iter);
  end(endit);
  while(iter != endit) {
    orient(*iter);
    ++iter;
  }

  stream.end_class();
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



void
TetVolMesh::get_cells(Cell::array_type &array, Node::index_type idx) const
{
  ASSERTMSG(is_frozen(),"only call get_cells with a node index if frozen!!");
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E, 
	    "Must call synchronize NODE_NEIGHBORS_E on TetVolMesh first.");
  array.clear();
  for (unsigned int i = 0; i < node_neighbors_[idx].size(); ++i)
    array.push_back(node_neighbors_[idx][i]/4);
}


} // namespace SCIRun
