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
#include <Core/Geometry/BBox.h>
#include <Core/Math/MusilRNG.h>

#include <algorithm>

#include <float.h> // for DBL_MAX

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
  synchronized_ &= ~EDGES_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~FACES_E;
  synchronized_ &= ~FACE_NEIGHBORS_E;
}

TetVolMesh::~TetVolMesh()
{
}


/* To generate a random point inside of a tetrahedron, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void
TetVolMesh::get_random_point(Point &p, Cell::index_type ei, int seed) const
{
  static MusilRNG rng;

  // get positions of the vertices
  Node::array_type ra(4);
  get_nodes(ra,ei);
  const Point &p0 = point(ra[0]);
  const Point &p1 = point(ra[1]);
  const Point &p2 = point(ra[2]);
  const Point &p3 = point(ra[3]);

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
    result.extend(point(*ni));
    ++ni;
  }
  return result;
}


void
TetVolMesh::transform(const Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  
  // Recompute grid.
  synchronized_ &= ~LOCATE_E;
}



void
TetVolMesh::compute_faces()
{  
  face_lock_.lock();
  if ((synchronized_ & FACES_E) && (synchronized_ & FACE_NEIGHBORS_E)) {
    face_lock_.unlock();
    return;
  }
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
  if ((synchronized_ & EDGES_E) && (synchronized_ & EDGE_NEIGHBORS_E)) {
    edge_lock_.unlock();
    return;
  }
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
  if (synchronized_ & NODE_NEIGHBORS_E) {
    node_neighbor_lock_.unlock();
    return;
  }
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
  if (tosync & LOCATE_E && !(synchronized_ & LOCATE_E))
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
  if (!(synchronized_&EDGES_E) && !(synchronized_&EDGE_NEIGHBORS_E)) return;
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
  if (!(synchronized_&EDGES_E) && !(synchronized_&EDGE_NEIGHBORS_E)) return;
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
  if (!(synchronized_&FACES_E) && !(synchronized_&FACE_NEIGHBORS_E)) return;
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
  if (!(synchronized_&FACES_E) && !(synchronized_&FACE_NEIGHBORS_E)) return;
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
  if (!(synchronized_ & NODE_NEIGHBORS_E)) return;
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
  if (!(synchronized_ & NODE_NEIGHBORS_E)) return;
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


// Always returns nodes in counter-clockwise order
void
TetVolMesh::get_nodes(Node::array_type &a, Face::index_type idx) const
{
  a.resize(3);  
  const unsigned int offset = idx%4;
  const unsigned int b = idx - offset; // base cell index
  switch (offset)
  {
  case 0: a[0] = cells_[b+3]; a[1] = cells_[b+2]; a[2] = cells_[b+1]; break;
  case 1: a[0] = cells_[b+0]; a[1] = cells_[b+2]; a[2] = cells_[b+3]; break;
  case 2: a[0] = cells_[b+3]; a[1] = cells_[b+1]; a[2] = cells_[b+0]; break;
  default:
  case 3: a[0] = cells_[b+0]; a[1] = cells_[b+1]; a[2] = cells_[b+2]; break;
  }
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

  delete_cell_edges(idx);
  delete_cell_faces(idx);
  delete_cell_node_neighbors(idx);

  for (int n = 0; n < 4; ++n)
    cells_[idx * 4 + n] = array[n];
  
  synchronized_ &= ~LOCATE_E;
  create_cell_edges(idx);
  create_cell_faces(idx);
  create_cell_node_neighbors(idx);
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
  neighbor.index_ = neigh.index_ / 4;
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
      if (cells_[c] != idx && inserted.find(cells_[c]) == inserted.end())
      {
	inserted.insert(cells_[c]);
	array.push_back(cells_[c]);
      }
  }
}


void
TetVolMesh::get_center(Point &p, Edge::index_type idx) const
{
  const double s = 1.0/2.0;
  Node::array_type arr(2);
  get_nodes(arr, idx);
  get_point(p, arr[0]);
  const Point &p1 = point(arr[1]);

  p.asVector() += p1.asVector();
  p.asVector() *= s;
}


void
TetVolMesh::get_center(Point &p, Face::index_type idx) const
{
  const double s = 1.0/3.0;
  Node::array_type arr(3);
  get_nodes(arr, idx);
  get_point(p, arr[0]);
  const Point &p1 = point(arr[1]);
  const Point &p2 = point(arr[2]);

  p.asVector() += p1.asVector();
  p.asVector() += p2.asVector();
  p.asVector() *= s;
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

    double mindist = DBL_MAX;
    for (int i=0; i<4; i++)
    {
      const Point &ptmp = point(nodes[i]);
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
    double mindist = DBL_MAX;
    Node::iterator bi; begin(bi);
    Node::iterator ei; end(ei);
    while (bi != ei)
    {
      const Point &c = point(*bi);
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
  double mindist = DBL_MAX;
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
  double mindist = DBL_MAX;
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
  // Check last cell found first.  Copy cache to cell first so that we
  // don't care about thread safeness, such that worst case on
  // context switch is that cache is not found.
  static Cell::index_type cache(0);
  cell = cache;
  if (cell > Cell::index_type(0) &&
      cell < Cell::index_type(cells_.size()/4) &&
      inside(cell, p))
  {
    return true;
  }

  if (!(synchronized_ & LOCATE_E))
    synchronize(LOCATE_E);
  ASSERT(grid_.get_rep());

  LatVolMeshHandle mesh = grid_->get_typed_mesh();
  LatVolMesh::Cell::index_type ci;
  if (!mesh->locate(ci, p)) { return false; }
  vector<Cell::index_type> v = grid_->value(ci);
  vector<Cell::index_type>::iterator iter = v.begin();
  while (iter != v.end())
  {
    if (inside(*iter, p))
    {
      cell = *iter;
      cache = cell;
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


// The volume x 6, used by get_weights to compute barycentric coordinates.
static double
tet_vol6(const Point &p1, const Point &p2, const Point &p3, const Point &p4)
{
  return fabs( Dot(Cross(p2-p1,p3-p1),p4-p1) );
}

void
TetVolMesh::get_weights(const Point &p,
			Node::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    const Point &p0 = point(l[0]);
    const Point &p1 = point(l[1]);
    const Point &p2 = point(l[2]);
    const Point &p3 = point(l[3]);

    w.resize(4);
    w[0] = tet_vol6(p, p1, p2, p3);
    w[1] = tet_vol6(p, p0, p2, p3);
    w[2] = tet_vol6(p, p1, p0, p3);
    w[3] = tet_vol6(p, p1, p2, p0);
    const double vol_sum_inv = 1.0 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= vol_sum_inv;
    w[1] *= vol_sum_inv;
    w[2] *= vol_sum_inv;
    w[3] *= vol_sum_inv;
  }
}


void
TetVolMesh::compute_grid()
{
  grid_lock_.lock();
  if (synchronized_ & LOCATE_E) {
    grid_lock_.unlock();
    return;
  }

  BBox bb = get_bounding_box();
  if (bb.valid())
  {
    // Cubed root of number of cells to get a subdivision ballpark.
    Cell::size_type csize;  size(csize);
    const int s = (int)(ceil(pow((double)csize , (1.0/3.0)))) / 2 + 2;
    const Vector cell_epsilon = bb.diagonal() * (1.0e-4 / s);
    bb.extend(bb.min() - cell_epsilon*2);
    bb.extend(bb.max() + cell_epsilon*2);

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

      // Add this cell index to all overlapping cells in grid_
      LatVolMesh::Cell::array_type carr;
      mesh->get_cells(carr, box);
      LatVolMesh::Cell::array_type::iterator giter = carr.begin();
      while (giter != carr.end())
      {
	// Would like to just get a reference to the vector at the cell
	// but can't from value. Bypass the interface.
	vector<Cell::index_type> &v = fd[*giter];
	v.push_back(*ci);

	++giter;
      }
      ++ci;
    }
  }

  synchronized_ |= LOCATE_E;
  grid_lock_.unlock();
}


#if 0
bool
TetVolMesh::inside(Cell::index_type idx, const Point &p)
{
  Point center;
  get_center(center, idx);

  Face::array_type faces;
  get_faces(faces, idx);

  for (unsigned int i=0; i<faces.size(); i++) {
    Node::array_type ra;
    get_nodes(ra, faces[i]);

    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);

    const Vector v0(p0 - p1), v1(p2 - p1);
    const Vector normal = Cross(v0, v1);
    const Vector off0(p - p1);
    const Vector off1(center - p1);

    double dotprod = Dot(off0, normal);

    // Account for round off - the point may be on the plane!!
    if( fabs( dotprod ) < 1.0e-8 )
      continue;

    // If orientated correctly the second dot product is not needed.
    // Only need to check to see if the sign is negitive.
    if (dotprod * Dot(off1, normal) < 0.0)
      return false;
  }
  return true;
}
#else
bool
TetVolMesh::inside(Cell::index_type idx, const Point &p)
{
  // TODO: This has not been tested.
  // TODO: Looks like too much code to check sign of 4 plane/point tests.
  const Point &p0 = points_[cells_[idx*4+0]];
  const Point &p1 = points_[cells_[idx*4+1]];
  const Point &p2 = points_[cells_[idx*4+2]];
  const Point &p3 = points_[cells_[idx*4+3]];
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
#endif


//! This code uses the robust geometric predicates 
//! in Core/Math/Predicates.h
//! for some reason they crash right now, so this code is not compiled in
#if 0
bool
TetVolMesh::inside(int i, const Point &p)
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
    if (synchronized_ & NODE_NEIGHBORS_E) {
      node_neighbor_lock_.lock();
      node_neighbors_.push_back(vector<Cell::index_type>());
      node_neighbor_lock_.unlock();
    }
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

  create_cell_node_neighbors(tet);
  create_cell_edges(tet);
  create_cell_faces(tet);
  synchronized_ &= ~LOCATE_E;

  return tet; 
}



TetVolMesh::Node::index_type
TetVolMesh::add_point(const Point &p)
{
  points_.push_back(p);
  if (synchronized_ & NODE_NEIGHBORS_E) {
    node_neighbor_lock_.lock();
    node_neighbors_.push_back(vector<Cell::index_type>());
    node_neighbor_lock_.unlock();
  }
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

  create_cell_node_neighbors(tet);
  create_cell_edges(tet);
  create_cell_faces(tet);
  synchronized_ &= ~LOCATE_E;

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

void
TetVolMesh::orient(Cell::index_type ci) {

  Node::array_type ra;
  get_nodes(ra,ci);
  const Point &p0 = point(ra[0]);
  const Point &p1 = point(ra[1]);
  const Point &p2 = point(ra[2]);
  const Point &p3 = point(ra[3]);

  // Unsigned volumex6 of the tet.
  double sgn=Dot(Cross(p1-p0,p2-p0),p3-p0);

  if(sgn < 0.0) {
    // Switch two of the edges so that the volume is positive
    unsigned int base = ci * 4;
    unsigned int tmp = cells_[base];
    cells_[base] = cells_[base + 1];
    cells_[base + 1] = tmp;
    sgn=-sgn;
  }

  if(sgn < 1.e-9){ // return 0; // Degenerate...
    cerr << "Warning - small element, volume=" << sgn << endl;
  }
}


bool
TetVolMesh::insert_node(const Point &p)
{
  Node::index_type pi = add_point(p);
  Cell::index_type cell;
  locate(cell, p);

  const unsigned index = cell*4;
  if (!inside(index, p)) return false;

  delete_cell_node_neighbors(cell);
  delete_cell_edges(cell);
  delete_cell_faces(cell);

  Cell::array_type tets(4,cell);

  tets[1] = add_tet(cells_[index+0], cells_[index+3], cells_[index+1], pi);
  tets[2] = add_tet(cells_[index+1], cells_[index+3], cells_[index+2], pi);
  tets[3] = add_tet(cells_[index+0], cells_[index+2], cells_[index+3], pi);
  
  cells_[index+3] = pi;
    
  create_cell_node_neighbors(cell);
  create_cell_edges(cell);
  create_cell_faces(cell);

  return true;
}

 
/* From Comp.Graphics.Algorithms FAQ 5.21
 * Circumsphere of 4 points a,b,c,d
 * 
 *    |                                                                       |
 *    | |d-a|^2 [(b-a)x(c-a)] + |c-a|^2 [(d-a)x(b-a)] + |b-a|^2 [(c-a)x(d-a)] |
 *    |                                                                       |
 * r= -------------------------------------------------------------------------
 *                             | bx-ax  by-ay  bz-az |
 *                           2 | cx-ax  cy-ay  cz-az |
 *                             | dx-ax  dy-ay  dz-az |
 * 
 *
 *
 *        |d-a|^2 [(b-a)x(c-a)] + |c-a|^2 [(d-a)x(b-a)] + |b-a|^2 [(c-a)x(d-a)]
 * m= a + ---------------------------------------------------------------------
 *                               | bx-ax  by-ay  bz-az |
 *                             2 | cx-ax  cy-ay  cz-az |
 *                               | dx-ax  dy-ay  dz-az |
 */

pair<Point,double>
TetVolMesh::circumsphere(const Cell::index_type cell)
{
  const Point &a = points_[cells_[cell*4+0]];
  const Point &b = points_[cells_[cell*4+1]];
  const Point &c = points_[cells_[cell*4+2]];
  const Point &d = points_[cells_[cell*4+3]];

  const Vector bma = b-a;
  const Vector cma = c-a;
  const Vector dma = d-a;

  const double denominator = 
    2*(bma.x()*(cma.y()*dma.z()-dma.y()*cma.z())-
       bma.y()*(cma.x()*dma.z()-dma.x()*cma.z())+
       bma.z()*(cma.x()*dma.y()-dma.x()*cma.y()));
  
  const Vector numerator = 
    dma.length2()*Cross(bma,cma) + 
    cma.length2()*Cross(dma,bma) +
    bma.length2()*Cross(cma,dma);

  return make_pair(a+numerator/denominator,numerator.length()/denominator);
}


// Bowyer-Watson Node insertion for Delaunay Tetrahedralization
TetVolMesh::Node::index_type
TetVolMesh::insert_node_watson(const Point &p, Cell::array_type *new_cells, Cell::array_type *mod_cells)
{
  Cell::index_type cell;
  synchronize(LOCATE_E | FACE_NEIGHBORS_E);
  if (!locate(cell,p)) { 
    cerr << "Watson outside volume: " << p.x() << ", " << p.y() << ", " << p.z() << endl;
    return (TetVolMesh::Node::index_type)(-1); 
  }

  Node::index_type new_point_index = add_point(p);

  // set of tets checked for circumsphere point intersection
  set<Cell::index_type> cells_checked, cells_removed;
  cells_removed.insert(cell);
  cells_checked.insert(cell);
  
  unsigned int face;
  // set of faces that need to be checked for neighboring tet removal
  set<Face::index_type> faces_todo;
  for (face = cell*4; face < (unsigned int)cell*4+4; ++face)
    faces_todo.insert(Face::index_type(face));

  // set of node triplets that form face on hull interior
  vector<Node::array_type> hull_nodes;

  // Propagate front until we have checked all faces on hull
  while (!faces_todo.empty())
  {
    set<Face::index_type> faces = faces_todo;  
    set<Face::index_type>::iterator faces_iter = faces.begin();
    set<Face::index_type>::iterator faces_end = faces.end();
    faces_todo.clear();
    for (;faces_iter != faces_end; ++faces_iter)
    {
      // Face index of neighboring tet that shares this face
      Face::index_type nbr;
      if (!get_neighbor(nbr, *faces_iter))
      {
	// This was a boundary face, therefore on the hull
	hull_nodes.push_back(Node::array_type());
	get_nodes(hull_nodes.back(),*faces_iter);
      }
      else // not a boundary face
      {	
	// Index of neighboring tet that we need to check for removal
        cell = Cell::index_type(nbr/4);
	// Check to see if we didnt reach this cell already from other path
	if (cells_checked.find(cell) == cells_checked.end())
	{
	  cells_checked.insert(cell);
	  // Get the circumsphere of tet
	  pair<Point,double> sphere = circumsphere(cell);
	  if ((sphere.first - p).length() < sphere.second)
	  {
	    // Point is within circumsphere of Cell
	    // mark for removal
	    cells_removed.insert(cell);
	    // Now add all of its faces (minus the one we crossed to get here)
	    // to be crossed the next time around
	    for (face = cell*4; face < (unsigned int)cell*4+4; ++face)
	      if (face != (unsigned int)nbr) // dont add nbr already crossed
		faces_todo.insert(Face::index_type(face));
	  }
	  else
	  {
	    // The point is not within the circumsphere of the cell
	    // therefore the face we crossed is on the interior hull
	    hull_nodes.push_back(Node::array_type());
	    get_nodes(hull_nodes.back(),*faces_iter);
	  }
	}
      }
    }
  }

  unsigned int num_hull_faces = hull_nodes.size();
  unsigned int num_cells_removed = cells_removed.size();
  ASSERT(num_hull_faces >= num_cells_removed);
  
  // A list of all tets that were modifed/added
  vector<Cell::index_type> tets(num_hull_faces);
  
  // Re-define already allocated tets to include new point
  // and the 3 points of an interior hulls face  
  set<Cell::index_type>::iterator cells_removed_iter = cells_removed.begin();
  for (face = 0; face < num_cells_removed; face++)
  {
    tets[face] = mod_tet(*cells_removed_iter, 
			 hull_nodes[face][0],
			 hull_nodes[face][1],
			 hull_nodes[face][2],
			 new_point_index);
    if (mod_cells) mod_cells->push_back(tets[face]);
    ++cells_removed_iter;
  }
  


  for (face = num_cells_removed; face < num_hull_faces; face++)
  {
    tets[face] = add_tet(hull_nodes[face][0],
			 hull_nodes[face][1],
			 hull_nodes[face][2],
			 new_point_index);
    if (new_cells) new_cells->push_back(tets[face]);
  }

  return new_point_index;
}



void
TetVolMesh::bisect_element(const Cell::index_type cell)
{
  synchronize(FACE_NEIGHBORS_E | EDGE_NEIGHBORS_E);
  int edge, face;
  vector<Edge::array_type> edge_nbrs(6);
  Node::array_type nodes;
  get_nodes(nodes,cell);
  // Loop through edges and create new nodes at center
  for (edge = 0; edge < 6; ++edge)
  {
    Point p;
    get_center(p, Edge::index_type(cell*6+edge));
    nodes.push_back(add_point(p));
    // Get all other tets that share an edge with this tet
    pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =
      all_edges_.equal_range(cell*6+edge);
    edge_nbrs[edge].insert(edge_nbrs[edge].end(), range.first, range.second);
  }

  // Get all other tets that share a face with this tet
  Face::array_type face_nbrs(4);
  for (face = 0; face < 4; ++face)
    get_neighbor(face_nbrs[face], Face::index_type(cell*4+face));

  // This is used below to weed out tets that have already been split
  set<Cell::index_type> done;
  done.insert(cell);
  
  // Vector of all tets that have been modified or added
  Cell::array_type tets;
  
  // Perform an 8:1 split on this tet
  tets.push_back(mod_tet(cell,nodes[4],nodes[6],nodes[5],nodes[0]));
  tets.push_back(add_tet(nodes[4], nodes[7], nodes[9], nodes[1]));
  tets.push_back(add_tet(nodes[7], nodes[5], nodes[8], nodes[2]));
  tets.push_back(add_tet(nodes[6], nodes[8], nodes[9], nodes[3]));
  tets.push_back(add_tet(nodes[4], nodes[9], nodes[8], nodes[6]));
  tets.push_back(add_tet(nodes[4], nodes[8], nodes[5], nodes[6]));
  tets.push_back(add_tet(nodes[4], nodes[5], nodes[8], nodes[7]));
  tets.push_back(add_tet(nodes[4], nodes[8], nodes[9], nodes[7]));

  // Perform a 4:1 split on tet sharing face 0
  if (face_nbrs[0] != -1)
  {
    Node::index_type opp = cells_[face_nbrs[0]];
    tets.push_back(mod_tet(face_nbrs[0]/4,nodes[7],nodes[8],nodes[9],opp));
    tets.push_back(add_tet(nodes[7],nodes[2],nodes[8],opp));
    tets.push_back(add_tet(nodes[8],nodes[3],nodes[9],opp));
    tets.push_back(add_tet(nodes[9],nodes[1],nodes[7],opp));
    done.insert(face_nbrs[0]/4);
  }
  // Perform a 4:1 split on tet sharing face 1
  if (face_nbrs[1] != -1)
  {
    Node::index_type opp = cells_[face_nbrs[1]];
    tets.push_back(mod_tet(face_nbrs[1]/4,nodes[5],nodes[6],nodes[8],opp));
    tets.push_back(add_tet(nodes[5],nodes[0],nodes[6],opp));
    tets.push_back(add_tet(nodes[6],nodes[3],nodes[8],opp));
    tets.push_back(add_tet(nodes[8],nodes[2],nodes[5],opp));
    done.insert(face_nbrs[1]/4);
  }
  // Perform a 4:1 split on tet sharing face 2
  if (face_nbrs[2] != -1)
  {
    Node::index_type opp = cells_[face_nbrs[2]];
    tets.push_back(mod_tet(face_nbrs[2]/4,nodes[4],nodes[9],nodes[6],opp));
    tets.push_back(add_tet(nodes[4],nodes[1],nodes[9],opp));
    tets.push_back(add_tet(nodes[9],nodes[3],nodes[6],opp));
    tets.push_back(add_tet(nodes[6],nodes[0],nodes[4],opp));
    done.insert(face_nbrs[2]/4);
  }
  // Perform a 4:1 split on tet sharing face 3
  if (face_nbrs[3] != -1)
  {
    Node::index_type opp = cells_[face_nbrs[3]];
    tets.push_back(mod_tet(face_nbrs[3]/4,nodes[4],nodes[5],nodes[7],opp));
    tets.push_back(add_tet(nodes[4],nodes[0],nodes[5],opp));
    tets.push_back(add_tet(nodes[5],nodes[2],nodes[7],opp));
    tets.push_back(add_tet(nodes[7],nodes[1],nodes[4],opp));
    done.insert(face_nbrs[3]/4);
  }
		   
  // Search every tet that shares an edge with the one we just split 8:1
  // If it hasnt been split 4:1 (because it shared a face) split it 2:1
  for (edge = 0; edge < 6; ++edge)
  {
    for (unsigned shared = 0; shared < edge_nbrs[edge].size(); ++shared)
    {
      // Edge index of tet that shares an edge
      Edge::index_type nedge = edge_nbrs[edge][shared];
      Cell::index_type ntet = nedge/6;
      // Check to only split tets that havent been split already
      if (done.find(ntet) == done.end())
      {	
	// Opposite edge index.  Opposite tet edges are: 0 & 4, 1 & 5, 2 & 3
	Edge::index_type oedge = (ntet*6+nedge%6+
				  (nedge%6>2?-1:1)*((nedge%6)/2==1?1:4));
	// Cell Indices of Tet that only shares one edge with tet we split 8:1
	pair<Node::index_type, Node::index_type> nnodes = Edge::edgei(nedge);
	pair<Node::index_type, Node::index_type> onodes = Edge::edgei(oedge);
	// Perform the 2:1 split
	tets.push_back(add_tet(nodes[4+edge], cells_[nnodes.first], 
			       cells_[onodes.second], cells_[onodes.first]));
	orient(tets.back());
	tets.push_back(mod_tet(ntet,nodes[4+edge], cells_[nnodes.second], 
			       cells_[onodes.second], cells_[onodes.first]));
	orient(tets.back());
	// dont think is necessasary, but make sure tet doesnt get split again
	done.insert(ntet);
      }
    }
  }  
}



TetVolMesh::Elem::index_type
TetVolMesh::mod_tet(Cell::index_type cell, 
		    Node::index_type a,
		    Node::index_type b,
		    Node::index_type c,
		    Node::index_type d)
{
  delete_cell_node_neighbors(cell);
  delete_cell_edges(cell);
  delete_cell_faces(cell);
  cells_[cell*4+0] = a;
  cells_[cell*4+1] = b;
  cells_[cell*4+2] = c;
  cells_[cell*4+3] = d;  
  create_cell_node_neighbors(cell);
  create_cell_edges(cell);
  create_cell_faces(cell);
  synchronized_ &= ~LOCATE_E;
  return cell;
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
