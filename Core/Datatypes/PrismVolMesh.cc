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
 *  PrismVolMesh.cc: Prism mesh with new design.
 *
 *  Written by:
 *   Allen Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2003 SCI Insititute
 *
 */

#include <Core/Datatypes/PrismVolMesh.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Plane.h>
#include <Core/Math/MusilRNG.h>
#include <algorithm>


namespace SCIRun {

using std::for_each;
using std::cerr;
using std::endl;
using std::copy;

Persistent* make_PrismVolMesh() {
  return scinew PrismVolMesh;
}

PersistentTypeID PrismVolMesh::type_id("PrismVolMesh", "Mesh",
				     make_PrismVolMesh);

const string
PrismVolMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name("PrismVolMesh");
  return name;
}


PrismVolMesh::PrismVolMesh() :
  points_(0),
  points_lock_("PrismVolMesh points_ fill lock"),
  cells_(0),
  cells_lock_("PrismVolMesh cells_ fill lock"),

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
  edge_lock_("PrismVolMesh edges_ fill lock"),

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
  face_lock_("PrismVolMesh faces_ fill lock"),

  node_neighbors_(0),
  node_neighbor_lock_("PrismVolMesh node_neighbors_ fill lock"),
  grid_(0),
  grid_lock_("PrismVolMesh grid_ fill lock"),
  synchronized_(CELLS_E | NODES_E)
{
}

PrismVolMesh::PrismVolMesh(const PrismVolMesh &copy):
  points_(copy.points_),
  points_lock_("PrismVolMesh points_ fill lock"),
  cells_(copy.cells_),
  cells_lock_("PrismVolMesh cells_ fill lock"),
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
  edge_lock_("PrismVolMesh edges_ fill lock"),
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
  face_lock_("PrismVolMesh edges_ fill lock"),
  node_neighbors_(0),
  node_neighbor_lock_("PrismVolMesh node_neighbors_ fill lock"),
  grid_(copy.grid_),
  grid_lock_("PrismVolMesh grid_ fill lock"),
  synchronized_(copy.synchronized_)
{
  synchronized_ &= ~EDGES_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~FACES_E;
  synchronized_ &= ~FACE_NEIGHBORS_E;
}

PrismVolMesh::~PrismVolMesh()
{
}


/* To generate a random point inside of a prism, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void
PrismVolMesh::get_random_point(Point &p, const Cell::index_type &ei,
			       int seed) const
{
  static MusilRNG rng;

  // get positions of the vertices
  Node::array_type ra;
  get_nodes(ra,ei);

  Vector v = Vector(0,0,0);

  double sum = 0;

  if( seed ) {
   MusilRNG rng1(seed);
   for( unsigned int i=0; i<PRISM_NNODES; i++ ) {
     const Point &p0 = point(ra[i]);
     const double w = rng1();
   
     v += p0.asVector() * w;
     sum += w;
   }
  } else {
    for( unsigned int i=0; i<PRISM_NNODES; i++ ) {
      const Point &p0 = point(ra[i]);
      const double w = rng();

      v += p0.asVector() * w;
      sum += w;
    }
  }

  p = (v / sum).asPoint();
}

BBox
PrismVolMesh::get_bounding_box() const
{
  //! TODO: This could be included in the synchronize scheme
  BBox result;

  Node::iterator ni, nie;
  begin(ni);
  end(nie);
  while (ni != nie) {
    const Point &p = point(*ni);
    result.extend(p);
    ++ni;
  }
  return result;
}


void
PrismVolMesh::transform(const Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr) {
    *itr = t.project(*itr);
    ++itr;
  }
  
  // Recompute grid.
  synchronized_ &= ~LOCATE_E;
}


void
PrismVolMesh::compute_faces()
{  
  face_lock_.lock();
  faces_.clear();
  all_faces_.clear();
  unsigned int num_cells = (cells_.size()) /  PRISM_NNODES *  PRISM_NFACES;
  for (unsigned int i = 0; i < num_cells; i++) {
    faces_.insert(i);
    all_faces_.insert(i);
  }

  synchronized_ |= FACES_E;
  synchronized_ |= FACE_NEIGHBORS_E;
  face_lock_.unlock();
}


void
PrismVolMesh::compute_edges()
{
  edge_lock_.lock();
  edges_.clear();
  all_edges_.clear();
  unsigned int num_cells = (cells_.size()) /  PRISM_NNODES *  PRISM_NEDGES;
  for (unsigned int i = 0; i < num_cells; i++) {
    edges_.insert(i);
    all_edges_.insert(i);
  }
  synchronized_ |= EDGES_E;
  synchronized_ |= EDGE_NEIGHBORS_E;
  edge_lock_.unlock();
}


void
PrismVolMesh::compute_node_neighbors()
{
  node_neighbor_lock_.lock();
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size());
  unsigned int num_cells = cells_.size();
  for (unsigned int i = 0; i < num_cells; i++)
    node_neighbors_[cells_[i]].push_back(i);

  synchronized_ |= NODE_NEIGHBORS_E;
  node_neighbor_lock_.unlock();
}


bool
PrismVolMesh::synchronize(unsigned int tosync)
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
PrismVolMesh::begin(PrismVolMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E, "Must call synchronize on mesh first");
  itr = 0;
}

void
PrismVolMesh::end(PrismVolMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E, "Must call synchronize on mesh first");
  itr = points_.size();
}

void
PrismVolMesh::size(PrismVolMesh::Node::size_type &s) const
{
  ASSERTMSG(synchronized_ & NODES_E, "Must call synchronize on mesh first");
  s = points_.size();
}

void
PrismVolMesh::begin(PrismVolMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E, "Must call synchronize on mesh first");
  itr = edges_.begin();
}

void
PrismVolMesh::end(PrismVolMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E, "Must call synchronize on mesh first");
  itr = edges_.end();
}

void
PrismVolMesh::size(PrismVolMesh::Edge::size_type &s) const
{
  ASSERTMSG(synchronized_ & EDGES_E, "Must call synchronize on mesh first");
  s = edges_.size();
}

void
PrismVolMesh::begin(PrismVolMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E, "Must call synchronize on mesh first");
  itr = faces_.begin();
}

void
PrismVolMesh::end(PrismVolMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E, "Must call synchronize on mesh first");
  itr = faces_.end();
}

void
PrismVolMesh::size(PrismVolMesh::Face::size_type &s) const
{
  ASSERTMSG(synchronized_ & FACES_E, "Must call synchronize on mesh first");
  s = faces_.size();
}

void
PrismVolMesh::begin(PrismVolMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & ELEMENTS_E, "Must call synchronize on mesh first");
  itr = 0;
}

void
PrismVolMesh::end(PrismVolMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & ELEMENTS_E, "Must call synchronize on mesh first");
  itr = cells_.size() / PRISM_NNODES;
}

void
PrismVolMesh::size(PrismVolMesh::Cell::size_type &s) const
{
  ASSERTMSG(synchronized_ & ELEMENTS_E, "Must call synchronize on mesh first");
  s = cells_.size() / PRISM_NNODES;
}


void
PrismVolMesh::create_cell_edges(Cell::index_type idx)
{
  //ASSERT(!is_frozen());
  edge_lock_.lock();
  const unsigned int base = idx * PRISM_NEDGES;
  for (unsigned int i=base; i<base+PRISM_NEDGES; ++i) {
    edges_.insert(i);
    all_edges_.insert(i);
  }
  edge_lock_.unlock();
}
      

void
PrismVolMesh::delete_cell_edges(Cell::index_type idx)
{
  //ASSERT(!is_frozen());
  edge_lock_.lock();
  const unsigned int base = idx * PRISM_NEDGES;
  for (unsigned int i=base; i<base+ PRISM_NEDGES; ++i) {
    //! If the Shared Edge Set is represented by the particular
    //! cell/edge index that is being recomputed, then
    //! remove it (and insert a non-recomputed edge if any left)
    bool shared_edge_exists = true;
    Edge::iterator shared_edge = edges_.find(i);
    // ASSERT guarantees edges were computed correctly for this cell
    ASSERT(shared_edge != edges_.end());
    if ((*shared_edge).index_ == i) {
      edges_.erase(shared_edge);
      shared_edge_exists = false;
    }
    
    Edge::HalfEdgeSet::iterator half_edge_to_delete = all_edges_.end();
    pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =
      all_edges_.equal_range(i);
    for (Edge::HalfEdgeSet::iterator e = range.first; e != range.second; ++e) {
      if ((*e).index_ == i) {
	half_edge_to_delete = e;
      }
      else if (!shared_edge_exists) {
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
PrismVolMesh::create_cell_faces(Cell::index_type idx)
{
  //ASSERT(!is_frozen());
  face_lock_.lock();
  const unsigned int base = idx * PRISM_NFACES;
  for (unsigned int i=base; i<base+PRISM_NFACES; i++) {
    faces_.insert(i);
    all_faces_.insert(i);
  }
  face_lock_.unlock();
}

void
PrismVolMesh::delete_cell_faces(Cell::index_type idx)
{
  //ASSERT(!is_frozen());
  face_lock_.lock();
  const unsigned int base = idx * PRISM_NFACES;
  for (unsigned int i=base; i<base+PRISM_NFACES; ++i) {
    // If the Shared Face Set is represented by the particular
    // cell/face index that is being recomputed, then
    // remove it (and insert a non-recomputed shared face if any exist)
    bool shared_face_exists = true;
    Face::FaceSet::iterator shared_face = faces_.find(i);
    ASSERT(shared_face != faces_.end());
    if ((*shared_face).index_ == i) {
      faces_.erase(shared_face);
      shared_face_exists = false;
    }
    
    Face::HalfFaceSet::iterator half_face_to_delete = all_faces_.end();
    pair<Face::HalfFaceSet::iterator, Face::HalfFaceSet::iterator> range =
      all_faces_.equal_range(i);
    for (Face::HalfFaceSet::iterator e = range.first; e != range.second; ++e) {
      if ((*e).index_ == i) {
	half_face_to_delete = e;
      }
      else if (!shared_face_exists) {
	faces_.insert((*e).index_);
	shared_face_exists = true;
      }
      if (half_face_to_delete != all_faces_.end() && shared_face_exists)
	break;
    }

    //! If this ASSERT is reached, it means that the faces
    //! were not computed correctlyfor this cell
    ASSERT(half_face_to_delete != all_faces_.end());
    all_faces_.erase(half_face_to_delete);
  }
  face_lock_.unlock();
}

void
PrismVolMesh::create_cell_node_neighbors(Cell::index_type idx)
{
  //ASSERT(!is_frozen());
  node_neighbor_lock_.lock();

  const unsigned int base = idx * PRISM_NNODES;

  for (unsigned int i=base; i<base+PRISM_NNODES; i++)
    node_neighbors_[cells_[i]].push_back(i);

  node_neighbor_lock_.unlock();
}

void
PrismVolMesh::delete_cell_node_neighbors(Cell::index_type idx)
{
  //ASSERT(!is_frozen());
  node_neighbor_lock_.lock();

  const unsigned int base = idx * PRISM_NNODES;

  for (unsigned int i=base; i<base+PRISM_NNODES; ++i) {

    const unsigned int n = cells_[i];

    vector<Cell::index_type>::iterator node_cells_end =
      node_neighbors_[n].end();

    vector<Cell::index_type>::iterator cell = node_neighbors_[n].begin();

    while (cell != node_cells_end && (*cell) != i)
      ++cell;

    //! ASSERT that the node_neighbors_ structure contains this cell
    ASSERT(cell != node_cells_end);

    node_neighbors_[n].erase(cell);
  }
  node_neighbor_lock_.unlock();      
}

//! Given two nodes (n0, n1), return all edge indexes that span those two nodes
bool
PrismVolMesh::is_edge(Node::index_type n0, Node::index_type n1,
		    Edge::array_type *array)
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on PrismVolMesh first.");
  edge_lock_.lock();
  cells_lock_.lock();

  //! Create a phantom cell with edge 0 being the one we're searching for
  const unsigned int fake_edge = cells_.size() /  PRISM_NNODES *  PRISM_NEDGES;
  vector<under_type>::iterator c0 = cells_.insert(cells_.end(),n0);
  vector<under_type>::iterator c1 = cells_.insert(cells_.end(),n1);

  //! Search the all_edges_ multiset for edges matching our fake_edge
  pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =
    all_edges_.equal_range(fake_edge);

  if (array) {
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

//! Given three nodes (n0, n1, n2), return all face indexes that
//! span those three nodes
bool
PrismVolMesh::is_face(Node::index_type n0,
		      Node::index_type n1, 
		      Node::index_type n2,
		      Face::array_type *array)
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on PrismVolMesh first.");
  face_lock_.lock();
  cells_lock_.lock();

  //! Create a phantom cell with face 3 being the one we're searching for
  const unsigned int fake_face = cells_.size() + 3;
  vector<under_type>::iterator c0 = cells_.insert(cells_.end(),n0);
  vector<under_type>::iterator c1 = cells_.insert(cells_.end(),n1);
  vector<under_type>::iterator c2 = cells_.insert(cells_.end(),n2);

  //! Search the all_face_ multiset for edges matching our fake_edge
  pair<Face::HalfFaceSet::const_iterator, Face::HalfFaceSet::const_iterator>
     range = all_faces_.equal_range(fake_face);

  if (array) {
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
  
//! Given four nodes (n0, n1, n2, n3), return all face indexes that
//! span those four nodes
bool
PrismVolMesh::is_face(Node::index_type n0,
		      Node::index_type n1,
		      Node::index_type n2,
		      Node::index_type n3,
		      Face::array_type *array)
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on PrismVolMesh first.");
  face_lock_.lock();
  cells_lock_.lock();

  //! Create a phantom cell with face 4 being the one we're searching for
  const unsigned int fake_face = cells_.size() + 4;
  vector<under_type>::iterator c0 = cells_.insert(cells_.end(),n0);
  vector<under_type>::iterator c1 = cells_.insert(cells_.end(),n1);
  vector<under_type>::iterator c2 = cells_.insert(cells_.end(),n2);
  vector<under_type>::iterator c3 = cells_.insert(cells_.end(),n3);

  //! Search the all_face_ multiset for edges matching our fake_edge
  pair<Face::HalfFaceSet::const_iterator, Face::HalfFaceSet::const_iterator>
     range = all_faces_.equal_range(fake_face);

  if (array) {
    array->clear();
    copy(range.first, range.second, array->end());
  }

  //! Delete the pahntom cell
  cells_.erase(c0);
  cells_.erase(c1);
  cells_.erase(c2);
  cells_.erase(c3);

  face_lock_.unlock();
  cells_lock_.unlock();
  return range.first != range.second;
}


void
PrismVolMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.resize(2);
  pair<Edge::index_type, Edge::index_type> edge = Edge::edgei(idx);
  array[0] = cells_[edge.first];
  array[1] = cells_[edge.second];
}


// Always returns nodes in counter-clockwise order
void
PrismVolMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  // Get the base cell index and the face offset
  const unsigned int offset = idx%PRISM_NFACES;
  const unsigned int base = idx / PRISM_NFACES * PRISM_NNODES;

  if( isTRI( offset ) )
    array.resize(3);
  else if( isQUAD( offset ) )
    array.resize(4);

  array[0] = cells_[base+PrismFaceTable[offset][0]];
  array[1] = cells_[base+PrismFaceTable[offset][1]];
  array[2] = cells_[base+PrismFaceTable[offset][2]];

  if( isQUAD( offset ) )
    array[3] = cells_[base+PrismFaceTable[offset][3]];
}


void
PrismVolMesh::get_nodes(Node::array_type &array, Cell::index_type idx) const
{
  array.resize(PRISM_NNODES);
  const unsigned int base = idx*PRISM_NNODES;
  for (int unsigned i=0; i<PRISM_NNODES; i++ )
    array[i] = cells_[base+i];
}

void
PrismVolMesh::set_nodes(Node::array_type &array, Cell::index_type idx)
{
  ASSERT(array.size() == PRISM_NNODES);

  if (synchronized_ & EDGES_E) delete_cell_edges(idx);
  if (synchronized_ & FACES_E) delete_cell_faces(idx);
  if (synchronized_ & NODE_NEIGHBORS_E) delete_cell_node_neighbors(idx);

  const unsigned int base = idx * PRISM_NNODES;

  for (unsigned int i=0; i<PRISM_NNODES; i++)
    cells_[base + i] = array[i];
  
  synchronized_ &= ~LOCATE_E;
  if (synchronized_ & EDGES_E) create_cell_edges(idx);
  if (synchronized_ & FACES_E) create_cell_faces(idx);
  if (synchronized_ & NODE_NEIGHBORS_E) create_cell_node_neighbors(idx);

}

void
PrismVolMesh::get_edges(Edge::array_type &/*array*/,
			Node::index_type /*idx*/) const
{
  ASSERTFAIL("Not implemented yet");
}


void
PrismVolMesh::get_edges(Edge::array_type &/*array*/,
			Face::index_type /*idx*/) const
{
  ASSERTFAIL("Not implemented yet");
}


void
PrismVolMesh::get_edges(Edge::array_type &array, Cell::index_type idx) const
{
  array.resize(PRISM_NEDGES);
  const unsigned int base = idx * PRISM_NEDGES;
  for (int i=0; i<PRISM_NEDGES; i++)
    array[base + i] = i;
}



void
PrismVolMesh::get_faces(Face::array_type &/*array*/,
			Node::index_type /*idx*/) const
{
  ASSERTFAIL("Not implemented yet");
}

void
PrismVolMesh::get_faces(Face::array_type &/*array*/,
			Edge::index_type /*idx*/) const
{
  ASSERTFAIL("Not implemented yet");
}


void
PrismVolMesh::get_faces(Face::array_type &array, Cell::index_type idx) const
{
  array.resize(PRISM_NFACES);
  const unsigned int base = idx * PRISM_NFACES;
  for (unsigned int i=0; i<PRISM_NFACES; i++)
    array[i] = base + i;
}

void
PrismVolMesh::get_cells(Cell::array_type &array, Edge::index_type idx) const
{
  pair<Edge::HalfEdgeSet::const_iterator,
       Edge::HalfEdgeSet::const_iterator> range = all_edges_.equal_range(idx);

  //! ASSERT that this cell's edges have been computed
  ASSERT(range.first != range.second);

  array.clear();
  while (range.first != range.second) {
    array.push_back((*range.first)/ PRISM_NEDGES);
    ++range.first;
  }
}


void
PrismVolMesh::get_cells(Cell::array_type &array, Face::index_type idx) const
{
  pair<Face::HalfFaceSet::const_iterator, 
       Face::HalfFaceSet::const_iterator> range = all_faces_.equal_range(idx);

  //! ASSERT that this cell's faces have been computed
  ASSERT(range.first != range.second);

  array.clear();
  while (range.first != range.second) {
    array.push_back((*range.first)/PRISM_NFACES);
    ++range.first;
  }
}
  

void
PrismVolMesh::get_cells(Cell::array_type &array, Node::index_type idx) const
{
  ASSERTMSG(is_frozen(),"only call get_cells with a node index if frozen!!");
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E, 
	    "Must call synchronize NODE_NEIGHBORS_E on PrismVolMesh first.");
  array.clear();
  for (unsigned int i=0; i<node_neighbors_[idx].size(); ++i)
    array.push_back(node_neighbors_[idx][i]/PRISM_NNODES);
}


//! this is a bad hack for existing code that calls this function
//! call the one below instead
bool
PrismVolMesh::get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
			   Face::index_type idx) const
{
  ASSERT(idx/PRISM_NFACES == from);
  Face::index_type neigh;
  bool ret_val = get_neighbor(neigh, idx);
  neighbor.index_ = neigh.index_ / PRISM_NFACES;
  return ret_val;
}


//! given a face index, return the face index that spans the same nodes
bool
PrismVolMesh::get_neighbor(Face::index_type &neighbor,
			   Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACE_NEIGHBORS_E,
	    "Must call synchronize FACE_NEIGHBORS_E on PrismVolMesh first.");
  pair<Face::HalfFaceSet::const_iterator,
       Face::HalfFaceSet::const_iterator> range = all_faces_.equal_range(idx);

  // ASSERT that this face was computed
  ASSERT(range.first != range.second);

  // Cell has no neighbor
  Face::HalfFaceSet::const_iterator second = range.first;
  if (++second == range.second) {
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
PrismVolMesh::get_neighbors(Cell::array_type &array,
			    Cell::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACE_NEIGHBORS_E,
	    "Must call synchronize FACE_NEIGHBORS_E on PrismVolMesh first.");
  array.clear();
  Face::index_type face;
  const unsigned int base = idx*PRISM_NFACES;
  for (unsigned int i=0; i<PRISM_NFACES; i++) {
    face.index_ = base + i;
    pair<const Face::HalfFaceSet::const_iterator,
         const Face::HalfFaceSet::const_iterator> range =
      all_faces_.equal_range(face);
    for (Face::HalfFaceSet::const_iterator iter = range.first;
	 iter != range.second; ++iter)
      if (*iter != face.index_)
	array.push_back(*iter/PRISM_NFACES );
  } 
}

void
PrismVolMesh::get_neighbors(Node::array_type &array,
			    Node::index_type idx) const
{
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E, 
	    "Must call synchronize NODE_NEIGHBORS_E on PrismVolMesh first.");
  array.clear();
  set<int> inserted;
  for (unsigned int i=0; i<node_neighbors_[idx].size(); i++) {
    const unsigned int base =
      node_neighbors_[idx][i]/PRISM_NNODES*PRISM_NNODES;
    for (int c=base; c<base+PRISM_NNODES; c++) {
      if (cells_[c] != idx && inserted.find(cells_[c]) == inserted.end()) {
	inserted.insert(cells_[c]);
	array.push_back(cells_[c]);
      }
    }
  }
}


void
PrismVolMesh::get_center(Point &p, Node::index_type idx) const
{
  p = points_[idx];
}

void
PrismVolMesh::get_center(Point &p, Edge::index_type idx) const
{
  Node::array_type arr;

  get_nodes(arr, idx);
  get_center(p, arr);
}


void
PrismVolMesh::get_center(Point &p, Face::index_type idx) const
{
  Node::array_type arr;

  get_nodes(arr, idx);
  get_center(p, arr);
}


void
PrismVolMesh::get_center(Point &p, Cell::index_type idx) const
{
  Node::array_type arr;

  get_nodes(arr, idx);
  get_center(p, arr);
}

void
PrismVolMesh::get_center(Point &p, Node::array_type& arr) const
{
  Vector v(0,0,0);

  for( unsigned int i=0; i<arr.size(); i++ ) {
    const Point &p0 = point(arr[i]);
    v += p0.asVector();
  }

  p = (v / (double) arr.size()).asPoint();
}


bool
PrismVolMesh::locate(Node::index_type &loc, const Point &p)
{
  Cell::index_type ci;
  if (locate(ci, p)) {// first try the fast way.
    Node::array_type nodes;
    get_nodes(nodes, ci);

    Point ptmp;
    double mindist;
    for (int i=0; i<PRISM_NNODES; i++) {
      get_center(ptmp, nodes[i]);
      double dist = (p - ptmp).length2();
      if (i == 0 || dist < mindist) {
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
    while (bi != ei) {
      Point c;
      get_center(c, *bi);
      const double dist = (p - c).length2();
      if (!found_p || dist < mindist) {
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
PrismVolMesh::locate(Edge::index_type &edge, const Point &p)
{
  bool found_p = false;
  double mindist;
  Edge::iterator bi; begin(bi);
  Edge::iterator ei; end(ei);
  while (bi != ei) {
    Point c;
    get_center(c, *bi);
    const double dist = (p - c).length2();
    if (!found_p || dist < mindist) {
      mindist = dist;
      edge = *bi;
      found_p = true;
    }
    ++bi;
  }
  return found_p;
}


bool
PrismVolMesh::locate(Face::index_type &face, const Point &p)
{
  bool found_p = false;
  double mindist;
  Face::iterator bi; begin(bi);
  Face::iterator ei; end(ei);
  while (bi != ei) {
    Point c;
    get_center(c, *bi);
    const double dist = (p - c).length2();
    if (!found_p || dist < mindist) {
      mindist = dist;
      face = *bi;
      found_p = true;
    }
    ++bi;
  }
  return found_p;
}


bool
PrismVolMesh::locate(Cell::index_type &cell, const Point &p)
{
  if (!(synchronized_ & LOCATE_E))
    synchronize(LOCATE_E);
  ASSERT(grid_.get_rep());

  LatVolMeshHandle mesh = grid_->get_typed_mesh();
  LatVolMesh::Cell::index_type ci;
  if (!mesh->locate(ci, p)) { return false; }
  vector<Cell::index_type> v = grid_->value(ci);
  vector<Cell::index_type>::iterator iter = v.begin();
  while (iter != v.end()) {
    if (inside(*iter, p)) {
      cell = *iter;
      return true;
    }
    ++iter;
  }
  return false;
}


void
PrismVolMesh::get_weights(const Point &p,
			  Cell::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p)) {
    l.push_back(idx);
    w.push_back(1.0);
  }
}


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
PrismVolMesh::polygon_area(const Node::array_type &ni, const Vector N) const
{
  double area = 0;
  double an, ax, ay, az;  // abs value of normal and its coords
  int   coord;           // coord to ignore: 1=x, 2=y, 3=z
  int   i, j, k;         // loop indices
  const unsigned int n = ni.size();

  // select largest abs coordinate to ignore for projection
  ax = (N.x()>0 ? N.x() : -N.x());     // abs x-coord
  ay = (N.y()>0 ? N.y() : -N.y());     // abs y-coord
  az = (N.z()>0 ? N.z() : -N.z());     // abs z-coord

  coord = 3;                     // ignore z-coord
  if (ax > ay) {
    if (ax > az) coord = 1;      // ignore x-coord
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

// The volume x 6, used by get_weights to compute barycentric coordinates.
static double
prism_vol6(const Point &p1, const Point &p2, const Point &p3, const Point &p4)
{
  return fabs( Dot(Cross(p2-p1,p3-p1),p4-p1) );
}

static double
prism_area2(const Point &p1, const Point &p2, const Point &p3)
{
  return Cross(p1-p2,p3-p1).length();
}

/* This code is based on the paper by
   Mark Meyer, Haeyoung Lee, Alan Barr, and Mathieu Desbrun.
   Generalized barycentric coordinates on irregular polygons.
   Journal of graphics tools, 7(1):13-22, 2002
*/
void
PrismVolMesh::get_weights(const Point &pt,
			  Node::array_type &nodes, vector<double> &w)
{
  synchronize (Mesh::FACES_E);
  Cell::index_type cell;

  if (locate(cell, pt))
  {
    unsigned int f,i,j,k;
    double total = 0.0;

    w.resize(PRISM_NNODES);

    get_nodes(nodes,cell);

    vector<Point> p(PRISM_NNODES);

    for( i=0; i<PRISM_NNODES; i++ ) {
      w[i] = 0;      
      p[i] = point(nodes[i]);
    }

    // Make the face check first. This assures that none of the tet
    // volumes calculated latter will be zero and cause a divide by
    // zero error. If a volume is zero the barycentric coordinates
    // are based on the face instead.
    for( f=0; f<PRISM_NFACES; f++ ) {
      const unsigned int *fTable = PrismFaceTable[f];

      // Special case point is on a face.
      if( prism_vol6( pt,
		      p[fTable[0]],
		      p[fTable[1]],
		      p[fTable[2]] ) < 1.0e-8 )
	break;
    }

    // Determine volume of the tets.
    if( f == PRISM_NFACES ) {
      for( i=0; i<PRISM_NNODES; i++ ) {
	const unsigned int *nTable = PrismNodeNeighborTable[i];

	// Get the volume of the tet formed by the node and
	// its three neighbors. 
	w[i] = prism_vol6( p[i],
			   p[nTable[0]],
			   p[nTable[1]],
			   p[nTable[2]] );

	// Get the volume of the tet formed by the point, node and
	// two of the nodes neighbors.
	// Note: Each node has exactly three neighbors.
	for( j=0,k=1; j<3; j++, k++ )
	  w[i] /= prism_vol6( pt,
			      p[i],
			      p[nTable[j%3]],
			      p[nTable[k%3]] );

	total += w[i];
      }
    } else {
      const unsigned int *fTable = PrismFaceTable[f];

      // Special case point is on a face.
      if( isTRI( f ) ) {
	// On a triangular face.

	// Fast calculation of barycentric coordinates which does not
	// have a division so if the point is on an edge there is no
	// divide by zero error.
	for( i=0,j=1,k=TRI_NNODES-1; i<TRI_NNODES; i++, j++, k++ ) {
	  w[fTable[i]] =
	    prism_area2( pt,
			 p[fTable[j%TRI_NNODES]],
			 p[fTable[k%TRI_NNODES]] );
	  
	  total += w[fTable[i]];
	}
      } else if( isQUAD( f ) ) {
	// On a quad face.
	vector< double > area(4);

	// Precalculate the areas formed by the point and two neighboring
	// nodes. If the area is zero the point is on an edge.
	for( i=0,j=1; i<QUAD_NNODES; i++,j++ ) {
	  area[i] = prism_area2(pt,
				p[fTable[i]],
				p[fTable[j%QUAD_NNODES]]);
	  
	  // Special case point is on an edge.
	  if( area[i] < 1.0e-8 )
	    break;
	}
	
	if( i == QUAD_NNODES ) {
	  // Determine area of the triangles.
	  for( i=0,j=1,k=QUAD_NNODES-1; i<QUAD_NNODES; i++,j++,k++ ) {
	    // Get the area of the triangle formed by the node and
	    // its two neighbors and divide it by the two triangles
	    // formed by the point, node and one of the nodes neighbors.
	    w[fTable[i]] =
	      prism_area2(p[fTable[i]],
			  p[fTable[j%QUAD_NNODES]],
			  p[fTable[k%QUAD_NNODES]]) /
	      (area[i] * area[k%QUAD_NNODES]);

	    total += w[fTable[i]];
	  }
	} else {
	  // Special case point is on an edge so bilenar interpolation
	  w[fTable[i]] =
	    (pt                      -
	     p[fTable[j%QUAD_NNODES]]).safe_normalize() /
	    (p[fTable[i]] -
	     p[fTable[j%QUAD_NNODES]]).safe_normalize();

	  w[fTable[j%QUAD_NNODES]] = 1.0 - w[fTable[i]];

	  // Since it is bilinear no further calculations are needed.
	  return;
	}	
      } 
    }

    total = 1.0 / total;

    for( i=0; i<PRISM_NNODES; i++ )
      w[i] *= total; 
  }
}


void
PrismVolMesh::compute_grid()
{
  grid_lock_.lock();
//if (grid_.get_rep() != 0) {grid_lock_.unlock(); return;} // only create once.

  BBox bb = get_bounding_box();
  if (!bb.valid()) { grid_lock_.unlock(); return; }
  // cubed root of number of cells to get a subdivision ballpark
  const double one_third = 1.L/3.L;
  Cell::size_type csize;  size(csize);
  const int s = ((int)ceil(pow((double)csize , one_third))) / 2 + 2;
  const Vector cell_epsilon = bb.diagonal() * (0.01 / s);
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
  while(ci != cie) {
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
  synchronized_ |= LOCATE_E;
  grid_lock_.unlock();
}


void
PrismVolMesh::orient(Cell::index_type idx) {
  Point center;
  get_center(center, idx);

  Face::array_type faces;
  get_faces(faces, idx);

  for (unsigned int i=0; i<PRISM_NFACES; i++) {
    Node::array_type ra;
    get_nodes(ra, faces[i]);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);

    const Vector v0(p0 - p1), v1(p2 - p1);
    const Vector normal = Cross(v0, v1);
    const Vector off1(center - p1);

    double dotprod = Dot(off1, normal);

    if( fabs( dotprod ) < 1.0e-8 ) {
      cerr << "Warning cell " << idx << " face " << i;
      cerr << " is malformed " << endl;
    }
  }
}


bool
PrismVolMesh::inside(Cell::index_type idx, const Point &p)
{
  Point center;
  get_center(center, idx);

  Face::array_type faces;
  get_faces(faces, idx);

  for (unsigned int i=0; i<PRISM_NFACES; i++) {
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

    if( Dot(off1, normal) < 0.0) {
      cerr << "Negative Face Normal " << i << endl;
      cerr << center << endl;
      cerr << p  << endl;
      cerr << p0 << endl;
      cerr << p1 << endl;
      cerr << p2 << endl;
      cerr << normal << endl;
    }

    // If orientated correctly the second dot product is not needed.
    // Only need to check to see if the sign is negitive.
    if (dotprod * Dot(off1, normal) < 0.0)
      return false;
  }
  return true;
}


PrismVolMesh::Node::index_type
PrismVolMesh::add_find_point(const Point &p, double err)
{
  Node::index_type i;
  if (locate(i, p) && (points_[i] - p).length2() < err)
    return i;
  else {
    points_.push_back(p);
    if (synchronized_ & NODE_NEIGHBORS_E)
      node_neighbors_.push_back(vector<Cell::index_type>());
    return points_.size() - 1;
  }
}


PrismVolMesh::Elem::index_type
PrismVolMesh::add_prism(Node::index_type a, Node::index_type b, 
			Node::index_type c, Node::index_type d,
			Node::index_type e, Node::index_type f)
{
  const unsigned int idx = cells_.size() / PRISM_NNODES;
  cells_.push_back(a);
  cells_.push_back(b);
  cells_.push_back(c);
  cells_.push_back(d);
  cells_.push_back(e);
  cells_.push_back(f);

  if (synchronized_ & NODE_NEIGHBORS_E) create_cell_node_neighbors(idx);
  if (synchronized_ & EDGES_E) create_cell_edges(idx);
  if (synchronized_ & FACES_E) create_cell_faces(idx);
  synchronized_ &= ~LOCATE_E;

  return idx; 
}



PrismVolMesh::Node::index_type
PrismVolMesh::add_point(const Point &p)
{
  points_.push_back(p);
  if (synchronized_ & NODE_NEIGHBORS_E)
    node_neighbors_.push_back(vector<Cell::index_type>());
  return points_.size() - 1;
}


PrismVolMesh::Elem::index_type
PrismVolMesh::add_prism(const Point &p0, const Point &p1, const Point &p2,
			const Point &p3, const Point &p4, const Point &p5)
{
  return add_prism(add_find_point(p0), add_find_point(p1), 
		   add_find_point(p2), add_find_point(p3), 
		   add_find_point(p4), add_find_point(p5));
}


PrismVolMesh::Elem::index_type
PrismVolMesh::add_elem(Node::array_type a)
{
  ASSERT(a.size() == PRISM_NNODES);

  const unsigned int idx = cells_.size() / PRISM_NNODES;
 
  for (unsigned int n = 0; n < a.size(); n++)
    cells_.push_back(a[n]);

  if (synchronized_ & NODE_NEIGHBORS_E) create_cell_node_neighbors(idx);
  if (synchronized_ & EDGES_E) create_cell_edges(idx);
  if (synchronized_ & FACES_E) create_cell_faces(idx);
  synchronized_ &= ~LOCATE_E;

  return idx;
}


void
PrismVolMesh::delete_cells(set<int> &to_delete)
{
  vector<under_type> old_cells = cells_;
  unsigned int i = 0;

  cells_.clear();
  cells_.reserve(old_cells.size() - to_delete.size()*PRISM_NNODES);

  for (set<int>::iterator deleted=to_delete.begin();
       deleted!=to_delete.end(); deleted++) {
    for (;i < *deleted; i++) {
      const unsigned int base = i * PRISM_NNODES;
      for (unsigned int c=base; c<base+PRISM_NNODES; c++)
	cells_.push_back(old_cells[c]);
    }

    ++i;
  }

  for (; i < (unsigned int)(old_cells.size()/PRISM_NNODES); i++) {
    const unsigned int base = i * PRISM_NNODES;
    for (unsigned int c=base; c<base+PRISM_NNODES; c++)
      cells_.push_back(old_cells[c]);
  }  
}


PrismVolMesh::Elem::index_type
PrismVolMesh::mod_prism(Cell::index_type idx, 
			Node::index_type a,
			Node::index_type b,
			Node::index_type c,
			Node::index_type d,
			Node::index_type e,
			Node::index_type f)
{
  if (synchronized_ & NODE_NEIGHBORS_E) delete_cell_node_neighbors(idx);
  if (synchronized_ & EDGES_E) delete_cell_edges(idx);
  if (synchronized_ & FACES_E) delete_cell_faces(idx);
  const unsigned int base = idx * PRISM_NNODES;
  cells_[base+0] = a;
  cells_[base+1] = b;
  cells_[base+2] = c;
  cells_[base+3] = d;  
  cells_[base+4] = e;  
  cells_[base+5] = f;  
  if (synchronized_ & NODE_NEIGHBORS_E) create_cell_node_neighbors(idx);
  if (synchronized_ & EDGES_E) create_cell_edges(idx);
  if (synchronized_ & FACES_E) create_cell_faces(idx);
  synchronized_ &= ~LOCATE_E;
  return idx;
}


#define PRISM_VOL_MESH_VERSION 1

void
PrismVolMesh::io(Piostream &stream)
{
  const int version = stream.begin_class(type_name(-1),
					 PRISM_VOL_MESH_VERSION);
  Mesh::io(stream);

  SCIRun::Pio(stream, points_);
  SCIRun::Pio(stream, cells_);
  if (version == 1) {
    vector<int> neighbors;
    SCIRun::Pio(stream, neighbors);
  }

  stream.end_class();
}

const TypeDescription*
PrismVolMesh::get_type_description() const
{
  return SCIRun::get_type_description((PrismVolMesh *)0);
}

const TypeDescription*
get_type_description(PrismVolMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
    td = scinew TypeDescription("PrismVolMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");

  return td;
}

const TypeDescription*
get_type_description(PrismVolMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
    td = scinew TypeDescription("PrismVolMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");

  return td;
}

const TypeDescription*
get_type_description(PrismVolMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
    td = scinew TypeDescription("PrismVolMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");

  return td;
}

const TypeDescription*
get_type_description(PrismVolMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
    td = scinew TypeDescription("PrismVolMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");

  return td;
}

const TypeDescription*
get_type_description(PrismVolMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
    td = scinew TypeDescription("PrismVolMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");

  return td;
}


} // namespace SCIRun
