/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#ifdef __ECC
  all_edges_(edge_hasher_),
  edges_(edge_hasher_),
#else
  edge_comp_(cells_),
  all_edges_(100,edge_hasher_,edge_comp_),
  edges_(100,edge_hasher_,edge_comp_),
#endif // ifdef __ECC
#else // ifdef HAVE_HASH_SET
  all_edges_(edge_comp_),
  edges_(edge_comp_),
#endif // ifdef HAVE_HASH_SET

  edge_lock_("TetVolMesh edges_ fill lock"),

  //! Unique Faces
#ifdef HAVE_HASH_SET
  face_hasher_(cells_),
#ifdef __ECC
  all_faces_(face_hasher_),
  faces_(face_hasher_),
#else
  face_comp_(cells_),
  all_faces_(100,face_hasher_,face_comp_),
  faces_(100,face_hasher_,face_comp_),
#endif // ifdef __ECC
#else // ifdef HAVE_HASH_SET
  all_faces_(face_comp_),
  faces_(face_comp_),
#endif // ifdef HAVE_HASH_SET

  face_lock_("TetVolMesh faces_ fill lock"),

  node_neighbors_(0),
  node_neighbor_lock_("TetVolMesh node_neighbors_ fill lock"),
  grid_(0),
  grid_lock_("TetVolMesh grid_ fill lock"),
  locate_cache_(0),
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
#ifdef __ECC
  all_edges_(edge_hasher_),
  edges_(edge_hasher_),
#else
  edge_comp_(cells_),
  all_edges_(100,edge_hasher_,edge_comp_),
  edges_(100,edge_hasher_,edge_comp_),
#endif // ifdef __ECC
#else // ifdef HAVE_HASH_SET
  all_edges_(edge_comp_),
  edges_(edge_comp_),
#endif // ifdef HAVE_HASH_SET

  edge_lock_("TetVolMesh edges_ fill lock"),

#ifdef HAVE_HASH_SET
  face_hasher_(cells_),
#ifdef __ECC
  all_faces_(face_hasher_),
  faces_(face_hasher_),
#else
  face_comp_(cells_),
  all_faces_(100,face_hasher_,face_comp_),
  faces_(100,face_hasher_,face_comp_),
#endif // ifdef __ECC
#else // ifdef HAVE_HASH_SET
  all_faces_(face_comp_),
  faces_(face_comp_),
#endif // ifdef HAVE_HASH_SET

  face_lock_("TetVolMesh edges_ fill lock"),

  node_neighbors_(0),
  node_neighbor_lock_("TetVolMesh node_neighbors_ fill lock"),
  grid_(copy.grid_),
  grid_lock_("TetVolMesh grid_ fill lock"),
  locate_cache_(0),
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
  Node::array_type ra;
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

  grid_lock_.lock();
  if (grid_.get_rep()) { grid_->transform(t); }
  grid_lock_.unlock();
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
  //faces_.resize((unsigned)(num_cells * 1.25));
  //all_faces_.resize((unsigned)(num_cells * 1.25));
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
  //  edges_.resize((unsigned)(num_cells * 1.25));
  //all_edges_.resize((unsigned)(num_cells * 1.25));
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
  for (unsigned int i = c*6; i < c*6+6; ++i)
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
  for (unsigned int i = c*6; i < c*6+6; ++i)
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
  for (unsigned int i = c*4; i < c*4+4; ++i)
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
  for (unsigned int i = c*4; i < c*4+4; ++i)
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
  for (unsigned int i = c*4; i < c*4+4; ++i)
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
  for (unsigned int i = c*4; i < c*4+4; ++i)
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
TetVolMesh::is_edge(const Node::index_type n0, const Node::index_type n1,
                    Edge::array_type *array)
{
  ASSERTMSG(synchronized_ & EDGES_E, "Must call synchronize EDGES_E on TetVolMesh first.");
  edge_lock_.lock();
  cells_lock_.lock();
  //! Create a phantom cell with edge 0 being the one we're searching for
  const unsigned int fake_edge = cells_.size() / 4 * 6;
  const unsigned int fake_cell = cells_.size() / 4;
  cells_.push_back(n0);
  cells_.push_back(n1);
  cells_.push_back(n0);
  cells_.push_back(n1);
                                                                                  
  //! Search the all_edges_ multiset for edges matching our fake_edge
  pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =  all_edges_.equal_range(fake_edge);

  bool ret_val = false;
  if(array)  array->clear();
  Edge::HalfEdgeSet::iterator iter = range.first;    
  while(iter != range.second) {
    if ((*iter)/6 != fake_cell) {
      if (array) {
        array->push_back(*iter);
      }
      ret_val = true;
    }
    ++iter;
  }
                                                                                  
  //! Delete the phantom cell
  cells_.resize(cells_.size()-4);
  
  edge_lock_.unlock();
  cells_lock_.unlock();
  return ret_val;
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
  array.resize(2);
  pair<Edge::index_type, Edge::index_type> edge = Edge::edgei(idx);
  array[0] = cells_[edge.first];
  array[1] = cells_[edge.second];
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
  array.resize(4);
  for (int i = 0; i < 4; i++)
  {
    array[i] = cells_[idx*4+i];
  }
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
TetVolMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
{
  //  ASSERTFAIL("Not implemented correctly");
  //#if 0
  array.clear();    
  static int table[4][3] =
  {
    {4, 3, 5},
    {1, 2, 4},
    {0, 2, 5},
    {1, 0, 3}
  };

  const int base = idx / 4 * 6;
  const int off = idx % 4;
  array.push_back(base + table[off][0]);
  array.push_back(base + table[off][1]);
  array.push_back(base + table[off][2]);
  //#endif
}


void
TetVolMesh::get_edges(Edge::array_type &array, Cell::index_type idx) const
{
  array.resize(6);
  for (int i = 0; i < 6; i++)
  {
    array[i] = idx * 6 + i;
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
  array.resize(4);
  for (int i = 0; i < 4; i++)
  {
    array[i] = idx * 4 + i;
  }
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
    array.push_back((*range.first)/6);
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


bool
TetVolMesh::split_cell_at_boundary(Cell::array_type &new_tets, 
				   Node::index_type &ni,
				   Cell::index_type ci, Face::index_type fi)
{
  new_tets.resize(3);
  Point face_center;
  get_center(face_center, fi);
  ni = add_point(face_center);

  Node::array_type face_nodes;
  get_nodes(face_nodes, fi);

  Node::index_type piv_ni = cells_[(int)fi];

  new_tets[0] = ci;
  mod_tet(ci, ni, piv_ni, face_nodes[0], face_nodes[1]);
  new_tets[1] = add_tet(ni, piv_ni, face_nodes[2], face_nodes[0]);
  new_tets[2] = add_tet(ni, piv_ni, face_nodes[1], face_nodes[2]);
  return true;
}

bool
TetVolMesh::split_2_to_3(Cell::array_type &new_tets, Node::index_type &c1_node,
			 Node::index_type &c2_node, Cell::index_type c1, 
			 Cell::index_type c2, Face::index_type c1face)
{
  new_tets.resize(3);
  
  Node::array_type face_nodes;
  get_nodes(face_nodes, c1face);

  c1_node = cells_[(int)c1face];

  Face::index_type c2face;
  get_neighbor(c2face, c1face);

  c2_node = cells_[(int)c2face];

  // FIX_ME needs to make sure that the ray between c1_node, and c2_node
  //        intersects the face shared by the tets.

  mod_tet(c1, c2_node, c1_node, face_nodes[0], face_nodes[1]);
  mod_tet(c2, c1_node, c2_node, face_nodes[0], face_nodes[2]);
  new_tets[0] = c1;
  new_tets[1] = c2;
  new_tets[2] = add_tet(c2_node, c1_node, face_nodes[1], face_nodes[2]);
  return true;
}

bool
TetVolMesh::get_edge(Edge::index_type &ei, Cell::index_type ci, 
		     Node::index_type n1, Node::index_type n2) const 
{
  Edge::index_type ebase = ci * 6;
  for (int i = 0; i < 6; i++) {
    pair<Node::index_type, Node::index_type> enodes = Edge::edgei(ebase + i);
    if ((cells_[enodes.first] == n1 && cells_[enodes.second] == n2) || 
	(cells_[enodes.first] == n2 && cells_[enodes.second] == n1)) {
      ei = ebase + i;
      return true;
    }
  }
  return false;
}


//! Given 3 tets that share an edge exclusively, combine them into 2 
//! tets that share a face.  This call orphans a cell index, which must be
//! deleted later.  use delete_cells with removed added to the set targeted
//! deletion.
bool
TetVolMesh::combine_3_to_2(Cell::index_type &removed,
			   Edge::index_type shared_edge)
{
  Node::array_type extrema;
  Edge::array_type edges;
  get_nodes(extrema, shared_edge);

  //! Search the all_edges_ multiset for edges matching our fake_edge
  pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =
    all_edges_.equal_range(shared_edge);

  Edge::HalfEdgeSet::iterator edge_iter = range.first;

  edges.push_back(*edge_iter++);
  edges.push_back(*edge_iter++);
  edges.push_back(*edge_iter);

  set<unsigned int, less<unsigned int> > ord_cells;
  ord_cells.insert(edges[0] / 6);
  ord_cells.insert(edges[1] / 6);
  ord_cells.insert(edges[2] / 6);
  set<unsigned int, less<unsigned int> >::iterator iter = ord_cells.begin();
  
  Cell::index_type c1 = *iter++;
  Cell::index_type c2 = *iter++;
  Cell::index_type c3 = *iter;

  // get the 3 nodes that are not on the shared edge
  Node::array_type opp0;
  Node::array_type opp1;
  Node::array_type opp2;

  get_nodes(opp0, Edge::opposite_edge(edges[0]));
  get_nodes(opp1, Edge::opposite_edge(edges[1]));
  get_nodes(opp2, Edge::opposite_edge(edges[2]));
  
  // filter out duplicates
  set<unsigned int, less<unsigned int> > shared_face;
  shared_face.insert(opp0[0]);
  shared_face.insert(opp0[1]);
  shared_face.insert(opp1[0]);
  shared_face.insert(opp1[1]);
  shared_face.insert(opp2[0]);
  shared_face.insert(opp2[1]);
  
  ASSERT(shared_face.size() == 3);

  iter = shared_face.begin();
  Node::array_type face;
  face.resize(3);
  face[0] = *iter++;
  face[1] = *iter++;
  face[2] = *iter++;

  // the cell index that is orphaned, needs to be added to the set for 
  // later deletion outside of this call.
  removed = c3;

  delete_cell_node_neighbors(removed);
  delete_cell_edges(removed);
  delete_cell_faces(removed);
  
  // FIX_ME needs to make sure that the ray between c1_node, and c2_node
  //        intersects the face shared by the tets.

  mod_tet(c1, extrema[0], face[0], face[1], face[2]);
  mod_tet(c2, extrema[1], face[1], face[0], face[2]);
  return true;
}

//! Undoes a previous center split.
bool
TetVolMesh::combine_4_to_1_cell(Cell::array_type &tets, 
				set<unsigned int> &removed_tets,
				set<unsigned int> &removed_nodes) 
{
  // the center node gets removed, it is at index 3 for each cell.
  unsigned c0, c1, c2, c3;
  c0 = tets[0]*4;
  c1 = tets[1]*4;
  c2 = tets[2]*4;
  c3 = tets[3]*4;

  // Sanity check that that we are removing the right node.
  if ((cells_[c0 + 3] == cells_[c1 + 3]) && 
      (cells_[c0 + 3] == cells_[c2 + 3]) && 
      (cells_[c0 + 3] == cells_[c3 + 3])) 
  {
    removed_nodes.insert(cells_[c0+3]);
    removed_tets.insert(tets[1]);
    removed_tets.insert(tets[2]);
    removed_tets.insert(tets[3]);
    // clean up to be removed tets
    for (int i = 1; i < 4; i++) {
      delete_cell_node_neighbors(tets[i]);
      delete_cell_edges(tets[i]);
      delete_cell_faces(tets[i]);
    }
    
    // redefine the remaining tet.
    // get the 4 unique nodes that make up the new single tet.
    set<unsigned int> new_tet;
    new_tet.insert(cells_[c0]);
    new_tet.insert(cells_[c0 + 1]);
    new_tet.insert(cells_[c0 + 2]);
    new_tet.insert(cells_[c1]);
    new_tet.insert(cells_[c1 + 1]);
    new_tet.insert(cells_[c1 + 2]);
    new_tet.insert(cells_[c2]);
    new_tet.insert(cells_[c2 + 1]);
    new_tet.insert(cells_[c2 + 2]);
    new_tet.insert(cells_[c3]);
    new_tet.insert(cells_[c3 + 1]);
    new_tet.insert(cells_[c3 + 2]);
    ASSERT(new_tet.size() == 4);

    set<unsigned int>::iterator iter = new_tet.begin();
    mod_tet(tets[0], *iter++, *iter++, *iter++, *iter);

    return true;
  }
  // should all share a center point, this set does not.
  return false;
}

void
TetVolMesh::nbors_from_2_to_3_split(Cell::index_type ci, 
				    Cell::array_type &nbors)
{
  //! The first 2 nodes make up the edge that the 3 tets share.
  //! Search the all_edges_ multiset for edges matching our edge.
  Edge::index_type ei;
  get_edge(ei, ci, ci * 4, ci * 4 + 1);

  pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =
    all_edges_.equal_range(ei);

  
  Edge::HalfEdgeSet::iterator edge_iter = range.first;
  nbors.clear();
  while(edge_iter != range.second) {
    nbors.push_back(*edge_iter / 6);
    ++edge_iter;
  }
}


//! ASSUMPTION: call this only if you know ci was created from a center split
//! operation. This call will succeed even if ci was not created this way
//! except if it is a boundary cell.
void
TetVolMesh::nbors_from_center_split(Cell::index_type ci, 
				    Cell::array_type &tets)
{
  // Any tet in the center split, knows its 3 nbors.  All faces
  // except the face opposite the center node define the nbors.

  unsigned index = ci * 4;
  Face::index_type f0 = cells_[index];
  Face::index_type f1 = cells_[index + 1];
  Face::index_type f2 = cells_[index + 2];
  
  tets.clear();
  tets.push_back(ci);
  Cell::index_type n;
  get_neighbor(n, ci, f0);
  tets.push_back(n);
  get_neighbor(n, ci, f1);
  tets.push_back(n);
  get_neighbor(n, ci, f2);
  tets.push_back(n);
}

//! Return in fi the face that is opposite the node ni in the cell ci.
//! Return false if bad input, else true indicating the face was found.
bool
TetVolMesh::get_face_opposite_node(Face::index_type &fi, Cell::index_type ci, 
				   Node::index_type ni) const
{
  for (unsigned int f = ci * 4; f < (ci * 4) + 4; f++)
  {
    if (cells_[f] == ni)
    {
      fi = f;
      return true;
    }
  }
  return false;
}

//! Get neigbor across a specific face.
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
    neighbor = MESH_NO_NEIGHBOR;
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
  ASSERTMSG(synchronized_ & FACE_NEIGHBORS_E,
	    "Must call synchronize FACE_NEIGHBORS_E on TetVolMesh first.");
  Face::index_type face;
  for (unsigned int i = idx*4; i < idx*4+4; i++)
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
TetVolMesh::get_neighbors(vector<Node::index_type> &array,
			  Node::index_type idx) const
{
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E, 
	    "Must call synchronize NODE_NEIGHBORS_E on TetVolMesh first.");
  set<unsigned int> inserted;
  for (unsigned int i = 0; i < node_neighbors_[idx].size(); i++)
  {
    const int base = node_neighbors_[idx][i]/4*4;
    for (int c = base; c < base+4; c++)
    {
      inserted.insert(cells_[c]);
    }
  }
  
  array.clear();
  array.reserve(inserted.size());
  array.insert(array.begin(), inserted.begin(), inserted.end());
}


void
TetVolMesh::get_center(Point &p, Edge::index_type idx) const
{
  const double s = 1.0/2.0;
  Node::array_type arr;
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
  Node::array_type arr;
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
  cell = locate_cache_;
  if (cell > Cell::index_type(0) &&
      cell < Cell::index_type(cells_.size()/4) &&
      inside(cell, p))
  {
      return true;
  }
  
  if (!(synchronized_ & LOCATE_E))
    synchronize(LOCATE_E);
  ASSERT(grid_.get_rep());

  unsigned int *iter, *end;
  if (grid_->lookup(&iter, &end, p))
  {
    while (iter != end)
    {
      if (inside(Cell::index_type(*iter), p))
      {
	cell = Cell::index_type(*iter);
	locate_cache_ = cell;
	return true;
      }
      ++iter;
    }
  }
  return false;
}


int
TetVolMesh::get_weights(const Point &p, Cell::array_type &l, double *w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}


// The volume x 6, used by get_weights to compute barycentric coordinates.
static double
tet_vol6(const Point &p1, const Point &p2, const Point &p3, const Point &p4)
{
  return fabs( Dot(Cross(p2-p1,p3-p1),p4-p1) );
}

void TetVolMesh::get_weights(const Point &p, Node::array_type &l,  vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    const Point &p0 = point(l[0]);
    const Point &p1 = point(l[1]);
    const Point &p2 = point(l[2]);
    const Point &p3 = point(l[3]);

    w.push_back(tet_vol6(p, p1, p2, p3));
    w.push_back(tet_vol6(p, p0, p2, p3));
    w.push_back(tet_vol6(p, p1, p0, p3));
    w.push_back(tet_vol6(p, p1, p2, p0));
    const double vol_sum_inv = 1.0 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= vol_sum_inv;
    w[1] *= vol_sum_inv;
    w[2] *= vol_sum_inv;
    w[3] *= vol_sum_inv;
    //   return 4;
  }
  //  return 0;
}

int
TetVolMesh::get_weights(const Point &p, Node::array_type &l, double *w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    const Point &p0 = point(l[0]);
    const Point &p1 = point(l[1]);
    const Point &p2 = point(l[2]);
    const Point &p3 = point(l[3]);

    w[0] = tet_vol6(p, p1, p2, p3);
    w[1] = tet_vol6(p, p0, p2, p3);
    w[2] = tet_vol6(p, p1, p0, p3);
    w[3] = tet_vol6(p, p1, p2, p0);
    const double vol_sum_inv = 1.0 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= vol_sum_inv;
    w[1] *= vol_sum_inv;
    w[2] *= vol_sum_inv;
    w[3] *= vol_sum_inv;
    return 4;
  }
  return 0;
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
    const int s = (int)(ceil(pow((double)csize , (1.0/3.0)))) / 2 + 1;
    const Vector cell_epsilon = bb.diagonal() * (1.0e-4 / s);
    bb.extend(bb.min() - cell_epsilon*2);
    bb.extend(bb.max() + cell_epsilon*2);

    SearchGridConstructor sgc(s, s, s, bb.min(), bb.max());

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

      sgc.insert(*ci, box);

      ++ci;
    }

    grid_ = scinew SearchGrid(sgc);
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

void TetVolMesh::get_basis(Cell::index_type ci, int gaussPt, double& g0, double& g1,
			       double& g2, double& g3)
{

  //Point& p1 = points_[cells_[ci * 4]];
  //Point& p2 = points_[cells_[ci * 4+1]];
  //Point& p3 = points_[cells_[ci * 4+2]];
  //Point& p4 = points_[cells_[ci * 4+3]];

 //  double x1=p1.x();
//   double y1=p1.y();
//   double z1=p1.z();
//   double x2=p2.x();
//   double y2=p2.y();
//   double z2=p2.z();
//   double x3=p3.x();
//   double y3=p3.y();
//   double z3=p3.z();
//   double x4=p4.x();
//   double y4=p4.y();
//   double z4=p4.z();
  
  double xi, nu, gam;

  switch(gaussPt) {
  case 0: 
    xi = 0.25;
    nu = 0.25;
    gam = 0.25;
    break;
  case 1:
    xi = 0.5;
    nu = 1.0/6.0;
    gam = 1.0/6.0;
    break;
  case 2:
    xi = 1.0/6.0;
    nu = 0.5;
    gam = 1.0/6.0;
    break;
  case 3:
    xi = 1.0/6.0;
    nu = 1.0/6.0;
    gam = 0.5;
    break;
  case 4:
    xi = 1.0/6.0;
    nu = 1.0/6.0;
    gam = 1.0/6.0;
    break;
  default: 
    xi = nu = gam = -1; // Removes compiler warning...
    cerr << "Error in get_basis: Incorrect index for gaussPt. "
	 << "index = " << gaussPt << endl;
  }

  g0 = 1-xi-nu-gam;
  g1 = xi;
  g2 = nu;
  g3 = gam;

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
TetVolMesh::delete_cells(set<unsigned int> &to_delete)
{
  cells_lock_.lock();
  set<unsigned int>::reverse_iterator iter = to_delete.rbegin();
  while (iter != to_delete.rend()) {
    // erase the correct cell
    TetVolMesh::Cell::index_type ci = *iter++;
    unsigned ind = ci * 4;
    vector<under_type>::iterator cb = cells_.begin() + ind;
    vector<under_type>::iterator ce = cb;
    ce+=4;
    cells_.erase(cb, ce);
  }
  cells_lock_.unlock();

  synchronized_ &= ~LOCATE_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  if (synchronized_ & FACE_NEIGHBORS_E) {
    synchronized_ &= ~FACE_NEIGHBORS_E;
    compute_faces();
  }
  if (synchronized_ & EDGE_NEIGHBORS_E) {
    synchronized_ &= ~EDGE_NEIGHBORS_E;
    compute_edges();
  }

}

void
TetVolMesh::delete_nodes(set<unsigned int> &to_delete)
{
  points_lock_.lock();
  set<unsigned int>::reverse_iterator iter = to_delete.rbegin();
  while (iter != to_delete.rend()) {
    TetVolMesh::Node::index_type n = *iter++;
    vector<Point>::iterator pit = points_.begin() + n;
    points_.erase(pit);
  }
  points_lock_.unlock();

  synchronized_ &= ~LOCATE_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  if (synchronized_ & FACE_NEIGHBORS_E) {
    synchronized_ &= ~FACE_NEIGHBORS_E;
    compute_faces();
  }
  if (synchronized_ & EDGE_NEIGHBORS_E) {
    synchronized_ &= ~EDGE_NEIGHBORS_E;
    compute_edges();
  }

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
    unsigned int base = ci * 4;
    mod_tet(ci, cells_[base+1],  cells_[base], cells_[base+2], cells_[base+3]);
    sgn=-sgn;
  }

  if(sgn < 1.e-9){ // return 0; // Degenerate...
    cerr << "Warning - small element, volume=" << sgn << endl;
  }
}


bool
TetVolMesh::insert_node_in_cell(Cell::array_type &tets, 
				Cell::index_type ci, 
				Node::index_type &pi, const Point &p)
{
  if (!inside(ci, p)) return false;

  pi = add_point(p);
  delete_cell_node_neighbors(ci);
  delete_cell_edges(ci);
  delete_cell_faces(ci);

  tets.resize(4, ci);
  const unsigned index = ci*4;
  tets[1] = add_tet(cells_[index+0], cells_[index+3], cells_[index+1], pi);
  tets[2] = add_tet(cells_[index+1], cells_[index+3], cells_[index+2], pi);
  tets[3] = add_tet(cells_[index+0], cells_[index+2], cells_[index+3], pi);
  
  cells_[index+3] = pi;
    
  create_cell_node_neighbors(ci);
  create_cell_edges(ci);
  create_cell_faces(ci);

  return true;
}

bool
TetVolMesh::insert_node(const Point &p)
{
  Node::index_type pi;
  Cell::index_type cell;
  locate(cell, p);

  Cell::array_type tets;
  return insert_node_in_cell(tets, cell, pi, p);
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
    return (TetVolMesh::Node::index_type)(MESH_NO_NEIGHBOR); 
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
TetVolMesh::refine_elements_levels(const Cell::array_type &cells, 
				   const vector<int> &refine_level,
				   cell_2_cell_map_t &child_2_parent)
{
  synchronize(FACE_NEIGHBORS_E | EDGE_NEIGHBORS_E);

  int current_level = 0;         
  Cell::array_type todo = cells;
  vector<int> new_level = refine_level, level;

  while(!todo.empty()) {
    ++current_level;
    const unsigned int num_todo = todo.size();    
    vector<Cell::array_type> todo_children(num_todo);
    cell_2_cell_map_t green_children;
    
    refine_elements(todo, todo_children, green_children);

    todo.clear();
    level = new_level;
    new_level.clear();

    for (unsigned int i = 0; i < num_todo; ++i) {
      Cell::index_type parent = todo_children[i][0];
      const unsigned int num_children = todo_children[i].size(); 
      ASSERT(num_children == 8);

      const cell_2_cell_map_t::iterator pos = child_2_parent.find(parent);
      if (pos != child_2_parent.end())
	parent = (*pos).second;

      for (unsigned int j = 0; j < num_children; ++j) {
	child_2_parent.insert(make_pair(todo_children[i][j],parent));
	if (level[i] > current_level) {
	  todo.push_back(todo_children[i][j]);
	  new_level.push_back(level[i]);
	}
      }
    }
    cell_2_cell_map_t::iterator iter = green_children.begin();
    const cell_2_cell_map_t::iterator iter_end = green_children.end();
    while (iter != iter_end) {
      child_2_parent.insert(*iter);
      iter++;
    }
  }
}
                                                                               
void
TetVolMesh::refine_elements(const Cell::array_type &cells, 
			    vector<Cell::array_type> &cell_children,
			    cell_2_cell_map_t &green_children)
{
#ifdef HAVE_HASH_MAP
#ifdef __ECC
  typedef hash_multimap<Edge::index_type, Node::index_type, Edge::CellEdgeHasher>  HalfEdgeMap;
  HalfEdgeMap inserted_nodes(edge_hasher_);
#else
  typedef hash_multimap<Edge::index_type, Node::index_type, Edge::CellEdgeHasher, Edge::EdgeComparitor>  HalfEdgeMap;
  HalfEdgeMap inserted_nodes(100, edge_hasher_, edge_comp_);
#endif // ifdef __ECC
#else // ifdef HAVE_HASH_SET
  typedef multimap<Edge::index_type, Node::index_type, Edge::EdgeComparitor>  HalfEdgeMap;
  HalfEdgeMap inserted_nodes(edge_comp_);
#endif // ifdef HAVE_HASH_SET
                                                                               
  // iterate over the cells
  const unsigned int num_cells = cells.size();
  for (unsigned int c = 0; c < num_cells; ++c) {
    Node::array_type nodes;
    const Cell::index_type &cell = cells[c];
    get_nodes(nodes,cell);

    // Loop through edges and create new nodes at center
    for (unsigned int edge = 0; edge < 6; ++edge)
    {
      unsigned int edgeNum = cell*6+edge;
      pair<Edge::HalfEdgeSet::iterator, Edge::HalfEdgeSet::iterator> range =
        all_edges_.equal_range(edgeNum);
      
      Node::index_type newnode;
      
      pair<HalfEdgeMap::iterator,HalfEdgeMap::iterator> iter =
        inserted_nodes.equal_range(edgeNum);

      if (iter.first == iter.second) {
        Point p;
        get_center(p, Edge::index_type(edgeNum));
        newnode = add_point(p);
        for (Edge::HalfEdgeSet::iterator e = range.first;
             e != range.second; ++e)
        {
          if(*e != edgeNum) {
             inserted_nodes.insert(make_pair(*e,newnode));
          }
        }
      } else {
        for (HalfEdgeMap::iterator e = iter.first; e != iter.second; ++e)
        {
          if((*e).first == edgeNum) {
            newnode = (*e).second;
            inserted_nodes.erase(e);
            break;
          }
        }
      }
      nodes.push_back(newnode);
                                                                               
    }
                                                                               
    // Perform an 8:1 split on this tet
    Elem::index_type t1 = mod_tet(cell,nodes[4],nodes[6],nodes[5],nodes[0]);
    Elem::index_type t2 = add_tet(nodes[4], nodes[7], nodes[9], nodes[1]);
    Elem::index_type t3 = add_tet(nodes[7], nodes[5], nodes[8], nodes[2]);
    Elem::index_type t4 = add_tet(nodes[6], nodes[9], nodes[8], nodes[3]);
                                                                               
    Point p4,p5,p6,p7,p8,p9;
    get_point(p4,nodes[4]);
    get_point(p5,nodes[5]);
    get_point(p6,nodes[6]);
    get_point(p7,nodes[7]);
    get_point(p8,nodes[8]);
    get_point(p9,nodes[9]);
                                                                                     
    double v48, v67, v59;
    v48 = (p4 - p8).length();
    v67 = (p6 - p7).length();
    v59 = (p5 - p9).length();
                                                                                     
    Elem::index_type t5,t6,t7,t8;
                                                                                     
    if(v48 >= v67 && v48 >= v59) {
      t5 = add_tet(nodes[4], nodes[9], nodes[8], nodes[6]);
      t6 = add_tet(nodes[4], nodes[8], nodes[5], nodes[6]);
      t7 = add_tet(nodes[4], nodes[5], nodes[8], nodes[7]);
      t8 = add_tet(nodes[4], nodes[8], nodes[9], nodes[7]);
    } else if(v67 >= v48 && v67 >= v59) {
      t5 = add_tet(nodes[9], nodes[4], nodes[6], nodes[7]);
      t6 = add_tet(nodes[8], nodes[6], nodes[5], nodes[7]);
      t7 = add_tet(nodes[4], nodes[5], nodes[6], nodes[7]);
      t8 = add_tet(nodes[7], nodes[6], nodes[9], nodes[8]);
    } else {
      t5 = add_tet(nodes[9], nodes[5], nodes[6], nodes[8]);
      t6 = add_tet(nodes[9], nodes[5], nodes[7], nodes[4]);
      t7 = add_tet(nodes[9], nodes[4], nodes[6], nodes[5]);
      t8 = add_tet(nodes[9], nodes[7], nodes[5], nodes[8]);
    }

    cell_children[c].push_back(t1);
    cell_children[c].push_back(t2);
    cell_children[c].push_back(t3);
    cell_children[c].push_back(t4);
    cell_children[c].push_back(t5);
    cell_children[c].push_back(t6);
    cell_children[c].push_back(t7);
    cell_children[c].push_back(t8);
  }

  typedef pair<Node::index_type, Node::index_type> node_pair_t;
  typedef map<node_pair_t, Node::index_type> edge_centers_t;
  edge_centers_t edge_centers;
  set<Cell::index_type> centersplits;
  HalfEdgeMap::iterator edge_iter = inserted_nodes.begin();
  while (edge_iter != inserted_nodes.end()) {
    const Edge::index_type edge = (*edge_iter).first;
    const Cell::index_type cell = edge/6;
    node_pair_t enodes = Edge::edgei((*edge_iter).first);
    enodes = make_pair(Max(cells_[enodes.first],cells_[enodes.second]),
                       Min(cells_[enodes.first],cells_[enodes.second]));
    edge_centers.insert(make_pair(enodes,(*edge_iter).second));
    centersplits.insert(cell);
    ++edge_iter;
  }
                                                                                  
  set<Cell::index_type>::iterator splitcell = centersplits.begin();
  hash_map<Cell::index_type, Cell::index_type> green_parent;
  while (splitcell != centersplits.end()) {
    const Cell::index_type cell = *splitcell;
    ++splitcell;

    // Make Center point of cell
    Point p;
    get_center(p, cell);
    Node::index_type center = add_point(p);

    // Get the nodes of the original cell
    Node::array_type cnodes;
    get_nodes(cnodes,cell);

    vector<Cell::index_type> tets(4);

    // Modify the first tet to be 1 of 4
    tets[0] = mod_tet(cell, center, cnodes[1], cnodes[0], cnodes[2]);
    // Create the last 3 tets
    tets[1] = add_tet(center, cnodes[2], cnodes[0], cnodes[3]);
    tets[2] = add_tet(center, cnodes[0], cnodes[1], cnodes[3]);
    tets[3] = add_tet(center, cnodes[1], cnodes[2], cnodes[3]);

    Node::array_type fnodes;
    node_pair_t fenodes;
    vector<edge_centers_t::iterator> edge_center_iters(3);
    for (unsigned int t = 0; t < 4; ++t) {
      green_children.insert(make_pair(tets[t], cell));
      get_nodes(fnodes, Face::index_type(tets[t]*4));
      fenodes = make_pair(Max(fnodes[0],fnodes[1]),
			  Min(fnodes[0],fnodes[1]));
      edge_center_iters[0] = edge_centers.find(fenodes);
      if (edge_center_iters[0] == edge_centers.end()) continue;

      fenodes = make_pair(Max(fnodes[0],fnodes[2]),
			  Min(fnodes[0],fnodes[2]));
      edge_center_iters[1] = edge_centers.find(fenodes);
      if (edge_center_iters[1] == edge_centers.end()) continue;

      fenodes = make_pair(Max(fnodes[1],fnodes[2]),
			  Min(fnodes[1],fnodes[2]));
      edge_center_iters[2] = edge_centers.find(fenodes);
      if (edge_center_iters[2] == edge_centers.end()) continue;

      Cell::index_type t1, t2, t3, t4;
      // Perform a 4:1 split on tet 
      t1 = mod_tet(tets[t], center, 
		   (*edge_center_iters[0]).second,
		   (*edge_center_iters[1]).second,
		   (*edge_center_iters[2]).second);
      
      t2 = add_tet(center,fnodes[0],
		   (*edge_center_iters[0]).second,
		   (*edge_center_iters[1]).second);

      t3 = add_tet(center,fnodes[1],
		   (*edge_center_iters[0]).second,
		   (*edge_center_iters[2]).second);

      t4 = add_tet(center,fnodes[2],
		   (*edge_center_iters[1]).second,
		   (*edge_center_iters[2]).second);

      orient(t1);
      orient(t2);
      orient(t3);
      orient(t4);

      green_children.insert(make_pair(t2, cell));
      green_children.insert(make_pair(t3, cell));
      green_children.insert(make_pair(t4, cell));
    }

      
  }

  edge_centers_t::iterator enodes_iter = edge_centers.begin();
  while (enodes_iter != edge_centers.end()) {
    const node_pair_t enodes = (*enodes_iter).first;
    const Node::index_type center = (*enodes_iter).second;
    Edge::array_type half_edges;
    if (is_edge(enodes.first, enodes.second, &half_edges)) {
      for (unsigned int e = 0; e < half_edges.size(); ++e) {
        const Edge::index_type edge = half_edges[e];
        const Cell::index_type cell = edge/6;
  
        pair<Node::index_type, Node::index_type> nnodes = Edge::edgei(edge);
        pair<Node::index_type, Node::index_type> onodes =
          Edge::edgei(Edge::opposite_edge(edge));

        // Perform the 2:1 split
	const Elem::index_type t1 = 
	  add_tet(center, cells_[nnodes.first],
		  cells_[onodes.second], cells_[onodes.first]);
        orient(t1);

	const cell_2_cell_map_t::iterator green_iter = 
	  green_children.find(cell);
	ASSERT(green_iter != green_children.end());
	
	green_children.insert(make_pair(t1, (*green_iter).second));

        orient(mod_tet(cell, center, cells_[nnodes.second],
                       cells_[onodes.second], cells_[onodes.first]));

      }
    }
    ++enodes_iter;
  }
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
  if (face_nbrs[0] != MESH_NO_NEIGHBOR)
  {
     Node::index_type opp = cells_[face_nbrs[0]];
    tets.push_back(mod_tet(face_nbrs[0]/4,nodes[7],nodes[8],nodes[9],opp));
    tets.push_back(add_tet(nodes[7],nodes[2],nodes[8],opp));
    tets.push_back(add_tet(nodes[8],nodes[3],nodes[9],opp));
    tets.push_back(add_tet(nodes[9],nodes[1],nodes[7],opp));
    done.insert(face_nbrs[0]/4);
  }
  // Perform a 4:1 split on tet sharing face 1
  if (face_nbrs[1] != MESH_NO_NEIGHBOR)
  {
    Node::index_type opp = cells_[face_nbrs[1]];
    tets.push_back(mod_tet(face_nbrs[1]/4,nodes[5],nodes[6],nodes[8],opp));
    tets.push_back(add_tet(nodes[5],nodes[0],nodes[6],opp));
    tets.push_back(add_tet(nodes[6],nodes[3],nodes[8],opp));
    tets.push_back(add_tet(nodes[8],nodes[2],nodes[5],opp));
    done.insert(face_nbrs[1]/4);
  }
  // Perform a 4:1 split on tet sharing face 2
  if (face_nbrs[2] != MESH_NO_NEIGHBOR)
  {
    Node::index_type opp = cells_[face_nbrs[2]];
    tets.push_back(mod_tet(face_nbrs[2]/4,nodes[4],nodes[9],nodes[6],opp));
    tets.push_back(add_tet(nodes[4],nodes[1],nodes[9],opp));
    tets.push_back(add_tet(nodes[9],nodes[3],nodes[6],opp));
    tets.push_back(add_tet(nodes[6],nodes[0],nodes[4],opp));
    done.insert(face_nbrs[2]/4);
  }
  // Perform a 4:1 split on tet sharing face 3
  if (face_nbrs[3] != MESH_NO_NEIGHBOR)
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
    vector<unsigned int> neighbors;
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
