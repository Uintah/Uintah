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
 *  LatVolMesh.cc: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan &&
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MusilRNG.h>
#include <vector>

namespace SCIRun {

using namespace std;


PersistentTypeID LatVolMesh::type_id("LatVolMesh", "Mesh", maker);


LatVolMesh::LatVolMesh(unsigned x, unsigned y, unsigned z,
		       const Point &min, const Point &max)
  : min_i_(0), min_j_(0), min_k_(0),
    ni_(x), nj_(y), nk_(z)
{
  transform_.pre_scale(Vector(1.0 / (x-1.0), 1.0 / (y-1.0), 1.0 / (z-1.0)));
  transform_.pre_scale(max - min);

  transform_.pre_translate(Vector(min));
  transform_.compute_imat();
}


void
LatVolMesh::get_random_point(Point &p, const Elem::index_type &ei,
			     int seed) const
{
  static MusilRNG rng;

  // build the three principal edge vectors
  Node::array_type ra;
  get_nodes(ra,ei);
  Point p0,p1,p2,p3;
  get_point(p0,ra[0]);
  get_point(p1,ra[1]);
  get_point(p2,ra[3]);
  get_point(p3,ra[4]);
  Vector v0(p1-p0);
  Vector v1(p2-p0);
  Vector v2(p3-p0);

  // choose a random point in the cell
  double t, u, v;
  if (seed) {
    MusilRNG rng1(seed);
    t = rng1();
    u = rng1();
    v = rng1();
  } else {
    t = rng();
    u = rng();
    v = rng();
  }
  p = p0+(v0*t)+(v1*u)+(v2*v);
}

BBox
LatVolMesh::get_bounding_box() const
{
  Point p0(min_i_,         min_j_,         min_k_);
  Point p1(min_i_ + ni_-1, min_j_,         min_k_);
  Point p2(min_i_ + ni_-1, min_j_ + nj_-1, min_k_);
  Point p3(min_i_,         min_j_ + nj_-1, min_k_);
  Point p4(min_i_,         min_j_,         min_k_ + nk_-1);
  Point p5(min_i_ + ni_-1, min_j_,         min_k_ + nk_-1);
  Point p6(min_i_ + ni_-1, min_j_ + nj_-1, min_k_ + nk_-1);
  Point p7(min_i_,         min_j_ + nj_-1, min_k_ + nk_-1);
  
  BBox result;
  result.extend(transform_.project(p0));
  result.extend(transform_.project(p1));
  result.extend(transform_.project(p2));
  result.extend(transform_.project(p3));
  result.extend(transform_.project(p4));
  result.extend(transform_.project(p5));
  result.extend(transform_.project(p6));
  result.extend(transform_.project(p7));
  return result;
}

Vector LatVolMesh::diagonal() const
{
  return get_bounding_box().diagonal();
}

void
LatVolMesh::transform(const Transform &t)
{
  transform_.pre_trans(t);
}

void
LatVolMesh::get_canonical_transform(Transform &t) 
{
  t = transform_;
  t.post_scale(Vector(ni_ - 1.0, nj_ - 1.0, nk_ - 1.0));
}

bool
LatVolMesh::get_min(vector<unsigned int> &array) const
{
  array.resize(3);
  array.clear();

  array.push_back(min_i_);
  array.push_back(min_j_);
  array.push_back(min_k_);

  return true;
}

bool
LatVolMesh::get_dim(vector<unsigned int> &array) const
{
  array.resize(3);
  array.clear();

  array.push_back(ni_);
  array.push_back(nj_);
  array.push_back(nk_);

  return true;
}

void
LatVolMesh::set_min(vector<unsigned int> min)
{
  min_i_ = min[0];
  min_j_ = min[1];
  min_k_ = min[2];
}

void
LatVolMesh::set_dim(vector<unsigned int> dim)
{
  ni_ = dim[0];
  nj_ = dim[1];
  nk_ = dim[2];
}


// Note: This code does not respect boundaries of the mesh
void
LatVolMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.resize(2);
  const unsigned int xidx = idx;
  if (xidx < (ni_ - 1) * nj_ * nk_)
  {
    const int i = xidx % (ni_ - 1);
    const int jk = xidx / (ni_ - 1);
    const int j = jk % nj_;
    const int k = jk / nj_;

    array[0] = Node::index_type(this, i+0, j, k);
    array[1] = Node::index_type(this, i+1, j, k);
  }
  else
  {
    const unsigned int yidx = idx - (ni_ - 1) * nj_ * nk_;
    if (yidx < (ni_ * (nj_ - 1) * nk_))
    {
      const int j = yidx % (nj_ - 1);
      const int ik = yidx / (nj_ - 1);
      const int i = ik / nk_;
      const int k = ik % nk_;

      array[0] = Node::index_type(this, i, j+0, k);
      array[1] = Node::index_type(this, i, j+1, k);
    }
    else
    {
      const unsigned int zidx = yidx - (ni_ * (nj_ - 1) * nk_);
      const int k = zidx % (nk_ - 1);
      const int ij = zidx / (nk_ - 1);
      const int i = ij % ni_;
      const int j = ij / ni_;

      array[0] = Node::index_type(this, i, j, k+0);
      array[1] = Node::index_type(this, i, j, k+1);
    }      
  }
}



// Note: This code does not respect boundaries of the mesh
void
LatVolMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.resize(4);
  const unsigned int xidx = idx;
  if (xidx < (ni_ - 1) * (nj_ - 1) * nk_)
  {
    const int i = xidx % (ni_ - 1);
    const int jk = xidx / (ni_ - 1);
    const int j = jk % (nj_ - 1);
    const int k = jk / (nj_ - 1);
    array[0] = Node::index_type(this, i+0, j+0, k);
    array[1] = Node::index_type(this, i+1, j+0, k);
    array[2] = Node::index_type(this, i+1, j+1, k);
    array[3] = Node::index_type(this, i+0, j+1, k);
  }
  else
  {
    const unsigned int yidx = idx - (ni_ - 1) * (nj_ - 1) * nk_;
    if (yidx < ni_ * (nj_ - 1) * (nk_ - 1))
    {
      const int j = yidx % (nj_ - 1);
      const int ik = yidx / (nj_ - 1);
      const int k = ik % (nk_ - 1);
      const int i = ik / (nk_ - 1);
      array[0] = Node::index_type(this, i, j+0, k+0);
      array[1] = Node::index_type(this, i, j+1, k+0);
      array[2] = Node::index_type(this, i, j+1, k+1);
      array[3] = Node::index_type(this, i, j+0, k+1);
    }
    else
    {
      const unsigned int zidx = yidx - ni_ * (nj_ - 1) * (nk_ - 1);
      const int k = zidx % (nk_ - 1);
      const int ij = zidx / (nk_ - 1);
      const int i = ij % (ni_ - 1);
      const int j = ij / (ni_ - 1);
      array[0] = Node::index_type(this, i+0, j, k+0);
      array[1] = Node::index_type(this, i+0, j, k+1);
      array[2] = Node::index_type(this, i+1, j, k+1);
      array[3] = Node::index_type(this, i+1, j, k+0);
    }
  }
}

// Note: This code does not respect boundaries of the mesh
void
LatVolMesh::get_nodes(Node::array_type &array, Cell::index_type idx) const
{
  array.resize(8);
  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;   array[0].k_ = idx.k_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;   array[1].k_ = idx.k_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1; array[2].k_ = idx.k_;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1; array[3].k_ = idx.k_;
  array[4].i_ = idx.i_;   array[4].j_ = idx.j_;   array[4].k_ = idx.k_+1;
  array[5].i_ = idx.i_+1; array[5].j_ = idx.j_;   array[5].k_ = idx.k_+1;
  array[6].i_ = idx.i_+1; array[6].j_ = idx.j_+1; array[6].k_ = idx.k_+1;
  array[7].i_ = idx.i_;   array[7].j_ = idx.j_+1; array[7].k_ = idx.k_+1;

  array[0].mesh_ = this;
  array[1].mesh_ = this;
  array[2].mesh_ = this;
  array[3].mesh_ = this;
  array[4].mesh_ = this;
  array[5].mesh_ = this;
  array[6].mesh_ = this;
  array[7].mesh_ = this;
}


void
LatVolMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
{
  array.resize(4);
  const unsigned int num_i_faces = (ni_-1)*(nj_-1)*nk_;  // lie in ij plane ijk
  const unsigned int num_j_faces = ni_*(nj_-1)*(nk_-1);  // lie in jk plane jki
  const unsigned int num_k_faces = (ni_-1)*nj_*(nk_-1);  // lie in ki plane kij

  const unsigned int num_i_edges = (ni_-1)*nj_*nk_; // ijk
  const unsigned int num_j_edges = ni_*(nj_-1)*nk_; // jki
  //  const unsigned int num_k_edges = ni_*nj_*(nk_-1); // kij

  unsigned int facei, facej, facek;
  unsigned int face = idx;
  
  if (face < num_i_faces)
  {
    facei = face % (ni_-1);
    facej = (face / (ni_-1)) % (nj_-1);
    facek = face / ((ni_-1)*(nj_-1));
    array[0] = facei+facej*(ni_-1)+facek*(ni_-1)*(nj_);
    array[1] = facei+(facej+1)*(ni_-1)+facek*(ni_-1)*(nj_);
    array[2] = num_i_edges + facei*(nj_-1)*(nk_)+facej+facek*(nj_-1);
    array[3] = num_i_edges + (facei+1)*(nj_-1)*(nk_)+facej+facek*(nj_-1);    
  }
  else if (face - num_i_faces < num_j_faces)
  {
    face -= num_i_faces;
    facei = face / ((nj_-1) *(nk_-1));
    facej = face % (nj_-1);
    facek = (face / (nj_-1)) % (nk_-1);
    array[0] = num_i_edges + facei*(nj_-1)*(nk_)+facej+facek*(nj_-1);
    array[1] = num_i_edges + facei*(nj_-1)*(nk_)+facej+(facek+1)*(nj_-1);    
    array[2] = (num_i_edges + num_j_edges + 
		facei*(nk_-1)+facej*(ni_)*(nk_-1)+facek);
    array[3] = (num_i_edges + num_j_edges + 
		facei*(nk_-1)+(facej+1)*(ni_)*(nk_-1)+facek);

  }
  else if (face - num_i_faces - num_j_faces < num_k_faces)
  {
    face -= (num_i_faces + num_j_faces);
    facei = (face / (nk_-1)) % (ni_-1);
    facej = face / ((ni_-1) * (nk_-1));
    facek = face % (nk_-1);
    array[0] = facei+facej*(ni_-1)+facek*(ni_-1)*(nj_);
    array[1] = facei+facej*(ni_-1)+(facek+1)*(ni_-1)*(nj_);
    array[2] = (num_i_edges + num_j_edges + 
		facei*(nk_-1)+facej*(ni_)*(nk_-1)+facek);
    array[3] = (num_i_edges + num_j_edges + 
		(facei+1)*(nk_-1)+facej*(ni_)*(nk_-1)+facek);
  }
  else ASSERTFAIL("LatVolMesh::get_edges(Edge, Face) Face idx out of bounds"); 

}
  
  
    
void
LatVolMesh::get_edges(Edge::array_type &array, Cell::index_type idx) const
{
  array.resize(12);
  const unsigned int j_start= (ni_-1)*nj_*nk_; 
  const unsigned int k_start = ni_*(nj_-1)*nk_ + j_start; 

  array[0] = idx.i_ + idx.j_*(ni_-1)     + idx.k_*(ni_-1)*(nj_);
  array[1] = idx.i_ + (idx.j_+1)*(ni_-1) + idx.k_*(ni_-1)*(nj_);
  array[2] = idx.i_ + idx.j_*(ni_-1)     + (idx.k_+1)*(ni_-1)*(nj_);
  array[3] = idx.i_ + (idx.j_+1)*(ni_-1) + (idx.k_+1)*(ni_-1)*(nj_);

  array[4] = j_start + idx.i_*(nj_-1)*(nk_)     + idx.j_ + idx.k_*(nj_-1);
  array[5] = j_start + (idx.i_+1)*(nj_-1)*(nk_) + idx.j_ + idx.k_*(nj_-1);
  array[6] = j_start + idx.i_*(nj_-1)*(nk_)     + idx.j_ + (idx.k_+1)*(nj_-1);
  array[7] = j_start + (idx.i_+1)*(nj_-1)*(nk_) + idx.j_ + (idx.k_+1)*(nj_-1);

  array[8] =  k_start + idx.i_*(nk_-1)     + idx.j_*(ni_)*(nk_-1)     + idx.k_;
  array[9] =  k_start + (idx.i_+1)*(nk_-1) + idx.j_*(ni_)*(nk_-1)     + idx.k_;
  array[10] = k_start + idx.i_*(nk_-1)     + (idx.j_+1)*(ni_)*(nk_-1) + idx.k_;
  array[11] = k_start + (idx.i_+1)*(nk_-1) + (idx.j_+1)*(ni_)*(nk_-1) + idx.k_;
  
}



void
LatVolMesh::get_faces(Face::array_type &array, Cell::index_type idx) const
{
  array.resize(6);

  const unsigned int i = idx.i_;
  const unsigned int j = idx.j_;
  const unsigned int k = idx.k_;

  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);

  array[0] = i + (j + k * (nj_-1)) * (ni_-1);
  array[1] = i + (j + (k+1) * (nj_-1)) * (ni_-1);

  array[2] = offset1 + j + (k + i * (nk_-1)) * (nj_-1);
  array[3] = offset1 + j + (k + (i+1) * (nk_-1)) * (nj_-1);

  array[4] = offset2 + k + (i + j * (ni_-1)) * (nk_-1);
  array[5] = offset2 + k + (i + (j+1) * (ni_-1)) * (nk_-1);
}


//! return all cell_indecies that overlap the BBox in arr.

void
LatVolMesh::get_cells(Cell::array_type &arr, const BBox &bbox)
{
  // Limited to range of ints.
  arr.clear();
  const Point minp = transform_.unproject(bbox.min());
  int mini = (int)floor(minp.x());
  int minj = (int)floor(minp.y());
  int mink = (int)floor(minp.z());
  if (mini < 0) { mini = 0; }
  if (minj < 0) { minj = 0; }
  if (mink < 0) { mink = 0; }

  const Point maxp = transform_.unproject(bbox.max());
  int maxi = (int)floor(maxp.x());
  int maxj = (int)floor(maxp.y());
  int maxk = (int)floor(maxp.z());
  if (maxi >= (int)(ni_ - 1)) { maxi = ni_ - 1; }
  if (maxj >= (int)(nj_ - 1)) { maxj = nj_ - 1; }
  if (maxk >= (int)(nk_ - 1)) { maxk = nk_ - 1; }
  
  int i, j, k;
  for (i = mini; i <= maxi; i++) {
    for (j = minj; j <= maxj; j++) {
      for (k = mink; k <= maxk; k++) {
	arr.push_back(Cell::index_type(this, i,j,k));
      }
    }
  }
}


bool
LatVolMesh::get_neighbor(Cell::index_type &neighbor,
			 const Cell::index_type &from,
			 const Face::index_type &face) const
{
  const unsigned int xidx = face;
  if (xidx < (ni_ - 1) * (nj_ - 1) * nk_)
  {
    //const unsigned int i = xidx % (ni_ - 1);
    const unsigned int jk = xidx / (ni_ - 1);
    //const unsigned int j = jk % (nj_ - 1);
    const unsigned int k = jk / (nj_ - 1);

    if (k == from.k_ && k > 0)
    {
      neighbor.i_ = from.i_;
      neighbor.j_ = from.j_;
      neighbor.k_ = k-1;
      return true;
    }
    else if (k == (from.k_+1) && k < (nk_-1))
    {
      neighbor.i_ = from.i_;
      neighbor.j_ = from.j_;
      neighbor.k_ = k;
      return true;
    }
  }
  else
  {
    const unsigned int yidx = xidx - (ni_ - 1) * (nj_ - 1) * nk_;
    if (yidx < ni_ * (nj_ - 1) * (nk_ - 1))
    {
      //const unsigned int j = yidx % (nj_ - 1);
      const unsigned int ik = yidx / (nj_ - 1);
      //const unsigned int k = ik % (nk_ - 1);
      const unsigned int i = ik / (nk_ - 1);

      if (i == from.i_ && i > 0)
      {
	neighbor.i_ = i-1;
	neighbor.j_ = from.j_;
	neighbor.k_ = from.k_;
	return true;
      }
      else if (i == (from.i_+1) && i < (ni_-1))
      {
	neighbor.i_ = i;
	neighbor.j_ = from.j_;
	neighbor.k_ = from.k_;
	return true;
      }
    }
    else
    {
      const unsigned int zidx = yidx - ni_ * (nj_ - 1) * (nk_ - 1);
      //const unsigned int k = zidx % (nk_ - 1);
      const unsigned int ij = zidx / (nk_ - 1);
      //const unsigned int i = ij % (ni_ - 1);
      const unsigned int j = ij / (ni_ - 1);

      if (j == from.j_ && j > 0)
      {
	neighbor.i_ = from.i_;
	neighbor.j_ = j-1;
	neighbor.k_ = from.k_;
	return true;
      }
      else if (j == (from.j_+1) && j < (nj_-1))
      {
	neighbor.i_ = from.i_;
	neighbor.j_ = j;
	neighbor.k_ = from.k_;
	return true;
      }
    }
  }
  return false;
}


//! return iterators over that fall within or on the BBox

// This function does a pretty good job of giving you the set of iterators
// that would loop over the mesh that falls within the BBox.  When the
// BBox falls outside the mesh boundaries, the iterators should equal each
// other so that a for loop using them [for(;iter != end; iter++)] will
// not enter.  There are some cases where you can get a range for cells on
// the edge when the BBox is to the side, but not inside, the mesh.
// This could be remedied, by more insightful inspection of the bounding
// box and the mesh.  Checking all cases would be tedious and probably
// fraught with error.
void
LatVolMesh::get_cell_range(Cell::range_iter &iter, Cell::iterator &end_iter,
			   const BBox &box) {
  // get the min and max points of the bbox and make sure that they lie
  // inside the mesh boundaries.
  BBox mesh_boundary = get_bounding_box();
  // crop by min boundary
  Point min = Max(box.min(), mesh_boundary.min());
  Point max = Max(box.max(), mesh_boundary.min());
  // crop by max boundary
  min = Min(min, mesh_boundary.max());
  max = Min(max, mesh_boundary.max());
  
  Cell::index_type min_index, mai_index;

  // If one of the locates return true, then we have a valid iteration
  bool min_located = locate(min_index, min);
  bool mai_located = locate(mai_index, max);
  if (min_located || mai_located) {
    // Initialize the range iterator
    iter = Cell::range_iter(this,
			    min_index.i_, min_index.j_, min_index.k_,
			    mai_index.i_, mai_index.j_, mai_index.k_);
  } else {
    // If both of these are false then we are outside the boundary.
    // Set the min and max extents of the range iterator to be the same thing.
    // When they are the same end_iter will be set to the starting state of
    // the range iterator, thereby causing any for loop using these
    // iterators [for(;iter != end_iter; iter++)] to never enter.
    iter = Cell::range_iter(this, 0, 0, 0, 0, 0, 0);
  }
  // initialize the end iterator
  iter.end(end_iter);
}

void
LatVolMesh::get_node_range(Node::range_iter &iter, Node::iterator &end_iter,
			   const BBox &box) {
  // get the min and max points of the bbox and make sure that they lie
  // inside the mesh boundaries.
  BBox mesh_boundary = get_bounding_box();
  // crop by min boundary
  Point min = Max(box.min(), mesh_boundary.min());
  Point max = Max(box.max(), mesh_boundary.min());
  // crop by max boundary
  min = Min(min, mesh_boundary.max());
  max = Min(max, mesh_boundary.max());
  
  Node::index_type min_index, mai_index;

  // If one of the locates return true, then we have a valid iteration
  bool min_located = locate(min_index, min);
  bool mai_located = locate(mai_index, max);
  if (min_located || mai_located) {
    // Initialize the range iterator
    iter = Node::range_iter(this,
			    min_index.i_, min_index.j_, min_index.k_,
			    mai_index.i_, mai_index.j_, mai_index.k_);
  } else {
    // If both of these are false then we are outside the boundary.
    // Set the min and max extents of the range iterator to be the same thing.
    // When they are the same end_iter will be set to the starting state of
    // the range iterator, thereby causing any for loop using these
    // iterators [for(;iter != end_iter; iter++)] to never enter.
    iter = Node::range_iter(this, 0, 0, 0, 0, 0, 0);
  }
  // initialize the end iterator
  iter.end(end_iter);
}

void
LatVolMesh::get_cell_range(Cell::range_iter &begin, Cell::iterator &end,
			   const Cell::index_type &begin_index,
			   const Cell::index_type &end_index) {
  begin = Cell::range_iter(this,
			   begin_index.i_, begin_index.j_, begin_index.k_,
			     end_index.i_,   end_index.j_,   end_index.k_);
  begin.end(end);
}

void
LatVolMesh::get_node_range(Node::range_iter &begin, Node::iterator &end,
			   const Node::index_type &begin_index,
			   const Node::index_type &end_index) {
  begin = Node::range_iter(this,
			   begin_index.i_, begin_index.j_, begin_index.k_,
			     end_index.i_,   end_index.j_,   end_index.k_);
  begin.end(end);
}

void
LatVolMesh::get_center(Point &result, Edge::index_type idx) const
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
LatVolMesh::get_center(Point &result, Face::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  Node::array_type::iterator nai = nodes.begin();
  get_point(result, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    result.asVector() += pp.asVector();
    ++nai;
  }
  result.asVector() *= (1.0 / 4.0);
}


void
LatVolMesh::get_center(Point &result, const Cell::index_type &idx) const
{
  Point p(idx.i_ + 0.5, idx.j_ + 0.5, idx.k_ + 0.5);
  result = transform_.project(p);
}

void
LatVolMesh::get_center(Point &result, const Node::index_type &idx) const
{
  Point p(idx.i_, idx.j_, idx.k_);
  result = transform_.project(p);
}



double
LatVolMesh::get_size(Node::index_type idx) const
{
  return 0.0;
}


double
LatVolMesh::get_size(Edge::index_type idx) const
{
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p0, p1;
  get_center(p0, arr[0]);
  get_center(p1, arr[1]);

  return (p1.asVector() - p0.asVector()).length();
}
  

double
LatVolMesh::get_size(Face::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  Point p0, p1, p2;
  get_point(p0, nodes[0]);
  get_point(p1, nodes[1]);
  get_point(p2, nodes[2]);
  Vector v0 = p1 - p0;
  Vector v1 = p2 - p0;
  return (v0.length() * v1.length());
}


double
LatVolMesh::get_size(Cell::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  Point p0, p1, p2, p3;
  get_point(p0, nodes[0]);
  get_point(p1, nodes[1]);
  get_point(p2, nodes[3]);
  get_point(p3, nodes[4]);
  Vector v0 = p1 - p0;
  Vector v1 = p2 - p0;
  Vector v2 = p3 - p0;
  return (v0.length() * v1.length() * v2.length());
}




bool
LatVolMesh::locate(Cell::index_type &cell, const Point &p)
{
  const Point r = transform_.unproject(p);

  // Rounds down, so catches intervals.  Might lose numerical precision on
  // upper edge (ie nodes on upper edges are not in any cell).
  // Nodes over 2 billion might suffer roundoff error.

  // These values are also clamped off at zero to prevent problems
  // when round off error places the value of r.x() less than zero.
  cell.i_ = (r.x()>0?(unsigned int)floor(r.x()):0);
  cell.j_ = (r.y()>0?(unsigned int)floor(r.y()):0);
  cell.k_ = (r.z()>0?(unsigned int)floor(r.z()):0);
  
  cell.mesh_ = this;

  if (cell.i_ >= (ni_-1) ||
      cell.j_ >= (nj_-1) ||
      cell.k_ >= (nk_-1))
  {
    return false;
  }
  else
  {
    return true;
  }
}



bool
LatVolMesh::locate(Node::index_type &node, const Point &p)
{
  const Point r = transform_.unproject(p);

  // Nodes over 2 billion might suffer roundoff error.
  node.i_ = (unsigned int)floor(r.x() + 0.5);
  node.j_ = (unsigned int)floor(r.y() + 0.5);
  node.k_ = (unsigned int)floor(r.z() + 0.5);
  node.mesh_ = this;

  if (node.i_ >= ni_ ||
      node.j_ >= nj_ ||
      node.k_ >= nk_)
  {
    return false;
  }
  else
  {
    return true;
  }
}


void
LatVolMesh::get_weights(const Point &p,
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
LatVolMesh::get_weights(const Point &p,
			Node::array_type &locs, vector<double> &w)
{
  const Point r = transform_.unproject(p);
  double ii = r.x();
  double jj = r.y();
  double kk = r.z();

  if (ii>(ni_-1) && (ii-(1.e-10))<(ni_-1)) ii=ni_-1-(1.e-10);
  if (jj>(nj_-1) && (jj-(1.e-10))<(nj_-1)) jj=nj_-1-(1.e-10);
  if (kk>(nk_-1) && (kk-(1.e-10))<(nk_-1)) kk=nk_-1-(1.e-10);
  if (ii<0 && ii>(-1.e-10)) ii=0;
  if (jj<0 && jj>(-1.e-10)) jj=0;
  if (kk<0 && kk>(-1.e-10)) kk=0;

  const unsigned int i = (unsigned int)floor(ii);
  const unsigned int j = (unsigned int)floor(jj);
  const unsigned int k = (unsigned int)floor(kk);

  if (i < (ni_-1) && i >= 0 &&
      j < (nj_-1) && j >= 0 &&
      k < (nk_-1) && k >= 0)
  {
    locs.resize(8);
    locs[0].i_ = i;   locs[0].j_ = j;   locs[0].k_ = k;   locs[0].mesh_=this;
    locs[1].i_ = i+1; locs[1].j_ = j;   locs[1].k_ = k;   locs[1].mesh_=this;
    locs[2].i_ = i+1; locs[2].j_ = j+1; locs[2].k_ = k;   locs[2].mesh_=this;
    locs[3].i_ = i;   locs[3].j_ = j+1; locs[3].k_ = k;   locs[3].mesh_=this;
    locs[4].i_ = i;   locs[4].j_ = j;   locs[4].k_ = k+1; locs[4].mesh_=this;
    locs[5].i_ = i+1; locs[5].j_ = j;   locs[5].k_ = k+1; locs[5].mesh_=this;
    locs[6].i_ = i+1; locs[6].j_ = j+1; locs[6].k_ = k+1; locs[6].mesh_=this;
    locs[7].i_ = i;   locs[7].j_ = j+1; locs[7].k_ = k+1; locs[7].mesh_=this;
    
    const double di = ii - (double)i;
    const double dj = jj - (double)j;
    const double dk = kk - (double)k;
    
    w.resize(8);
    w[0] = (1.0 - di) * (1.0 - dj) * (1.0 - dk);
    w[1] = di         * (1.0 - dj) * (1.0 - dk);
    w[2] = di         * dj         * (1.0 - dk);
    w[3] = (1.0 - di) * dj         * (1.0 - dk);
    w[4] = (1.0 - di) * (1.0 - dj) * dk;
    w[5] = di         * (1.0 - dj) * dk;
    w[6] = di         * dj         * dk;
    w[7] = (1.0 - di) * dj         * dk;
  }
}

const TypeDescription* get_type_description(LatVolMesh::NodeIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("LatVolMesh::NodeIndex",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription* get_type_description(LatVolMesh::CellIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("LatVolMesh::CellIndex",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

void
Pio(Piostream& stream, LatVolMesh::NodeIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    Pio(stream, n.k_);
    stream.end_cheap_delim();
}

void
Pio(Piostream& stream, LatVolMesh::CellIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    Pio(stream, n.k_);
    stream.end_cheap_delim();
}

const string find_type_name(LatVolMesh::NodeIndex *)
{
  static string name = "LatVolMesh::NodeIndex";
  return name;
}
const string find_type_name(LatVolMesh::CellIndex *)
{
  static string name = "LatVolMesh::CellIndex";
  return name;
}

#define LATVOLMESH_VERSION 3

void
LatVolMesh::io(Piostream& stream)
{
  int version = stream.begin_class(type_name(-1), LATVOLMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, ni_);
  Pio(stream, nj_);
  Pio(stream, nk_);

  if (version < 2 && stream.reading())
  {
    Point min, max;
    Pio(stream, min);
    Pio(stream, max);
    transform_.pre_scale(Vector(1.0 / (ni_ - 1.0),
				1.0 / (nj_ - 1.0),
				1.0 / (nk_ - 1.0)));
    transform_.pre_scale(max - min);
    transform_.pre_translate(Vector(min));
    transform_.compute_imat();
  } else if (version < 3 && stream.reading() ) {
    Pio_old(stream, transform_);
  }
  else
  {
    Pio(stream, transform_);
  }

  stream.end_class();
}

const string
LatVolMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "LatVolMesh";
  return name;
}


void
LatVolMesh::begin(LatVolMesh::Node::iterator &itr) const
{
  itr = Node::iterator(this, min_i_, min_j_, min_k_);
}

void
LatVolMesh::end(LatVolMesh::Node::iterator &itr) const
{
  itr = Node::iterator(this, min_i_, min_j_, min_k_ + nk_);
}

void
LatVolMesh::size(LatVolMesh::Node::size_type &s) const
{
  s = Node::size_type(ni_,nj_,nk_);
}

void
LatVolMesh::begin(LatVolMesh::Edge::iterator &itr) const
{
  itr = Edge::iterator(0);
}

void
LatVolMesh::end(LatVolMesh::Edge::iterator &itr) const
{
  itr = ((ni_-1) * nj_ * nk_) + (ni_ * (nj_-1) * nk_) + (ni_ * nj_ * (nk_-1));
}

void
LatVolMesh::size(LatVolMesh::Edge::size_type &s) const
{
  s = ((ni_-1) * nj_ * nk_) + (ni_ * (nj_-1) * nk_) + (ni_ * nj_ * (nk_-1));
}

void
LatVolMesh::begin(LatVolMesh::Face::iterator &itr) const
{
  itr = Face::iterator(0);
}

void
LatVolMesh::end(LatVolMesh::Face::iterator &itr) const
{
  itr = (ni_-1) * (nj_-1) * nk_ +
    ni_ * (nj_ - 1 ) * (nk_ - 1) +
    (ni_ - 1) * nj_ * (nk_ - 1);
}

void
LatVolMesh::size(LatVolMesh::Face::size_type &s) const
{
  s =  (ni_-1) * (nj_-1) * nk_ +
    ni_ * (nj_ - 1 ) * (nk_ - 1) +
    (ni_ - 1) * nj_ * (nk_ - 1);
}

void
LatVolMesh::begin(LatVolMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(this,  min_i_, min_j_, min_k_);
}

void
LatVolMesh::end(LatVolMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(this, min_i_, min_j_, min_k_ + nk_-1);
}

void
LatVolMesh::size(LatVolMesh::Cell::size_type &s) const
{
  s = Cell::size_type(ni_-1, nj_-1,nk_-1);
}

int
LatVolMesh::get_valence(LatVolMesh::Node::index_type i) const
{
  return (((i.i_ == 0 || i.i_ == ni_) ? 1 : 2) +
	  ((i.j_ == 0 || i.j_ == nj_) ? 1 : 2) +
	  ((i.k_ == 0 || i.k_ == nk_) ? 1 : 2));
}

int
LatVolMesh::get_valence(LatVolMesh::Edge::index_type i) const
{
  return 1;
}

int
LatVolMesh::get_valence(LatVolMesh::Face::index_type i) const
{
  return 1;
}


int
LatVolMesh::get_valence(LatVolMesh::Cell::index_type i) const
{
  return 1;
}




const TypeDescription*
LatVolMesh::get_type_description() const
{
  return SCIRun::get_type_description((LatVolMesh *)0);
}

const TypeDescription*
get_type_description(LatVolMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LatVolMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(LatVolMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LatVolMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(LatVolMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LatVolMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(LatVolMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LatVolMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(LatVolMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("LatVolMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
