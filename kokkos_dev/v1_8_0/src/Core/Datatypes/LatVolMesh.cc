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
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Geometry/BBox.h>
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID LatVolMesh::type_id("LatVolMesh", "Mesh", maker);


LatVolMesh::LatVolMesh(unsigned x, unsigned y, unsigned z,
		       const Point &min, const Point &max)
  : min_x_(0), min_y_(0), min_z_(0),
    nx_(x), ny_(y), nz_(z)
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
  Point p0(min_x_,         min_y_,         min_z_);
  Point p1(min_x_ + nx_-1, min_y_,         min_z_);
  Point p2(min_x_ + nx_-1, min_y_ + ny_-1, min_z_);
  Point p3(min_x_,         min_y_ + ny_-1, min_z_);
  Point p4(min_x_,         min_y_,         min_z_ + nz_-1);
  Point p5(min_x_ + nx_-1, min_y_,         min_z_ + nz_-1);
  Point p6(min_x_ + nx_-1, min_y_ + ny_-1, min_z_ + nz_-1);
  Point p7(min_x_,         min_y_ + ny_-1, min_z_ + nz_-1);
  
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


void
LatVolMesh::transform(Transform &t)
{
  transform_.pre_trans(t);
}


void
LatVolMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.resize(2);
  const unsigned int xidx = idx;
  if (xidx < (nx_ - 1) * ny_ * nz_)
  {
    const int i = xidx % (nx_ - 1);
    const int jk = xidx / (nx_ - 1);
    const int j = jk % ny_;
    const int k = jk / ny_;

    array[0] = Node::index_type(this, i+0, j, k);
    array[1] = Node::index_type(this, i+1, j, k);
  }
  else
  {
    const unsigned int yidx = idx - (nx_ - 1) * ny_ * nz_;
    if (yidx < (nx_ * (ny_ - 1) * nz_))
    {
      const int j = yidx % (ny_ - 1);
      const int ik = yidx / (ny_ - 1);
      const int i = ik / nz_;
      const int k = ik % nz_;

      array[0] = Node::index_type(this, i, j+0, k);
      array[1] = Node::index_type(this, i, j+1, k);
    }
    else
    {
      const unsigned int zidx = yidx - (nx_ * (ny_ - 1) * nz_);
      const int k = zidx % (nz_ - 1);
      const int ij = zidx / (nz_ - 1);
      const int i = ij % nx_;
      const int j = ij / nx_;

      array[0] = Node::index_type(this, i, j, k+0);
      array[1] = Node::index_type(this, i, j, k+1);
    }      
  }
}


void
LatVolMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.resize(4);
  const unsigned int xidx = idx;
  if (xidx < (nx_ - 1) * (ny_ - 1) * nz_)
  {
    const int i = xidx % (nx_ - 1);
    const int jk = xidx / (nx_ - 1);
    const int j = jk % (ny_ - 1);
    const int k = jk / (ny_ - 1);
    array[0] = Node::index_type(this, i+0, j+0, k);
    array[1] = Node::index_type(this, i+1, j+0, k);
    array[2] = Node::index_type(this, i+1, j+1, k);
    array[3] = Node::index_type(this, i+0, j+1, k);
  }
  else
  {
    const unsigned int yidx = idx - (nx_ - 1) * (ny_ - 1) * nz_;
    if (yidx < nx_ * (ny_ - 1) * (nz_ - 1))
    {
      const int j = yidx % (ny_ - 1);
      const int ik = yidx / (ny_ - 1);
      const int k = ik % (nz_ - 1);
      const int i = ik / (nz_ - 1);
      array[0] = Node::index_type(this, i, j+0, k+0);
      array[1] = Node::index_type(this, i, j+1, k+0);
      array[2] = Node::index_type(this, i, j+1, k+1);
      array[3] = Node::index_type(this, i, j+0, k+1);
    }
    else
    {
      const unsigned int zidx = yidx - nx_ * (ny_ - 1) * (nz_ - 1);
      const int k = zidx % (nz_ - 1);
      const int ij = zidx / (nz_ - 1);
      const int i = ij % (nx_ - 1);
      const int j = ij / (nx_ - 1);
      array[0] = Node::index_type(this, i+0, j, k+0);
      array[1] = Node::index_type(this, i+0, j, k+1);
      array[2] = Node::index_type(this, i+1, j, k+1);
      array[3] = Node::index_type(this, i+1, j, k+0);
    }
  }
}


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
}


void
LatVolMesh::get_faces(Face::array_type &array, Cell::index_type idx) const
{
  array.resize(6);

  const unsigned int i = idx.i_;
  const unsigned int j = idx.j_;
  const unsigned int k = idx.k_;

  const unsigned int offset1 = (nx_ - 1) * (ny_ - 1) * nz_;
  const unsigned int offset2 = offset1 + nx_ * (ny_ - 1) * (nz_ - 1);

  array[0] = i + (j + k * (ny_-1)) * (nx_-1);
  array[1] = i + (j + (k+1) * (ny_-1)) * (nx_-1);

  array[2] = offset1 + j + (k + i * (nz_-1)) * (ny_-1);
  array[3] = offset1 + j + (k + (i+1) * (nz_-1)) * (ny_-1);

  array[4] = offset2 + k + (i + j * (nx_-1)) * (nz_-1);
  array[5] = offset2 + k + (i + (j+1) * (nx_-1)) * (nz_-1);
}


//! return all cell_indecies that overlap the BBox in arr.

void
LatVolMesh::get_cells(Cell::array_type &arr, const BBox &bbox)
{
  arr.clear();
  Cell::index_type min;
  locate(min, bbox.min());
  Cell::index_type max;
  locate(max, bbox.max());

  if (max.i_ >= nx_ - 1) max.i_ = Max(((int)nx_) - 2, 0);
  if (max.j_ >= ny_ - 1) max.j_ = Max(((int)ny_) - 2, 0);
  if (max.k_ >= nz_ - 1) max.k_ = Max(((int)nz_) - 2, 0);

  for (unsigned i = min.i_; i <= max.i_; i++) {
    for (unsigned j = min.j_; j <= max.j_; j++) {
      for (unsigned k = min.k_; k <= max.k_; k++) {
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
  if (xidx < (nx_ - 1) * (ny_ - 1) * nz_)
  {
    //const unsigned int i = xidx % (nx_ - 1);
    const unsigned int jk = xidx / (nx_ - 1);
    //const unsigned int j = jk % (ny_ - 1);
    const unsigned int k = jk / (ny_ - 1);

    if (k == from.k_ && k > 0)
    {
      neighbor.i_ = from.i_;
      neighbor.j_ = from.j_;
      neighbor.k_ = k-1;
      return true;
    }
    else if (k == (from.k_+1) && k < (nz_-1))
    {
      neighbor.i_ = from.i_;
      neighbor.j_ = from.j_;
      neighbor.k_ = k;
      return true;
    }
  }
  else
  {
    const unsigned int yidx = xidx - (nx_ - 1) * (ny_ - 1) * nz_;
    if (yidx < nx_ * (ny_ - 1) * (nz_ - 1))
    {
      //const unsigned int j = yidx % (ny_ - 1);
      const unsigned int ik = yidx / (ny_ - 1);
      //const unsigned int k = ik % (nz_ - 1);
      const unsigned int i = ik / (nz_ - 1);

      if (i == from.i_ && i > 0)
      {
	neighbor.i_ = i-1;
	neighbor.j_ = from.j_;
	neighbor.k_ = from.k_;
	return true;
      }
      else if (i == (from.i_+1) && i < (nx_-1))
      {
	neighbor.i_ = i;
	neighbor.j_ = from.j_;
	neighbor.k_ = from.k_;
	return true;
      }
    }
    else
    {
      const unsigned int zidx = yidx - nx_ * (ny_ - 1) * (nz_ - 1);
      //const unsigned int k = zidx % (nz_ - 1);
      const unsigned int ij = zidx / (nz_ - 1);
      //const unsigned int i = ij % (nx_ - 1);
      const unsigned int j = ij / (nx_ - 1);

      if (j == from.j_ && j > 0)
      {
	neighbor.i_ = from.i_;
	neighbor.j_ = j-1;
	neighbor.k_ = from.k_;
	return true;
      }
      else if (j == (from.j_+1) && j < (ny_-1))
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
  
  Cell::index_type min_index, max_index;

  // If one of the locates return true, then we have a valid iteration
  bool min_located = locate(min_index, min);
  bool max_located = locate(max_index, max);
  if (min_located || max_located) {
    // Initialize the range iterator
    iter = Cell::range_iter(this,
			    min_index.i_, min_index.j_, min_index.k_,
			    max_index.i_, max_index.j_, max_index.k_);
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
  
  Node::index_type min_index, max_index;

  // If one of the locates return true, then we have a valid iteration
  bool min_located = locate(min_index, min);
  bool max_located = locate(max_index, max);
  if (min_located || max_located) {
    // Initialize the range iterator
    iter = Node::range_iter(this,
			    min_index.i_, min_index.j_, min_index.k_,
			    max_index.i_, max_index.j_, max_index.k_);
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
  Point p0, p1;
  get_center(p0, arr[0]);
  get_center(p1, arr[1]);

  result = (p0.asVector() + p1.asVector() * 0.5).asPoint();
}


void
LatVolMesh::get_center(Point &/*result*/, Face::index_type /*idx*/) const
{
#if 0 // TODO: Fix get_nodes
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
  result = v.asPoint();
#endif
}


void
LatVolMesh::get_center(Point &result, Cell::index_type idx) const
{
  Point p(idx.i_ + 0.5, idx.j_ + 0.5, idx.k_ + 0.5);
  result = transform_.project(p);
}

void
LatVolMesh::get_center(Point &result, Node::index_type idx) const
{
  Point p(idx.i_, idx.j_, idx.k_);
  result = transform_.project(p);
}


bool
LatVolMesh::locate(Cell::index_type &cell, const Point &p)
{
  const Point r = transform_.unproject(p);

  // Rounds down, so catches intervals.  Might lose numerical precision on
  // upper edge (ie nodes on upper edges are not in any cell).
  // Nodes over 2 billion might suffer roundoff error.
  cell.i_ = (unsigned int)r.x();
  cell.j_ = (unsigned int)r.y();
  cell.k_ = (unsigned int)r.z();

  if (cell.i_ >= (nx_-1) ||
      cell.j_ >= (ny_-1) ||
      cell.k_ >= (nz_-1))
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
  node.i_ = (unsigned int)(r.x() + 0.5);
  node.j_ = (unsigned int)(r.y() + 0.5);
  node.k_ = (unsigned int)(r.z() + 0.5);

  if (node.i_ >= nx_ ||
      node.j_ >= ny_ ||
      node.k_ >= nz_)
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
			Node::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    get_nodes( l, idx );
    w.resize(l.size());
    vector<double>::iterator wit = w.begin();
    Node::array_type::iterator it = l.begin();

    Point np, pmin, pmax;
    get_point(pmin, l[0]);
    get_point(pmax, l[6]);

    Vector diag(pmax - pmin);

    while( it != l.end()) {
      Node::index_type ni = *it;
      ++it;
      get_point(np, ni);
      *wit = ( 1 - fabs(p.x() - np.x())/diag.x() ) *
	( 1 - fabs(p.y() - np.y())/diag.y() ) *
	( 1 - fabs(p.z() - np.z())/diag.z() );
      ++wit;
    }
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
  Pio(stream, nx_);
  Pio(stream, ny_);
  Pio(stream, nz_);

  if (version < 2 && stream.reading())
  {
    Point min, max;
    Pio(stream, min);
    Pio(stream, max);
    transform_.pre_scale(Vector(1.0 / (nx_ - 1.0),
				1.0 / (ny_ - 1.0),
				1.0 / (nz_ - 1.0)));
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
  itr = Node::iterator(this, min_x_, min_y_, min_z_);
}

void
LatVolMesh::end(LatVolMesh::Node::iterator &itr) const
{
  itr = Node::iterator(this, min_x_, min_y_, min_z_ + nz_);
}

void
LatVolMesh::size(LatVolMesh::Node::size_type &s) const
{
  s = Node::size_type(nx_,ny_,nz_);
}

void
LatVolMesh::begin(LatVolMesh::Edge::iterator &itr) const
{
  itr = Edge::iterator(0);
}

void
LatVolMesh::end(LatVolMesh::Edge::iterator &itr) const
{
  itr = ((nx_-1) * ny_ * nz_) + (nx_ * (ny_-1) * nz_) + (nx_ * ny_ * (nz_-1));
}

void
LatVolMesh::size(LatVolMesh::Edge::size_type &s) const
{
  s = ((nx_-1) * ny_ * nz_) + (nx_ * (ny_-1) * nz_) + (nx_ * ny_ * (nz_-1));
}

void
LatVolMesh::begin(LatVolMesh::Face::iterator &itr) const
{
  itr = Face::iterator(0);
}

void
LatVolMesh::end(LatVolMesh::Face::iterator &itr) const
{
  itr = (nx_-1) * (ny_-1) * nz_ +
    nx_ * (ny_ - 1 ) * (nz_ - 1) +
    (nx_ - 1) * ny_ * (nz_ - 1);
}

void
LatVolMesh::size(LatVolMesh::Face::size_type &s) const
{
  s =  (nx_-1) * (ny_-1) * nz_ +
    nx_ * (ny_ - 1 ) * (nz_ - 1) +
    (nx_ - 1) * ny_ * (nz_ - 1);
}

void
LatVolMesh::begin(LatVolMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(this,  min_x_, min_y_, min_z_);
}

void
LatVolMesh::end(LatVolMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(this, min_x_, min_y_, min_z_ + nz_-1);
}

void
LatVolMesh::size(LatVolMesh::Cell::size_type &s) const
{
  s = Cell::size_type(nx_-1, ny_-1,nz_-1);
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
