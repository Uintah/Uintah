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
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID LatVolMesh::type_id("LatVolMesh", "Mesh", maker);

void LatVolMesh::get_random_point(Point &p, const Elem::index_type &ei,
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
  BBox result;
  result.extend(min_);
  result.extend(max_);
  return result;
}


// TODO: Fix this
void
LatVolMesh::transform(Transform &t)
{
  ASSERTFAIL("Fix this when latvolmesh is transformable");
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

//! return all cell_indecies that overlap the BBox in arr.
void
LatVolMesh::get_cells(Cell::array_type &arr, const BBox &bbox) const
{
  arr.clear();
  Cell::index_type min;
  locate(min, bbox.min());
  Cell::index_type max;
  locate(max, bbox.max());

  if (max.i_ >= nx_ - 1) max.i_ = nx_ - 2;
  if (max.j_ >= ny_ - 1) max.j_ = ny_ - 2;
  if (max.k_ >= nz_ - 1) max.k_ = nz_ - 2;

  for (unsigned i = min.i_; i <= max.i_; i++) {
    for (unsigned j = min.j_; j <= max.j_; j++) {
      for (unsigned k = min.k_; k <= max.k_; k++) {
	arr.push_back(Cell::index_type(i,j,k));
      }
    }
  }
}


void
LatVolMesh::get_center(Point &result, Node::index_type idx) const
{
  const double sx = (max_.x() - min_.x()) / (nx_ - 1);
  const double sy = (max_.y() - min_.y()) / (ny_ - 1);
  const double sz = (max_.z() - min_.z()) / (nz_ - 1);

  result.x(idx.i_ * sx + min_.x());
  result.y(idx.j_ * sy + min_.y());
  result.z(idx.k_ * sz + min_.z());
}


void
LatVolMesh::get_center(Point &result, Cell::index_type idx) const
{
  const double sx = (max_.x() - min_.x()) / (nx_ - 1);
  const double sy = (max_.y() - min_.y()) / (ny_ - 1);
  const double sz = (max_.z() - min_.z()) / (nz_ - 1);

  result.x((idx.i_ + 0.5) * sx + min_.x());
  result.y((idx.j_ + 0.5) * sy + min_.y());
  result.z((idx.k_ + 0.5) * sz + min_.z());
}

bool
LatVolMesh::locate(Cell::index_type &cell, const Point &p) const
{
  double i = (p.x() - min_.x()) / (max_.x() - min_.x()) * (nx_ - 1) + 0.5;
  double j = (p.y() - min_.y()) / (max_.y() - min_.y()) * (ny_ - 1) + 0.5;
  double k = (p.z() - min_.z()) / (max_.z() - min_.z()) * (nz_ - 1) + 0.5;

  cell.i_ = (unsigned int)i;
  cell.j_ = (unsigned int)j;
  cell.k_ = (unsigned int)k;

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
LatVolMesh::locate(Node::index_type &node, const Point &p) const
{
  Node::array_type nodes;     // storage for node_indeces
  Cell::index_type cell;
  double max;
  int loop;

  // locate the cell enclosing the point (including weights)
  if (!locate(cell,p)) return false;
  weight_array w;
  calc_weights(this, cell, p, w);

  // get the node_indeces in this cell
  get_nodes(nodes,cell);

  // find, and return, the "heaviest" node
  max = w[0];
  loop=1;
  while (loop<8) {
    if (w[loop]>max) {
      max=w[loop];
      node=nodes[loop];
    }
  }
  return true;
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
				LatVolMesh::get_h_file_path(),
				"SCIRun");
  }
  return td;
}
const TypeDescription* get_type_description(LatVolMesh::EdgeIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("LatVolMesh::EdgeIndex",
				LatVolMesh::get_h_file_path(),
				"SCIRun");
}
  return td;
}
const TypeDescription* get_type_description(LatVolMesh::FaceIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("LatVolMesh::FaceIndex",
				LatVolMesh::get_h_file_path(),
				"SCIRun");
  }
  return td;
}
const TypeDescription* get_type_description(LatVolMesh::CellIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("LatVolMesh::CellIndex",
				LatVolMesh::get_h_file_path(),
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
Pio(Piostream& stream, LatVolMesh::EdgeIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    stream.end_cheap_delim();
}

void
Pio(Piostream& stream, LatVolMesh::FaceIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
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
const string find_type_name(LatVolMesh::EdgeIndex *)
{
  static string name = "LatVolMesh::EdgeIndex";
  return name;
}
const string find_type_name(LatVolMesh::FaceIndex *)
{
  static string name = "LatVolMesh::FaceIndex";
  return name;
}
const string find_type_name(LatVolMesh::CellIndex *)
{
  static string name = "LatVolMesh::CellIndex";
  return name;
}

#define LATVOLMESH_VERSION 1

void
LatVolMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), LATVOLMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, nx_);
  Pio(stream, ny_);
  Pio(stream, nz_);
  Pio(stream, min_);
  Pio(stream, max_);

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
  itr = Edge::iterator(this,0);
}

void
LatVolMesh::end(LatVolMesh::Edge::iterator &itr) const
{
  itr = Edge::iterator(this,0);
}

void
LatVolMesh::size(LatVolMesh::Edge::size_type &s) const
{
  s = Edge::size_type(0);
}

void
LatVolMesh::begin(LatVolMesh::Face::iterator &itr) const
{
  itr = Face::iterator(this,0);
}

void
LatVolMesh::end(LatVolMesh::Face::iterator &itr) const
{
  itr = Face::iterator(this,0);
}

void
LatVolMesh::size(LatVolMesh::Face::size_type &s) const
{
  s = Face::size_type(0);
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


const string& 
LatVolMesh::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
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
