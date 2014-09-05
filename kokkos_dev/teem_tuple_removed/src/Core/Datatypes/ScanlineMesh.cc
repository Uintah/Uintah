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
 *  ScanlineMesh.cc: Templated Mesh defined on a 3D Regular Grid
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

#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Geometry/BBox.h>
#include <iostream>
#include <vector>


namespace SCIRun {

using namespace std;

PersistentTypeID ScanlineMesh::type_id("ScanlineMesh", "Mesh", maker);


ScanlineMesh::ScanlineMesh(unsigned int ni,
			   const Point &min, const Point &max)
  : min_i_(0), ni_(ni)
{
  transform_.pre_scale(Vector(1.0 / (ni_ - 1.0), 1.0, 1.0));
  transform_.pre_scale(max - min);
  transform_.pre_translate(Vector(min));
  transform_.compute_imat();
}

BBox
ScanlineMesh::get_bounding_box() const
{
  Point p0(0.0, 0.0, 0.0);
  Point p1(ni_ - 1, 0.0, 0.0);
  
  BBox result;
  result.extend(transform_.project(p0));
  result.extend(transform_.project(p1));
  return result;
}

Vector ScanlineMesh::diagonal() const
{
  return get_bounding_box().diagonal();
}

void
ScanlineMesh::transform(const Transform &t)
{
  transform_.pre_trans(t);
}

void 
ScanlineMesh::get_canonical_transform(Transform &t) 
{
  t = transform_;
  t.post_scale(Vector(ni_ - 1.0, 1.0, 1.0));
}

bool ScanlineMesh::get_min(vector<unsigned int> &array ) const
{
  array.resize(1);
  array.clear();

  array.push_back(min_i_);

  return true;
}

bool
ScanlineMesh::get_dim(vector<unsigned int> &array) const
{
  array.resize(1);
  array.clear();

  array.push_back(ni_);

  return true;
}

void
ScanlineMesh::set_min(vector<unsigned int> min)
{
  min_i_ = min[0];
}

void
ScanlineMesh::set_dim(vector<unsigned int> dim)
{
  ni_ = dim[0];
}

void
ScanlineMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.resize(2);
  array[0] = Node::index_type(idx);
  array[1] = Node::index_type(idx + 1);
}

//! return all cell_indecies that overlap the BBox in arr.
void
ScanlineMesh::get_edges(Edge::array_type &/* arr */, const BBox &/*bbox*/) const
{
  // TODO: implement this
}


void
ScanlineMesh::get_center(Point &result, Node::index_type idx) const
{
  Point p(idx, 0.0, 0.0);
  result = transform_.project(p);
}


void
ScanlineMesh::get_center(Point &result, Edge::index_type idx) const
{
  Point p(idx + 0.5, 0.0, 0.0);
  result = transform_.project(p);
}

// TODO: verify
bool
ScanlineMesh::locate(Edge::index_type &elem, const Point &p)
{
  const Point r = transform_.unproject(p);
  elem = (unsigned int)(r.x());

  if (elem >= (ni_ - 1))
  {
    return false;
  }
  else
  {
    return true;
  }
}


// TODO: verify
bool
ScanlineMesh::locate(Node::index_type &node, const Point &p)
{
  const Point r = transform_.unproject(p);
  node = (unsigned int)(r.x() + 0.5);

  if (node >= ni_)
  {
    return false;
  }
  else
  {
    return true;
  }
}


void
ScanlineMesh::get_weights(const Point &p,
			  Node::array_type &locs, vector<double> &weights)
{
  const Point r = transform_.unproject(p);
  Node::index_type node0, node1;

  double ii=r.x();
  if (ii>(ni_-1) && (ii-(1.e-10))<(ni_-1)) ii=ni_-1-(1.e-10);
  if (ii<0 && ii>(-1.e-10)) ii=0;
  node0 = (unsigned int)floor(ii);

  if (node0 < (ni_-1) && node0 >= 0) 
  {
    const double dx1 = ii - node0;
    const double dx0 = 1.0 - dx1;

    node1 = node0 + 1;
    
    locs.push_back(node0);
    locs.push_back(node1);
    
    weights.push_back(dx0);
    weights.push_back(dx1);
  }
}


void
ScanlineMesh::get_weights(const Point &p,
			  Edge::array_type &l, vector<double> &w)
{
  Edge::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}

#define SCANLINEMESH_VERSION 2

void
ScanlineMesh::io(Piostream& stream)
{
  int version = stream.begin_class(type_name(-1), SCANLINEMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, ni_);
  if (version < 2 && stream.reading() ) {
    Pio_old(stream, transform_);
  } else {
    Pio(stream, transform_);
  }
  stream.end_class();
}

const string
ScanlineMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "ScanlineMesh";
  return name;
}


void
ScanlineMesh::begin(ScanlineMesh::Node::iterator &itr) const
{
  itr = Node::iterator(min_i_);
}

void
ScanlineMesh::end(ScanlineMesh::Node::iterator &itr) const
{
  itr = Node::iterator(min_i_ + ni_);
}

void
ScanlineMesh::size(ScanlineMesh::Node::size_type &s) const
{
  s = Node::size_type(ni_);
}

void
ScanlineMesh::begin(ScanlineMesh::Edge::iterator &itr) const
{
  itr = Edge::iterator(min_i_);
}

void
ScanlineMesh::end(ScanlineMesh::Edge::iterator &itr) const
{
  itr = Edge::iterator(min_i_+ni_-1);
}

void
ScanlineMesh::size(ScanlineMesh::Edge::size_type &s) const
{
  s = Edge::size_type(ni_ - 1);
}

void
ScanlineMesh::begin(ScanlineMesh::Face::iterator &itr) const
{
  itr = Face::iterator(0);
}

void
ScanlineMesh::end(ScanlineMesh::Face::iterator &itr) const
{
  itr = Face::iterator(0);
}

void
ScanlineMesh::size(ScanlineMesh::Face::size_type &s) const
{
  s = Face::size_type(0);
}

void
ScanlineMesh::begin(ScanlineMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(0);
}

void
ScanlineMesh::end(ScanlineMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(0);
}

void
ScanlineMesh::size(ScanlineMesh::Cell::size_type &s) const
{
  s = Cell::size_type(0);
}


const TypeDescription*
ScanlineMesh::get_type_description() const
{
  return SCIRun::get_type_description((ScanlineMesh *)0);
}

const TypeDescription*
get_type_description(ScanlineMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ScanlineMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ScanlineMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ScanlineMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ScanlineMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ScanlineMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ScanlineMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ScanlineMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ScanlineMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ScanlineMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
