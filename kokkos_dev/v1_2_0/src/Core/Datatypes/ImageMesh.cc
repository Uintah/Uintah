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
 *  ImageMesh.cc: Templated Mesh defined on a 3D Regular Grid
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

#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID ImageMesh::type_id("ImageMesh", "MeshBase", maker);


ImageMesh::ImageMesh(unsigned x, unsigned y,
		     const Point &min, const Point &max)
  : min_x_(0), min_y_(0), nx_(x), ny_(y)
{
  transform_.pre_scale(Vector(1.0 / (x-1.0), 1.0 / (y-1.0), 1.0));
  transform_.pre_scale(max - min);
  transform_.pre_translate(Vector(min));
  transform_.compute_imat();
}



BBox
ImageMesh::get_bounding_box() const
{
  Point p0(0.0, 0.0, 0.0);
  Point p1(nx_, 0.0, 0.0);
  Point p2(nx_, ny_, 0.0);
  Point p3(0.0, ny_, 0.0);
  
  BBox result;
  result.extend(transform_.project(p0));
  result.extend(transform_.project(p1));
  result.extend(transform_.project(p2));
  result.extend(transform_.project(p3));
  return result;
}


void
ImageMesh::transform(Transform &t)
{
  transform_.pre_trans(t);
}



void
ImageMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.resize(4);
  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1;
}

//! return all face_indecies that overlap the BBox in arr.
void
ImageMesh::get_faces(Face::array_type &arr, const BBox &bbox)
{
  arr.clear();
  Face::index_type min;
  locate(min, bbox.min());
  Face::index_type max;
  locate(max, bbox.max());

  if (max.i_ >= nx_ - 1) max.i_ = nx_ - 2;
  if (max.j_ >= ny_ - 1) max.j_ = ny_ - 2;

  for (unsigned i = min.i_; i <= max.i_; i++) {
    for (unsigned j = min.j_; j <= max.j_; j++) {
      arr.push_back(Face::index_type(i,j));
    }
  }
}


void
ImageMesh::get_center(Point &result, Node::index_type idx) const
{
  Point p(idx.i_, idx.j_, 0.0);
  result = transform_.project(p);
}


void
ImageMesh::get_center(Point &result, Face::index_type idx) const
{
  Point p(idx.i_ + 0.5, idx.j_ + 0.5, 0.0);
  result = transform_.project(p);
}

bool
ImageMesh::locate(Face::index_type &face, const Point &p)
{
  const Point r = transform_.unproject(p);

  face.i_ = (unsigned int)r.x();
  face.j_ = (unsigned int)r.y();

  if (face.i_ >= (nx_-1) ||
      face.j_ >= (ny_-1))
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool
ImageMesh::locate(Node::index_type &node, const Point &p)
{
  Node::array_type nodes;     // storage for node_indeces
  Face::index_type face;
  double max;
  int loop;

  // locate the face enclosing the point (including weights)
  if (!locate(face,p)) return false;
  weight_array w;
  calc_weights(this, face, p, w);

  // get the node_indeces in this face
  get_nodes(nodes,face);

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
ImageMesh::get_weights(const Point &p,
		       Node::array_type &locs, vector<double> &weights)
{
  const Point r = transform_.unproject(p);
  Node::index_type node0, node1, node2, node3;

  node0.i_ = (unsigned int)r.x();
  node0.j_ = (unsigned int)r.y();

  if (node0.i_ >= 0 && node0.i_ < (nx_-1) ||
      node0.j_ >= 0 && node0.j_ < (ny_-1))
  {
    const double dx1 = r.x() - node0.i_;
    const double dy1 = r.y() - node0.j_;
    const double dx0 = 1.0 - dx1;
    const double dy0 = 1.0 - dy1;

    node1.i_ = node0.i_ + 1;
    node1.j_ = node0.j_ + 0;

    node2.i_ = node0.i_ + 1;
    node2.j_ = node0.j_ + 1;

    node3.i_ = node0.i_ + 0;
    node3.j_ = node0.j_ + 1;

    locs.push_back(node0);
    locs.push_back(node1);
    locs.push_back(node2);
    locs.push_back(node3);

    weights.push_back(dx0 * dy0);
    weights.push_back(dx1 * dy0);
    weights.push_back(dx1 * dy1);
    weights.push_back(dx0 * dy1);
  }
}


void
ImageMesh::get_weights(const Point &p,
		       Face::array_type &l, vector<double> &w)
{
  Face::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}


const TypeDescription* get_type_description(ImageMesh::NodeIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("ImageMesh::NodeIndex",
				ImageMesh::get_h_file_path(),
				"SCIRun");
  }
  return td;
}
const TypeDescription* get_type_description(ImageMesh::EdgeIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("ImageMesh::EdgeIndex",
				ImageMesh::get_h_file_path(),
				"SCIRun");
}
  return td;
}
const TypeDescription* get_type_description(ImageMesh::FaceIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("ImageMesh::FaceIndex",
				ImageMesh::get_h_file_path(),
				"SCIRun");
  }
  return td;
}
const TypeDescription* get_type_description(ImageMesh::CellIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("ImageMesh::CellIndex",
				ImageMesh::get_h_file_path(),
				"SCIRun");
  }
  return td;
}


void
Pio(Piostream& stream, ImageMesh::NodeIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    stream.end_cheap_delim();
}

void
Pio(Piostream& stream, ImageMesh::EdgeIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    stream.end_cheap_delim();
}

void
Pio(Piostream& stream, ImageMesh::FaceIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    stream.end_cheap_delim();
}

void
Pio(Piostream& stream, ImageMesh::CellIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    stream.end_cheap_delim();
}


const string find_type_name(ImageMesh::NodeIndex *)
{
  static string name = "ImageMesh::NodeIndex";
  return name;
}
const string find_type_name(ImageMesh::EdgeIndex *)
{
  static string name = "ImageMesh::EdgeIndex";
  return name;
}
const string find_type_name(ImageMesh::FaceIndex *)
{
  static string name = "ImageMesh::FaceIndex";
  return name;
}
const string find_type_name(ImageMesh::CellIndex *)
{
  static string name = "ImageMesh::CellIndex";
  return name;
}


#define LATVOLMESH_VERSION 1

void
ImageMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), LATVOLMESH_VERSION);

  MeshBase::io(stream);

  // IO data members, in order
  Pio(stream, nx_);
  Pio(stream, ny_);

  stream.end_class();
}

const string
ImageMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "ImageMesh";
  return name;
}

template<>
ImageMesh::Node::iterator
ImageMesh::tbegin(ImageMesh::Node::iterator *) const
{
  return Node::iterator(this, min_x_, min_y_);
}

template<>
ImageMesh::Node::iterator
ImageMesh::tend(ImageMesh::Node::iterator *) const
{
  return Node::iterator(this, min_x_, min_y_ + ny_);
}

template<>
ImageMesh::Node::size_type
ImageMesh::tsize(ImageMesh::Node::size_type *) const
{
  return Node::size_type(nx_, ny_);
}


template<>
ImageMesh::Edge::iterator
ImageMesh::tbegin(ImageMesh::Edge::iterator *) const
{
  return Edge::iterator(this, 0);
}

template<>
ImageMesh::Edge::iterator
ImageMesh::tend(ImageMesh::Edge::iterator *) const
{
  return Edge::iterator(this, 0);
}

template<>
ImageMesh::Edge::size_type
ImageMesh::tsize(ImageMesh::Edge::size_type *) const
{
  return Edge::size_type(0);
}

template<>
ImageMesh::Face::iterator
ImageMesh::tbegin(ImageMesh::Face::iterator *) const
{
  return Face::iterator(this,  min_x_, min_y_);
}

template<>
ImageMesh::Face::iterator
ImageMesh::tend(ImageMesh::Face::iterator *) const
{
  return Face::iterator(this, min_x_, min_y_ + ny_ - 1);
}

template<>
ImageMesh::Face::size_type
ImageMesh::tsize(ImageMesh::Face::size_type *) const
{
  return Face::size_type(nx_-1, ny_-1);
}


template<>
ImageMesh::Cell::iterator
ImageMesh::tbegin(ImageMesh::Cell::iterator *) const
{
  return Cell::iterator(this, 0);
}

template<>
ImageMesh::Cell::iterator
ImageMesh::tend(ImageMesh::Cell::iterator *) const
{
  return Cell::iterator(this, 0);
}

template<>
ImageMesh::Cell::size_type
ImageMesh::tsize(ImageMesh::Cell::size_type *) const
{
  return Cell::size_type(0);
}


ImageMesh::Node::iterator ImageMesh::node_begin() const
{ return tbegin((Node::iterator *)0); }
ImageMesh::Edge::iterator ImageMesh::edge_begin() const
{ return tbegin((Edge::iterator *)0); }
ImageMesh::Face::iterator ImageMesh::face_begin() const
{ return tbegin((Face::iterator *)0); }
ImageMesh::Cell::iterator ImageMesh::cell_begin() const
{ return tbegin((Cell::iterator *)0); }

ImageMesh::Node::iterator ImageMesh::node_end() const
{ return tend((Node::iterator *)0); }
ImageMesh::Edge::iterator ImageMesh::edge_end() const
{ return tend((Edge::iterator *)0); }
ImageMesh::Face::iterator ImageMesh::face_end() const
{ return tend((Face::iterator *)0); }
ImageMesh::Cell::iterator ImageMesh::cell_end() const
{ return tend((Cell::iterator *)0); }

ImageMesh::Node::size_type ImageMesh::nodes_size() const
{ return tsize((Node::size_type *)0); }
ImageMesh::Edge::size_type ImageMesh::edges_size() const
{ return tsize((Edge::size_type *)0); }
ImageMesh::Face::size_type ImageMesh::faces_size() const
{ return tsize((Face::size_type *)0); }
ImageMesh::Cell::size_type ImageMesh::cells_size() const
{ return tsize((Cell::size_type *)0); }

const TypeDescription*
ImageMesh::get_type_description() const
{
  return SCIRun::get_type_description((ImageMesh *)0);
}

const TypeDescription*
get_type_description(ImageMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ImageMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ImageMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ImageMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ImageMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
