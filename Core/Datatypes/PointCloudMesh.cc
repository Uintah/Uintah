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
 *  PointCloudMesh.cc: PointCloud mesh
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Vector.h>
#include <float.h>  // for DBL_MAX
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID PointCloudMesh::type_id("PointCloudMesh", "Mesh", maker);

BBox
PointCloudMesh::get_bounding_box() const
{
  BBox result;

  for (Node::iterator i = node_begin();
       i!=node_end();
       ++i)
    result.extend(points_[*i]);

  return result;
}


void
PointCloudMesh::transform(Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
}


bool
PointCloudMesh::locate(Node::index_type &idx, const Point &p) const
{
  Node::iterator ni = node_begin();
  idx = *node_begin();

  if (ni==node_end())
    return false;

  double closest = (p-points_[*ni]).length2();

  ++ni;
  for (; ni != node_end(); ++ni) {
    if ( (p-points_[*ni]).length2() < closest ) {
      closest = (p-points_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}


void
PointCloudMesh::get_weights(const Point &p,
			    Node::array_type &l, vector<double> &w)
{
  Node::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}


PointCloudMesh::Node::index_type
PointCloudMesh::add_point(const Point &p)
{
  points_.push_back(p);
  return points_.size() - 1;
}

#define PointCloudMESH_VERSION 1

void
PointCloudMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), PointCloudMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream,points_);

  stream.end_class();
}

const string
PointCloudMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "PointCloudMesh";
  return name;
}

template<>
PointCloudMesh::Node::iterator
PointCloudMesh::tbegin(PointCloudMesh::Node::iterator *) const
{
  return 0;
}

template<>
PointCloudMesh::Node::iterator
PointCloudMesh::tend(PointCloudMesh::Node::iterator *) const
{
  return (unsigned)points_.size();
}

template<>
PointCloudMesh::Node::size_type
PointCloudMesh::tsize(PointCloudMesh::Node::size_type *) const
{
  return (unsigned)points_.size();
}

template<>
PointCloudMesh::Edge::iterator
PointCloudMesh::tbegin(PointCloudMesh::Edge::iterator *) const
{
  return 0;
}

template<>
PointCloudMesh::Edge::iterator
PointCloudMesh::tend(PointCloudMesh::Edge::iterator *) const
{
  return 0;
}

template<>
PointCloudMesh::Edge::size_type
PointCloudMesh::tsize(PointCloudMesh::Edge::size_type *) const
{
  return 0;
}

template<>
PointCloudMesh::Face::iterator
PointCloudMesh::tbegin(PointCloudMesh::Face::iterator *) const
{
  return 0;
}

template<>
PointCloudMesh::Face::iterator
PointCloudMesh::tend(PointCloudMesh::Face::iterator *) const
{
  return 0;
}

template<>
PointCloudMesh::Face::size_type
PointCloudMesh::tsize(PointCloudMesh::Face::size_type *) const
{
  return 0;
}

template<>
PointCloudMesh::Cell::iterator
PointCloudMesh::tbegin(PointCloudMesh::Cell::iterator *) const
{
  return 0;
}

template<>
PointCloudMesh::Cell::iterator
PointCloudMesh::tend(PointCloudMesh::Cell::iterator *) const
{
  return 0;
}

template<>
PointCloudMesh::Cell::size_type
PointCloudMesh::tsize(PointCloudMesh::Cell::size_type *) const
{
  return 0;
}


PointCloudMesh::Node::iterator PointCloudMesh::node_begin() const
{ return tbegin((Node::iterator *)0); }
PointCloudMesh::Edge::iterator PointCloudMesh::edge_begin() const
{ return tbegin((Edge::iterator *)0); }
PointCloudMesh::Face::iterator PointCloudMesh::face_begin() const
{ return tbegin((Face::iterator *)0); }
PointCloudMesh::Cell::iterator PointCloudMesh::cell_begin() const
{ return tbegin((Cell::iterator *)0); }

PointCloudMesh::Node::iterator PointCloudMesh::node_end() const
{ return tend((Node::iterator *)0); }
PointCloudMesh::Edge::iterator PointCloudMesh::edge_end() const
{ return tend((Edge::iterator *)0); }
PointCloudMesh::Face::iterator PointCloudMesh::face_end() const
{ return tend((Face::iterator *)0); }
PointCloudMesh::Cell::iterator PointCloudMesh::cell_end() const
{ return tend((Cell::iterator *)0); }

PointCloudMesh::Node::size_type PointCloudMesh::nodes_size() const
{ return tsize((Node::size_type *)0); }
PointCloudMesh::Edge::size_type PointCloudMesh::edges_size() const
{ return tsize((Edge::size_type *)0); }
PointCloudMesh::Face::size_type PointCloudMesh::faces_size() const
{ return tsize((Face::size_type *)0); }
PointCloudMesh::Cell::size_type PointCloudMesh::cells_size() const
{ return tsize((Cell::size_type *)0); }

const TypeDescription*
PointCloudMesh::get_type_description() const
{
  return SCIRun::get_type_description((PointCloudMesh *)0);
}

const TypeDescription*
get_type_description(PointCloudMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(PointCloudMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(PointCloudMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(PointCloudMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(PointCloudMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
