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
 *  ContourMesh.cc: contour mesh
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

#include <Core/Datatypes/ContourMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Vector.h>
#include <float.h>  // for DBL_MAX
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID ContourMesh::type_id("ContourMesh", "Mesh", maker);

BBox
ContourMesh::get_bounding_box() const
{
  BBox result;

  for (Node::iterator i = node_begin();
       i!=node_end();
       ++i)
    result.extend(nodes_[*i]);

  return result;
}


void
ContourMesh::transform(Transform &t)
{
  vector<Point>::iterator itr = nodes_.begin();
  vector<Point>::iterator eitr = nodes_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
}


bool
ContourMesh::locate(Node::index_type &idx, const Point &p) const
{
  Node::iterator ni = node_begin();
  idx = *node_begin();

  if (ni==node_end())
    return false;

  double closest = (p-nodes_[*ni]).length2();

  ++ni;
  for (; ni != node_end(); ++ni) {
    if ( (p-nodes_[*ni]).length2() < closest ) {
      closest = (p-nodes_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}

bool
ContourMesh::locate(Edge::index_type &idx, const Point &p) const
{
  Edge::iterator ei;
  double cosa, closest=DBL_MAX;
  Node::array_type nra;
  double dist1, dist2, dist3, dist4;
  Point n1,n2,q;

  if (ei==edge_end())
    return false;

  for (ei = edge_begin(); ei != edge_end(); ++ei) {
    get_nodes(nra,*ei);

    n1 = nodes_[nra[0]];
    n2 = nodes_[nra[1]];

    dist1 = (p-n1).length();
    dist2 = (p-n2).length();
    dist3 = (n1-n2).length();

    cosa = Dot(n1-p,n1-n2)/((n1-p).length()*dist3);

    q = n1 + (n1-n2) * (n1-n2)/dist3;

    dist4 = (p-q).length();

    if ( (cosa > 0) && (cosa < dist3) && (dist4 < closest) ) {
      closest = dist4;
      idx = *ei;
    } else if ( (cosa < 0) && (dist1 < closest) ) {
      closest = dist1;
      idx = *ei;
    } else if ( (cosa > dist3) && (dist2 < closest) ) {
      closest = dist2;
      idx = *ei;
    }
  }

  return true;
}


void
ContourMesh::get_weights(const Point &p,
			 Node::array_type &l, vector<double> &w)
{
  Edge::index_type idx;
  if (locate(idx, p))
  {
    Node::array_type ra(2);
    get_nodes(ra,idx);
    Point p0,p1;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    double dist0, dist1, dist_sum;
    dist0 = (p0-p).length();
    dist1 = (p1-p).length();
    dist_sum = dist0 + dist1;
    l.push_back(ra[0]);
    l.push_back(ra[1]);
    w.push_back(dist0/dist_sum);
    w.push_back(dist1/dist_sum);
  }
}

void
ContourMesh::get_weights(const Point &p,
			 Edge::array_type &l, vector<double> &w)
{
  Edge::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}



#define CONTOURMESH_VERSION 1

void
ContourMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(), CONTOURMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream,nodes_);
  Pio(stream,edges_);

  stream.end_class();
}

const string
ContourMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "ContourMesh";
  return name;
}

template<>
ContourMesh::Node::iterator
ContourMesh::tbegin(ContourMesh::Node::iterator *) const
{
  return 0;
}

template<>
ContourMesh::Node::iterator
ContourMesh::tend(ContourMesh::Node::iterator *) const
{
  return (unsigned)nodes_.size();
}

template<>
ContourMesh::Edge::iterator
ContourMesh::tbegin(ContourMesh::Edge::iterator *) const
{
  return 0;
}


template<>
ContourMesh::Edge::iterator
ContourMesh::tend(ContourMesh::Edge::iterator *) const
{
  return (unsigned)edges_.size();
}

template<>
ContourMesh::Face::iterator
ContourMesh::tbegin(ContourMesh::Face::iterator *) const
{
  return 0;
}

template<>
ContourMesh::Face::iterator
ContourMesh::tend(ContourMesh::Face::iterator *) const
{
  return 0;
}

template<>
ContourMesh::Cell::iterator
ContourMesh::tbegin(ContourMesh::Cell::iterator *) const
{
  return 0;
}

template<>
ContourMesh::Cell::iterator
ContourMesh::tend(ContourMesh::Cell::iterator *) const
{
  return 0;
}

template<>
ContourMesh::Node::size_type
ContourMesh::tsize(ContourMesh::Node::size_type *) const
{
  return (unsigned)nodes_.size();
}

template<>
ContourMesh::Edge::size_type
ContourMesh::tsize(ContourMesh::Edge::size_type *) const
{
  return (unsigned)edges_.size();
}

template<>
ContourMesh::Face::size_type
ContourMesh::tsize(ContourMesh::Face::size_type *) const
{
  return 0;
}

template<>
ContourMesh::Cell::size_type
ContourMesh::tsize(ContourMesh::Cell::size_type *) const
{
  return 0;
}


ContourMesh::Node::iterator ContourMesh::node_begin() const
{ return tbegin((Node::iterator *)0); }
ContourMesh::Edge::iterator ContourMesh::edge_begin() const
{ return tbegin((Edge::iterator *)0); }
ContourMesh::Face::iterator ContourMesh::face_begin() const
{ return tbegin((Face::iterator *)0); }
ContourMesh::Cell::iterator ContourMesh::cell_begin() const
{ return tbegin((Cell::iterator *)0); }

ContourMesh::Node::iterator ContourMesh::node_end() const
{ return tend((Node::iterator *)0); }
ContourMesh::Edge::iterator ContourMesh::edge_end() const
{ return tend((Edge::iterator *)0); }
ContourMesh::Face::iterator ContourMesh::face_end() const
{ return tend((Face::iterator *)0); }
ContourMesh::Cell::iterator ContourMesh::cell_end() const
{ return tend((Cell::iterator *)0); }

ContourMesh::Node::size_type ContourMesh::nodes_size() const
{ return tsize((Node::size_type *)0); }
ContourMesh::Edge::size_type ContourMesh::edges_size() const
{ return tsize((Edge::size_type *)0); }
ContourMesh::Face::size_type ContourMesh::faces_size() const
{ return tsize((Face::size_type *)0); }
ContourMesh::Cell::size_type ContourMesh::cells_size() const
{ return tsize((Cell::size_type *)0); }


const TypeDescription*
ContourMesh::get_type_description() const
{
  return SCIRun::get_type_description((ContourMesh *)0);
}

const TypeDescription*
get_type_description(ContourMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ContourMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ContourMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ContourMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ContourMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ContourMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ContourMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ContourMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ContourMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ContourMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
