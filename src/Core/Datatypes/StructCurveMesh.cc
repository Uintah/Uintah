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
 *  StructCurveMesh.cc: Templated Mesh defined on a 1D Structured Grid
 *
 *  Written by:
 *   Allen R. Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 2002 SCI Group
 *
 */

/*
  See StructCurveMesh.h for field/mesh comments.
*/

#include <Core/Datatypes/StructCurveMesh.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MusilRNG.h>
#include <iostream>
#include <float.h>  // for DBL_MAX

namespace SCIRun {

using namespace std;


PersistentTypeID StructCurveMesh::type_id("StructCurveMesh", "Mesh", maker);


StructCurveMesh::StructCurveMesh(unsigned int n)
  : ScanlineMesh(n, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
    points_(n)
{
}

StructCurveMesh::StructCurveMesh(const StructCurveMesh &copy)
  : ScanlineMesh(copy)
    //    points_(copy.points_)
{
}

BBox
StructCurveMesh::get_bounding_box() const
{
  BBox result;
  
  Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie) {
    result.extend(points_[*i]);
    ++i;
  }

  return result;
}

void
StructCurveMesh::transform(const Transform &t)
{
  Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie) {
    points_[*i] = t.project(points_[*i]);

    ++i;
  }
}

double
StructCurveMesh::get_cord_length() const
{
  double result = 0.0;
  
  Node::iterator i, i1, ie;
  begin(i);
  begin(i1);
  end(ie);

  while (i1 != ie)
  {
    ++i1;
    result += (points_[*i] - points_[*i1]).length();
    ++i;
  }

  return result;
}

void
StructCurveMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.resize(2);
  array[0] = Node::index_type(idx);
  array[1] = Node::index_type(idx + 1);
}

void
StructCurveMesh::get_center(Point &result, const Node::index_type &idx) const
{
  result = points_[idx];
}

void
StructCurveMesh::get_center(Point &result, const Edge::index_type &idx) const
{
  Point p0 = points_[Node::index_type(idx)];
  Point p1 = points_[Node::index_type(idx+1)];

  result = Point(p0+p1)/2.0;
}

bool
StructCurveMesh::locate(Node::index_type &idx, const Point &p) const
{
  Node::iterator ni, nie;
  begin(ni);
  end(nie);

  idx = *ni;

  if (ni == nie)
  {
    return false;
  }

  double closest = (p-points_[*ni]).length2();

  ++ni;
  for (; ni != nie; ++ni)
  {
    if ( (p-points_[*ni]).length2() < closest )
    {
      closest = (p-points_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}

bool
StructCurveMesh::locate(Edge::index_type &idx, const Point &p) const
{
  Edge::iterator ei;
  Edge::iterator eie;
  double cosa, closest=DBL_MAX;
  Node::array_type nra;
  double dist1, dist2, dist3, dist4;
  Point n1,n2,q;

  begin(ei);
  end(eie);

  if (ei==eie)
    return false;
  
  for (; ei != eie; ++ei) {
    get_nodes(nra,*ei);

    n1 = points_[nra[0]];
    n2 = points_[nra[1]];

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
StructCurveMesh::get_weights(const Point &p,
			     Node::array_type &l, vector<double> &w)
{
  Edge::index_type idx;
  if (locate(idx, p)) {
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
StructCurveMesh::get_weights(const Point &p,
			     Edge::array_type &l, vector<double> &w)
{
  Edge::index_type idx;
  if (locate(idx, p)) {
    l.push_back(idx);
    w.push_back(1.0);
  }
}

#define STRUCT_CURVE_MESH_VERSION 1

void
StructCurveMesh::io(Piostream& stream)
{
  //  int version = stream.begin_class(type_name(-1), STRUCT_CURVE_MESH_VERSION);

  Mesh::io(stream);

 // IO data members, in order
  Pio(stream,points_);

  stream.end_class();
}

const string
StructCurveMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "StructCurveMesh";
  return name;
}

const TypeDescription*
StructCurveMesh::get_type_description() const
{
  return SCIRun::get_type_description((StructCurveMesh *)0);
}

const TypeDescription*
get_type_description(StructCurveMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("StructCurveMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
