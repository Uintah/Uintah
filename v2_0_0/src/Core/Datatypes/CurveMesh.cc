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
 *  CurveMesh.cc: contour mesh
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

#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <float.h>  // for DBL_MAX
#include <iostream>
#include <sci_hash_map.h>

namespace SCIRun {

using namespace std;


PersistentTypeID CurveMesh::type_id("CurveMesh", "Mesh", maker);

BBox
CurveMesh::get_bounding_box() const
{
  BBox result;
  
  Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie)
  {
    result.extend(nodes_[*i]);
    ++i;
  }

  return result;
}


void
CurveMesh::transform(const Transform &t)
{
  vector<Point>::iterator itr = nodes_.begin();
  vector<Point>::iterator eitr = nodes_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
}


void
CurveMesh::get_center(Point &result, Edge::index_type idx) const
{
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);
  
  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


int
CurveMesh::get_valence(Node::index_type idx) const
{
  int count = 0;
  for (unsigned int i = 0; i < edges_.size(); i++)
    if (edges_[i].first == idx || edges_[i].second == idx) count++;
  return count;
}



bool
CurveMesh::locate(Node::index_type &idx, const Point &p) const
{
  Node::iterator ni, nie;
  begin(ni);
  end(nie);

  idx = *ni;

  if (ni == nie)
  {
    return false;
  }

  double closest = (p-nodes_[*ni]).length2();

  ++ni;
  for (; ni != nie; ++ni)
  {
    if ( (p-nodes_[*ni]).length2() < closest )
    {
      closest = (p-nodes_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}

bool
CurveMesh::locate(Edge::index_type &idx, const Point &p) const
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
  {
    return false;
  }
  
  for (; ei != eie; ++ei) {
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
CurveMesh::get_weights(const Point &p,
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
CurveMesh::get_weights(const Point &p,
			 Edge::array_type &l, vector<double> &w)
{
  Edge::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}



#define CURVE_MESH_VERSION 1

void
CurveMesh::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(), CURVE_MESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream,nodes_);
  Pio(stream,edges_);

  stream.end_class();
}

const string
CurveMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "CurveMesh";
  return name;
}

void
CurveMesh::begin(CurveMesh::Node::iterator &itr) const
{
  itr = 0;
}

void
CurveMesh::end(CurveMesh::Node::iterator &itr) const
{
  itr = static_cast<Node::iterator>(nodes_.size());
}

void
CurveMesh::begin(CurveMesh::Edge::iterator &itr) const
{
  itr = 0;
}


void
CurveMesh::end(CurveMesh::Edge::iterator &itr) const
{
  itr = (unsigned)edges_.size();
}

void
CurveMesh::begin(CurveMesh::Face::iterator &itr) const
{
  itr = 0;
}

void
CurveMesh::end(CurveMesh::Face::iterator &itr) const
{
  itr = 0;
}

void
CurveMesh::begin(CurveMesh::Cell::iterator &itr) const
{
  itr = 0;
}

void
CurveMesh::end(CurveMesh::Cell::iterator &itr) const
{
  itr = 0;
}

void
CurveMesh::size(CurveMesh::Node::size_type &s) const
{
  s = (unsigned)nodes_.size();
}

void
CurveMesh::size(CurveMesh::Edge::size_type &s) const
{
  s = (unsigned)edges_.size();
}

void
CurveMesh::size(CurveMesh::Face::size_type &s) const
{
  s = 0;
}

void
CurveMesh::size(CurveMesh::Cell::size_type &s) const
{
  s = 0;
}



const TypeDescription*
CurveMesh::get_type_description() const
{
  return SCIRun::get_type_description((CurveMesh *)0);
}

const TypeDescription*
get_type_description(CurveMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("CurveMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(CurveMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("CurveMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(CurveMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("CurveMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(CurveMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("CurveMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(CurveMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("CurveMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
