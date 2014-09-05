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
 *  StructQuadSurfMesh.cc: Templated Mesh defined on a 2D Regular Grid
 *
 *  Written by:
 *   Allen R. Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   November 2002
 *
 *  Copyright (C) 2002 SCI Group
 *
 */

/*
  See StructQuadSurfMesh.h for field/mesh comments.
*/

#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MusilRNG.h>
#include <iostream>

namespace SCIRun {

PersistentTypeID StructQuadSurfMesh::type_id("StructQuadSurfMesh", "Mesh", maker);

StructQuadSurfMesh::StructQuadSurfMesh()
  : synchronized_(ALL_ELEMENTS_E)
{
}
  
StructQuadSurfMesh::StructQuadSurfMesh(unsigned int x, unsigned int y)
  : ImageMesh(x, y, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
    points_(x, y),
    normals_(x,y),
    synchronized_(ALL_ELEMENTS_E)
{
}

StructQuadSurfMesh::StructQuadSurfMesh(const StructQuadSurfMesh &copy)
  : ImageMesh(copy),
    synchronized_(copy.synchronized_)
{
  points_.copy( copy.points_ );
  normals_.copy( copy.normals_ );
}

bool
StructQuadSurfMesh::get_dim(vector<unsigned int> &array) const
{
  array.resize(2);
  array.clear();

  array.push_back(ni_);
  array.push_back(nj_);

  return true;
}

BBox
StructQuadSurfMesh::get_bounding_box() const
{
  BBox result;

  Node::iterator ni, nie;
  begin(ni);
  end(nie);
  while (ni != nie)
  {
    Point p;
    get_center(p, *ni);
    result.extend(p);
    ++ni;
  }
  return result;
}

void
StructQuadSurfMesh::transform(Transform &t)
{
  Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie) {
    points_((*i).i_,(*i).j_) = t.project(points_((*i).i_,(*i).j_));

    ++i;
  }
}

void
StructQuadSurfMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.resize(2);

  const int yidx = idx - (ni_-1) * nj_;
  if (yidx >= 0)
  {
    const int i = yidx / (nj_ - 1);
    const int j = yidx % (nj_ - 1);
    array[0] = Node::index_type(this, i, j);
    array[1] = Node::index_type(this, i, j+1);
  }
  else
  {
    const int i = idx % (ni_ - 1);
    const int j = idx / (ni_ - 1);
    array[0] = Node::index_type(this, i, j);
    array[1] = Node::index_type(this, i+1, j);
  }
}

void
StructQuadSurfMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  const int arr_size = 4;
  array.resize(arr_size);

  for (int i = 0; i < arr_size; i++)
    array[i].mesh_ = idx.mesh_;

  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1;
}

void
StructQuadSurfMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
{
  array.clear();
  array.push_back(idx * 4 + 0);
  array.push_back(idx * 4 + 1);
  array.push_back(idx * 4 + 2);
  array.push_back(idx * 4 + 3);
}

void
StructQuadSurfMesh::get_normal(Vector &result,
			       const Node::index_type &idx ) const
{
  result = normals_(idx.i_, idx.j_);
}

void
StructQuadSurfMesh::get_center(Point &result,
			       const Node::index_type &idx) const
{
  result = points_(idx.i_, idx.j_);
}

void
StructQuadSurfMesh::get_center(Point &result, Edge::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  Node::array_type::iterator nai = nodes.begin();
  Vector v(0.0, 0.0, 0.0);
  while (nai != nodes.end()) {
    Point pp;
    get_point(pp, *nai);
    v += pp.asVector();
    ++nai;
  }
  
  v *= 1.0 / static_cast<double>(nodes.size());
  result = v.asPoint();
}


void
StructQuadSurfMesh::get_center(Point &result,
			       const Face::index_type &idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  Node::array_type::iterator nai = nodes.begin();
  Vector v(0.0, 0.0, 0.0);
  while (nai != nodes.end()) {
    Point pp;
    get_point(pp, *nai);
    v += pp.asVector();
    ++nai;
  }
  v *= 1.0 / static_cast<double>(nodes.size());
  result = v.asPoint();
}

bool
StructQuadSurfMesh::locate(Node::index_type &node, const Point &p) const
{
  node.mesh_ = this;
  Face::index_type fi;
  if (locate(fi, p)) { // first try the fast way.
    Node::array_type nodes;
    get_nodes(nodes, fi);

    double dmin = (p-points_(nodes[0].i_, nodes[0].j_)).length2();
    node = nodes[0];
    for (unsigned int i = 1; i < nodes.size(); i++)
    {
      const double d = (p-points_(nodes[i].i_, nodes[i].j_)).length2();
      if (d < dmin)
      {
	dmin = d;
	node = nodes[i];
      }
    }
    return true;
  }
  else
  {  // do exhaustive search.
    Node::iterator ni, nie;
    begin(ni);
    end(nie);
    if (ni == nie) { return false; }

    double min_dist = (p-points_((*ni).i_, (*ni).j_)).length2();
    node = *ni;
    ++ni;

    while (ni != nie)
    {
      const double dist = (p-points_((*ni).i_, (*ni).j_)).length2();
      if (dist < min_dist)
      {
	node = *ni;
	min_dist = dist;
      }
      ++ni;
    }
    return true;
  }
}

bool
StructQuadSurfMesh::locate(Edge::index_type &node, const Point &) const
{
  ASSERTFAIL("Locate Edge not implemented in StructQuadSurfMesh");
  return false;
}


bool
StructQuadSurfMesh::locate(Face::index_type &loc, const Point &) const
{
  ASSERTFAIL("Locate Face not implemented in StructQuadSurfMesh");
  return false;
}


bool
StructQuadSurfMesh::locate(Cell::index_type &loc, const Point &) const
{
  ASSERTFAIL("Locate Cell not implemented in StructQuadSurfMesh");
  return false;
}

void
StructQuadSurfMesh::get_weights(const Point &p,
				Node::array_type &locs, vector<double> &weights)
{
  ASSERTFAIL("get_weights for nodes not implemented in StructQuadSurfMesh");
}

void
StructQuadSurfMesh::get_weights(const Point &p,
				Face::array_type &l, vector<double> &w)
{
  Face::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}

void
StructQuadSurfMesh::set_point(const Point &point, const Node::index_type &index)
{
  points_(index.i_, index.j_) = point;
}

bool
StructQuadSurfMesh::synchronize(unsigned int tosync)
{
  if (tosync & NORMALS_E && !(synchronized_ & NORMALS_E))
    compute_normals();
  return true;
}

void
StructQuadSurfMesh::compute_normals()
{
  normals_.resize(points_.dim1(), points_.dim2()); // 1 per node

  // build table of faces that touch each node
  Array2< vector<Face::index_type> >
    node_in_faces(points_.dim1(), points_.dim2());

  //! face normals (not normalized) so that magnitude is also the area.
  Array2<Vector> face_normals((points_.dim1()-1),(points_.dim2()-1));

  // Computing normal per face.
  Node::array_type nodes(4);
  Face::iterator iter, iter_end;
  begin(iter);
  end(iter_end);
  while (iter != iter_end)
  {
    get_nodes(nodes, *iter);

    Point p0, p1, p2, p3;
    get_point(p0, nodes[0]);
    get_point(p1, nodes[1]);
    get_point(p2, nodes[2]);
    get_point(p3, nodes[3]);

    // build table of faces that touch each node
    node_in_faces(nodes[0].i_,nodes[0].j_).push_back(*iter);
    node_in_faces(nodes[1].i_,nodes[1].j_).push_back(*iter);
    node_in_faces(nodes[2].i_,nodes[2].j_).push_back(*iter);
    node_in_faces(nodes[3].i_,nodes[3].j_).push_back(*iter);

    Vector v0 = p1 - p0;
    Vector v1 = p2 - p1;
    Vector n = Cross(v0, v1);
    face_normals((*iter).i_, (*iter).j_) = n;

    ++iter;
  }

  //Averaging the normals.
  Node::iterator nif_iter, nif_iter_end;
  begin( nif_iter );
  end( nif_iter_end );

  while (nif_iter != nif_iter_end) {
    vector<Face::index_type> v = node_in_faces((*nif_iter).i_, (*nif_iter).j_);
    vector<Face::index_type>::const_iterator fiter = v.begin();
    Vector ave(0.L,0.L,0.L);
    while(fiter != v.end()) {
      ave += face_normals((*fiter).i_,(*fiter).j_);
      ++fiter;
    }
    ave.normalize();
    normals_((*nif_iter).i_, (*nif_iter).j_) = ave;
    ++nif_iter;
  }
  synchronized_ |= NORMALS_E;
}

#define STRUCT_QUAD_SURF_MESH_VERSION 1

void
StructQuadSurfMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), STRUCT_QUAD_SURF_MESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, ni_);
  Pio(stream, nj_);

  stream.end_class();
}

const string
StructQuadSurfMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "StructQuadSurfMesh";
  return name;
}

const TypeDescription*
StructQuadSurfMesh::get_type_description() const
{
  return SCIRun::get_type_description((StructQuadSurfMesh *)0);
}

const TypeDescription*
get_type_description(StructQuadSurfMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("StructQuadSurfMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
