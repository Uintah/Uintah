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
 *  TriSurfMesh.cc: Tetrahedral mesh with new design.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Persistent/PersistentSTL.h>
#include <sci_hash_map.h>

namespace SCIRun {

PersistentTypeID TriSurfMesh::type_id("TriSurfMesh", "Mesh", NULL);


const string
TriSurfMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "TriSurfMesh";
  return name;
}



TriSurfMesh::TriSurfMesh()
{
}

TriSurfMesh::TriSurfMesh(const TriSurfMesh &copy)
  : points_(copy.points_),
    faces_(copy.faces_),
    neighbors_(copy.neighbors_)
{
}

TriSurfMesh::~TriSurfMesh()
{
}

/* To generate a random point inside of a triangle, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void TriSurfMesh::get_random_point(Point &p, const Face::index_type &ei,
				   int seed) const
{
  static MusilRNG rng;

  // get the positions of the vertices
  Node::array_type ra;
  get_nodes(ra,ei);
  Point p0,p1,p2;
  get_point(p0,ra[0]);
  get_point(p1,ra[1]);
  get_point(p2,ra[2]);

  // generate the barrycentric coordinates
  double u,v;
  if (seed) {
    MusilRNG rng1(seed);
    u = rng1(); 
    v = rng1()*(1.-u);
  } else {
    u = rng(); 
    v = rng()*(1.-u);
  }

  // compute the position of the random point
  p = p0+((p1-p0)*u)+((p2-p0)*v);
}


BBox
TriSurfMesh::get_bounding_box() const
{
  BBox result;

  for (vector<Point>::size_type i = 0; i < points_.size(); i++)
  {
    result.extend(points_[i]);
  }

  return result;
}


void
TriSurfMesh::transform(Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
}


void
TriSurfMesh::begin(TriSurfMesh::Node::iterator &itr) const
{
  itr = 0;
}


void
TriSurfMesh::end(TriSurfMesh::Node::iterator &itr) const
{
  itr = points_.size();
}

void
TriSurfMesh::begin(TriSurfMesh::Edge::iterator &itr) const
{
  itr = 0;
}

void
TriSurfMesh::end(TriSurfMesh::Edge::iterator &itr) const
{
  itr = faces_.size();
}

void
TriSurfMesh::begin(TriSurfMesh::Face::iterator &itr) const
{
  itr = 0;
}

void
TriSurfMesh::end(TriSurfMesh::Face::iterator &itr) const
{
  itr = faces_.size() / 3;
}

void
TriSurfMesh::begin(TriSurfMesh::Cell::iterator &itr) const
{
  itr = 0;
}

void
TriSurfMesh::end(TriSurfMesh::Cell::iterator &itr) const
{
  itr = 0;
}


void
TriSurfMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  static int table[6][2] =
  {
    {0, 1},
    {1, 2},
    {2, 0},
  };

  const int off = idx % 3;
  const int node = idx - off;
  array.clear();
  array.push_back(faces_[node + table[off][0]]);
  array.push_back(faces_[node + table[off][1]]);
}


void
TriSurfMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.clear();
  array.push_back(faces_[idx * 3 + 0]);
  array.push_back(faces_[idx * 3 + 1]);
  array.push_back(faces_[idx * 3 + 2]);
}

void
TriSurfMesh::get_nodes(Node::array_type &array, Cell::index_type cidx) const
{
  array.clear();
  array.push_back(faces_[cidx * 3 + 0]);
  array.push_back(faces_[cidx * 3 + 1]);
  array.push_back(faces_[cidx * 3 + 2]);
}


void
TriSurfMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
{
  array.clear();
  array.push_back(idx * 3 + 0);
  array.push_back(idx * 3 + 1);
  array.push_back(idx * 3 + 2);
}


void
TriSurfMesh::get_neighbor(Face::index_type &neighbor, Edge::index_type idx) const
{
  neighbor = neighbors_[idx];
}


static double
distance2(const Point &p0, const Point &p1)
{
  const double dx = p0.x() - p1.x();
  const double dy = p0.y() - p1.y();
  const double dz = p0.z() - p1.z();
  return dx * dx + dy * dy + dz * dz;
}


bool
TriSurfMesh::locate(Node::index_type &loc, const Point &p) const
{
  Node::iterator ni, nie;
  begin(ni);
  end(nie);

  loc = *ni;
  if (ni == nie)
  {
    return false;
  }

  double min_dist = distance2(p, points_[*ni]);
  loc = *ni;
  ++ni;

  while (ni != nie)
  {
    const double dist = distance2(p, points_[*ni]);
    if (dist < min_dist)
    {
      loc = *ni;
    }
    ++ni;
  }
  return true;
}


bool
TriSurfMesh::locate(Edge::index_type &loc, const Point &) const
{
  loc = 0;
  return false;
}


bool
TriSurfMesh::locate(Face::index_type &loc, const Point &) const
{
  loc = 0;
  return false;
}


bool
TriSurfMesh::locate(Cell::index_type &loc, const Point &) const
{
  loc = 0;
  return false;
}


void
TriSurfMesh::get_weights(const Point &p,
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
TriSurfMesh::get_weights(const Point &p,
			 Node::array_type &l, vector<double> &w)
{
  Face::index_type idx;
  if (locate(idx, p))
  {
    Node::array_type ra(3);
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    double area0, area1, area2, area_sum;
    area0 = (Cross(p1-p,p2-p)).length();
    area1 = (Cross(p0-p,p2-p)).length();
    area2 = (Cross(p0-p,p1-p)).length();
    area_sum = area0+area1+area2;
    l.push_back(ra[0]);
    l.push_back(ra[1]);
    l.push_back(ra[2]);
    w.push_back(area0/area_sum);
    w.push_back(area1/area_sum);
    w.push_back(area2/area_sum);
  }
}


void
TriSurfMesh::get_center(Point &p, Edge::index_type i) const
{
  Node::array_type nodes;
  get_nodes(nodes, i);
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
  p = v.asPoint();
}


void
TriSurfMesh::get_center(Point &p, Face::index_type i) const
{
  Node::array_type nodes;
  get_nodes(nodes, i);
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
  p = v.asPoint();
}



bool
TriSurfMesh::inside4_p(int i, const Point &p)
{
  // TODO: This has not been tested.
  // TODO: Looks like too much code to check sign of 4 plane/point tests.

  const Point &p0 = points_[faces_[i+0]];
  const Point &p1 = points_[faces_[i+1]];
  const Point &p2 = points_[faces_[i+2]];
  const Point &p3 = points_[faces_[i+3]];
  const double x0 = p0.x();
  const double y0 = p0.y();
  const double z0 = p0.z();
  const double x1 = p1.x();
  const double y1 = p1.y();
  const double z1 = p1.z();
  const double x2 = p2.x();
  const double y2 = p2.y();
  const double z2 = p2.z();
  const double x3 = p3.x();
  const double y3 = p3.y();
  const double z3 = p3.z();

  const double a0 = + x1*(y2*z3-y3*z2) + x2*(y3*z1-y1*z3) + x3*(y1*z2-y2*z1);
  const double a1 = - x2*(y3*z0-y0*z3) - x3*(y0*z2-y2*z0) - x0*(y2*z3-y3*z2);
  const double a2 = + x3*(y0*z1-y1*z0) + x0*(y1*z3-y3*z1) + x1*(y3*z0-y0*z3);
  const double a3 = - x0*(y1*z2-y2*z1) - x1*(y2*z0-y0*z2) - x2*(y0*z1-y1*z0);
  const double iV6 = 1.0 / (a0+a1+a2+a3);

  const double b0 = - (y2*z3-y3*z2) - (y3*z1-y1*z3) - (y1*z2-y2*z1);
  const double c0 = + (x2*z3-x3*z2) + (x3*z1-x1*z3) + (x1*z2-x2*z1);
  const double d0 = - (x2*y3-x3*y2) - (x3*y1-x1*y3) - (x1*y2-x2*y1);
  const double s0 = iV6 * (a0 + b0*p.x() + c0*p.y() + d0*p.z());
  if (s0 < -1.e-6)
    return false;

  const double b1 = + (y3*z0-y0*z3) + (y0*z2-y2*z0) + (y2*z3-y3*z2);
  const double c1 = - (x3*z0-x0*z3) - (x0*z2-x2*z0) - (x2*z3-x3*z2);
  const double d1 = + (x3*y0-x0*y3) + (x0*y2-x2*y0) + (x2*y3-x3*y2);
  const double s1 = iV6 * (a1 + b1*p.x() + c1*p.y() + d1*p.z());
  if (s1 < -1.e-6)
    return false;

  const double b2 = - (y0*z1-y1*z0) - (y1*z3-y3*z1) - (y3*z0-y0*z3);
  const double c2 = + (x0*z1-x1*z0) + (x1*z3-x3*z1) + (x3*z0-x0*z3);
  const double d2 = - (x0*y1-x1*y0) - (x1*y3-x3*y1) - (x3*y0-x0*y3);
  const double s2 = iV6 * (a2 + b2*p.x() + c2*p.y() + d2*p.z());
  if (s2 < -1.e-6)
    return false;

  const double b3 = +(y1*z2-y2*z1) + (y2*z0-y0*z2) + (y0*z1-y1*z0);
  const double c3 = -(x1*z2-x2*z1) - (x2*z0-x0*z2) - (x0*z1-x1*z0);
  const double d3 = +(x1*y2-x2*y1) + (x2*y0-x0*y2) + (x0*y1-x1*y0);
  const double s3 = iV6 * (a3 + b3*p.x() + c3*p.y() + d3*p.z());
  if (s3 < -1.e-6)
    return false;

  return true;
}

void
TriSurfMesh::flush_changes()
{
  compute_normals();
}

void
TriSurfMesh::compute_normals()
{
  if (normals_.size() > 0) { return; }
  normals_.resize(points_.size()); // 1 per node

  // build table of faces that touch each node
  vector<vector<Face::index_type> > node_in_faces(points_.size());
  //! face normals (not normalized) so that magnitude is also the area.
  vector<Vector> face_normals(faces_.size());
  // Computing normal per face.
  Node::array_type nodes(3);
  Face::iterator iter, iter_end;
  begin(iter);
  end(iter_end);
  while (iter != iter_end)
  {
    get_nodes(nodes, *iter);

    Point p1, p2, p3;
    get_point(p1, nodes[0]);
    get_point(p2, nodes[1]);
    get_point(p3, nodes[2]);
    // build table of faces that touch each node
    node_in_faces[nodes[0]].push_back(*iter);
    node_in_faces[nodes[1]].push_back(*iter);
    node_in_faces[nodes[2]].push_back(*iter);

    Vector v0 = p2 - p1;
    Vector v1 = p3 - p2;
    Vector n = Cross(v0, v1);
    face_normals[*iter] = n;
//    cerr << "normal mag: " << n.length() << endl;
    ++iter;
  }
  //Averaging the normals.
  vector<vector<Face::index_type> >::iterator nif_iter = node_in_faces.begin();
  int i = 0;
  while (nif_iter != node_in_faces.end()) {
    const vector<Face::index_type> &v = *nif_iter;
    vector<Face::index_type>::const_iterator fiter = v.begin();
    Vector ave(0.L,0.L,0.L);
    Vector sum(0.L,0.L,0.L);
    while(fiter != v.end()) {
      sum += face_normals[*fiter];
      ++fiter;
    }
    fiter = v.begin();
    while(fiter != v.end()) {
      if (Dot(face_normals[*fiter],sum)>0)
	ave += face_normals[*fiter];
      else
	ave -= face_normals[*fiter];
      ++fiter;
    }
    if (ave.length2()) {
      ave.normalize();
      normals_[i] = ave; ++i;
    }
    ++nif_iter;
  }
}

TriSurfMesh::Node::index_type
TriSurfMesh::add_find_point(const Point &p, double err)
{
  Node::index_type i;
  if (locate(i, p) && distance2(points_[i], p) < err)
  {
    return i;
  }
  else
  {
    points_.push_back(p);
    return points_.size() - 1;
  }
}


void
TriSurfMesh::add_triangle(Node::index_type a, Node::index_type b, Node::index_type c)
{
  faces_.push_back(a);
  faces_.push_back(b);
  faces_.push_back(c);
}


TriSurfMesh::Elem::index_type
TriSurfMesh::add_elem(Node::array_type a)
{
  faces_.push_back(a[0]);
  faces_.push_back(a[1]);
  faces_.push_back(a[2]);
  return (faces_.size() - 1) / 3;
}


void
TriSurfMesh::connect(double err)
{
  // Collapse point set by err.
  // TODO: average in stead of first found for new point?
  vector<Point> points(points_);
  vector<int> mapping(points_.size());
  vector<Point>::size_type i;
  points_.clear();
  for (i = 0; i < points.size(); i++)
  {
    mapping[i] = add_find_point(points[i], err);
  }

  // Repair faces.
  for (i=0; i < faces_.size(); i++)
  {
    faces_[i] = mapping[i];
  }

  // TODO: Remove all degenerate faces here.

  // TODO: fix forward/backward facing problems.

  // Find neighbors
  vector<list<int> > edgemap(points_.size());
  for (i=0; i< faces_.size(); i++)
  {
    edgemap[faces_[i]].push_back(i);
  }

  for (i=0; i<edgemap.size(); i++)
  {
    list<int>::iterator li1 = edgemap[i].begin();

    while (li1 != edgemap[i].end())
    {
      int e1 = *li1;
      li1++;

      list<int>::iterator li2 = li1;
      while (li2 != edgemap[i].end())
      {
	int e2 = *li2;
	li2++;
	
	if ( faces_[next(e1)] == faces_[prev(e2)])
	{
	  neighbors_[e1] = e2;
	  neighbors_[e2] = e1;
	}
      }
    }
  }

  // Remove unused points.
  // Reuse mapping array, edgemap array.
  vector<Point> dups(points_);
  points_.clear();

  for (i=0; i<dups.size(); i++)
  {
    if(edgemap[i].begin() != edgemap[i].end())
    {
      points_.push_back(dups[i]);
      mapping[i] = points_.size() - 1;
    }
  }

  // Repair faces.
  for (i=0; i < faces_.size(); i++)
  {
    faces_[i] = mapping[i];
  }
}



TriSurfMesh::Node::index_type
TriSurfMesh::add_point(const Point &p)
{
  points_.push_back(p);
  return points_.size() - 1;
}



void
TriSurfMesh::add_triangle(const Point &p0, const Point &p1, const Point &p2)
{
  add_triangle(add_find_point(p0), add_find_point(p1), add_find_point(p2));
}

void
TriSurfMesh::add_triangle_unconnected(const Point &p0,
				      const Point &p1,
				      const Point &p2)
{
  add_triangle(add_point(p0), add_point(p1), add_point(p2));
}


#define TRISURFMESH_VERSION 1

void
TriSurfMesh::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), TRISURFMESH_VERSION);

  Mesh::io(stream);

  Pio(stream, points_);
  Pio(stream, faces_);
  Pio(stream, neighbors_);

  stream.end_class();

  if (stream.reading())
  {
    flush_changes();
  }
}


void
TriSurfMesh::size(TriSurfMesh::Node::size_type &s) const
{
  Node::iterator itr; end(itr);
  s = *itr;
}

void
TriSurfMesh::size(TriSurfMesh::Edge::size_type &s) const
{
  Edge::iterator itr; end(itr);
  s = *itr;
}

void
TriSurfMesh::size(TriSurfMesh::Face::size_type &s) const
{
  Face::iterator itr; end(itr);
  s = *itr;
}

void
TriSurfMesh::size(TriSurfMesh::Cell::size_type &s) const
{
  Cell::iterator itr; end(itr);
  s = *itr;
}



const TypeDescription*
TriSurfMesh::get_type_description() const
{
  return SCIRun::get_type_description((TriSurfMesh *)0);
}

const TypeDescription*
get_type_description(TriSurfMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TriSurfMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(TriSurfMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TriSurfMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(TriSurfMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TriSurfMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(TriSurfMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TriSurfMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(TriSurfMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("TriSurfMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


} // namespace SCIRun
