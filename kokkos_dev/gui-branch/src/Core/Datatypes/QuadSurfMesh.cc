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
 *  QuadSurfMesh.cc: Tetrahedral mesh with new design.
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

#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Persistent/PersistentSTL.h>
#include <sci_hash_map.h>

namespace SCIRun {

PersistentTypeID QuadSurfMesh::type_id("QuadSurfMesh", "Mesh", NULL);


const string
QuadSurfMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "QuadSurfMesh";
  return name;
}



QuadSurfMesh::QuadSurfMesh()
{
}

QuadSurfMesh::QuadSurfMesh(const QuadSurfMesh &copy)
  : points_(copy.points_),
    faces_(copy.faces_),
    neighbors_(copy.neighbors_)
{
}

QuadSurfMesh::~QuadSurfMesh()
{
}

/* To generate a random point inside of a triangle, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
#if 0
void
QuadSurfMesh::get_random_point(Point &p, const Face::index_type &ei,
			       int seed) const
{
  static MusilRNG rng;

  // generate the barrycentric coordinates
  double t,u,v;
  if (seed) {
    MusilRNG rng1(seed);
    t = rng1(); 
    u = rng1(); 
    v = rng1();
    w = rng1();
  } else {
    t = rng(); 
    u = rng(); 
    v = rng();
    w = rng();
  }
  double sum = t+u+v;
  t/=sum;
  u/=sum;
  v/=sum;

  Node::array_type ra;
  get_nodes(ra,ei);
  Point p0,p1,p2;
  if (w < ratio)  // Relative area
  {
    // get the positions of the vertices
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
  }
  else
  {
    get_point(p0,ra[1]);
    get_point(p1,ra[2]);
    get_point(p2,ra[3]);
  }

  // compute the position of the random point
  p = (p0.vector()*t+p1.vector()*u+p2.vector()*v).point();
}
#endif


BBox
QuadSurfMesh::get_bounding_box() const
{
  BBox result;

  for (vector<Point>::size_type i = 0; i < points_.size(); i++)
  {
    result.extend(points_[i]);
  }

  return result;
}


void
QuadSurfMesh::transform(Transform &t)
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
QuadSurfMesh::begin(QuadSurfMesh::Node::iterator &itr) const
{
  itr = 0;
}


void
QuadSurfMesh::end(QuadSurfMesh::Node::iterator &itr) const
{
  itr = points_.size();
}

void
QuadSurfMesh::begin(QuadSurfMesh::Edge::iterator &itr) const
{
  itr = 0;
}

void
QuadSurfMesh::end(QuadSurfMesh::Edge::iterator &itr) const
{
  itr = faces_.size();
}

void
QuadSurfMesh::begin(QuadSurfMesh::Face::iterator &itr) const
{
  itr = 0;
}

void
QuadSurfMesh::end(QuadSurfMesh::Face::iterator &itr) const
{
  itr = faces_.size() / 4;
}

void
QuadSurfMesh::begin(QuadSurfMesh::Cell::iterator &itr) const
{
  itr = 0;
}

void
QuadSurfMesh::end(QuadSurfMesh::Cell::iterator &itr) const
{
  itr = 0;
}


void
QuadSurfMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  static int table[8][2] =
  {
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 0},
  };

  const int off = idx % 4;
  const int node = idx - off;
  array.clear();
  array.push_back(faces_[node + table[off][0]]);
  array.push_back(faces_[node + table[off][1]]);
}


void
QuadSurfMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.clear();
  array.push_back(faces_[idx * 4 + 0]);
  array.push_back(faces_[idx * 4 + 1]);
  array.push_back(faces_[idx * 4 + 2]);
  array.push_back(faces_[idx * 4 + 3]);
}


void
QuadSurfMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
{
  array.clear();
  array.push_back(idx * 4 + 0);
  array.push_back(idx * 4 + 1);
  array.push_back(idx * 4 + 2);
  array.push_back(idx * 4 + 3);
}


void
QuadSurfMesh::get_neighbor(Face::index_type &neighbor, Edge::index_type idx) const
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
QuadSurfMesh::locate(Node::index_type &loc, const Point &p) const
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
QuadSurfMesh::locate(Edge::index_type &loc, const Point &) const
{
  loc = 0;
  return false;
}


bool
QuadSurfMesh::locate(Face::index_type &loc, const Point &) const
{
  loc = 0;
  return false;
}


bool
QuadSurfMesh::locate(Cell::index_type &loc, const Point &) const
{
  loc = 0;
  return false;
}


void
QuadSurfMesh::get_weights(const Point &p,
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
QuadSurfMesh::get_weights(const Point &p,
			 Node::array_type &l, vector<double> &w)
{
#if 0
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
#endif
}



void
QuadSurfMesh::get_center(Point &p, Edge::index_type i) const
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
QuadSurfMesh::get_center(Point &p, Face::index_type i) const
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


#if 0
bool
QuadSurfMesh::inside4_p(int i, const Point &p)
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
#endif

void
QuadSurfMesh::flush_changes()
{
  compute_normals();
}

void
QuadSurfMesh::compute_normals()
{
  if (normals_.size() > 0) { return; }
  normals_.resize(points_.size()); // 1 per node

  // build table of faces that touch each node
  vector<vector<Face::index_type> > node_in_faces(points_.size());
  //! face normals (not normalized) so that magnitude is also the area.
  vector<Vector> face_normals(faces_.size());
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
    node_in_faces[nodes[0]].push_back(*iter);
    node_in_faces[nodes[1]].push_back(*iter);
    node_in_faces[nodes[2]].push_back(*iter);
    node_in_faces[nodes[3]].push_back(*iter);

    Vector v0 = p1 - p0;
    Vector v1 = p2 - p1;
    Vector n = Cross(v0, v1);
    face_normals[*iter] = n;

    ++iter;
  }
  //Averaging the normals.
  vector<vector<Face::index_type> >::iterator nif_iter = node_in_faces.begin();
  int i = 0;
  while (nif_iter != node_in_faces.end()) {
    const vector<Face::index_type> &v = *nif_iter;
    vector<Face::index_type>::const_iterator fiter = v.begin();
    Vector ave(0.L,0.L,0.L);
    while(fiter != v.end()) {
      ave += face_normals[*fiter];
      ++fiter;
    }
    ave.normalize();
    normals_[i] = ave; ++i;
    ++nif_iter;
  }
}


QuadSurfMesh::Node::index_type
QuadSurfMesh::add_find_point(const Point &p, double err)
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
QuadSurfMesh::add_quad(Node::index_type a, Node::index_type b,
		       Node::index_type c, Node::index_type d)
{
  faces_.push_back(a);
  faces_.push_back(b);
  faces_.push_back(c);
  faces_.push_back(d);
}



QuadSurfMesh::Elem::index_type
QuadSurfMesh::add_elem(Node::array_type a)
{
  faces_.push_back(a[0]);
  faces_.push_back(a[1]);
  faces_.push_back(a[2]);
  faces_.push_back(a[3]);
  return (faces_.size() - 1) >> 2;
}



void
QuadSurfMesh::connect(double err)
{
#if 0 
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
#endif
}



QuadSurfMesh::Node::index_type
QuadSurfMesh::add_point(const Point &p)
{
  points_.push_back(p);
  return points_.size() - 1;
}



void
QuadSurfMesh::add_quad(const Point &p0, const Point &p1,
		       const Point &p2, const Point &p3)
{
  add_quad(add_find_point(p0), add_find_point(p1),
	   add_find_point(p2), add_find_point(p3));
}

void
QuadSurfMesh::add_quad_unconnected(const Point &p0, const Point &p1,
				   const Point &p2, const Point &p3)
{
  add_quad(add_point(p0), add_point(p1), add_point(p2), add_point(p3));
}


#define QUADSURFMESH_VERSION 1

void
QuadSurfMesh::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), QUADSURFMESH_VERSION);

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
QuadSurfMesh::size(QuadSurfMesh::Node::size_type &s) const
{
  Node::iterator itr; end(itr);
  s = *itr;
}

void
QuadSurfMesh::size(QuadSurfMesh::Edge::size_type &s) const
{
  Edge::iterator itr; end(itr);
  s = *itr;
}

void
QuadSurfMesh::size(QuadSurfMesh::Face::size_type &s) const
{
  Face::iterator itr; end(itr);
  s = *itr;
}

void
QuadSurfMesh::size(QuadSurfMesh::Cell::size_type &s) const
{
  Cell::iterator itr; end(itr);
  s = *itr;
}



const TypeDescription*
QuadSurfMesh::get_type_description() const
{
  return SCIRun::get_type_description((QuadSurfMesh *)0);
}

const TypeDescription*
get_type_description(QuadSurfMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("QuadSurfMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(QuadSurfMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("QuadSurfMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(QuadSurfMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("QuadSurfMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(QuadSurfMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("QuadSurfMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(QuadSurfMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("QuadSurfMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


} // namespace SCIRun
