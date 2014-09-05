/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <sci_hash_map.h>

namespace SCIRun {

PersistentTypeID QuadSurfMesh::type_id("QuadSurfMesh", "Mesh", maker);


const string
QuadSurfMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "QuadSurfMesh";
  return name;
}



QuadSurfMesh::QuadSurfMesh()
  : points_(0),
    faces_(0),
    edge_neighbors_(0),
    point_lock_("QuadSurfMesh point_lock_"),
    edge_lock_("QuadSurfMesh edge_lock_"),
    face_lock_("QuadSurfMesh face_lock_"),
    edge_neighbor_lock_("QuadSurfMesh edge_neighbor_lock_"),
    normal_lock_("QuadSurfMesh normal_lock_"),
    synchronized_(NODES_E | FACES_E | CELLS_E)
{
}

QuadSurfMesh::QuadSurfMesh(const QuadSurfMesh &copy)
  : points_(copy.points_),
    edges_(copy.edges_),
    faces_(copy.faces_),
    edge_neighbors_(copy.edge_neighbors_),
    normals_( copy.normals_ ),
    point_lock_("QuadSurfMesh point_lock_"),
    edge_lock_("QuadSurfMesh edge_lock_"),
    face_lock_("QuadSurfMesh face_lock_"),
    edge_neighbor_lock_("QuadSurfMesh edge_neighbor_lock_"),
    normal_lock_("QuadSurfMesh normal_lock_"),
    synchronized_(copy.synchronized_)
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
QuadSurfMesh::transform(const Transform &t)
{
  point_lock_.lock();
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  point_lock_.unlock();
}


void
QuadSurfMesh::begin(QuadSurfMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on QuadSurfMesh first");
  itr = 0;
}


void
QuadSurfMesh::end(QuadSurfMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E,
	    "Must call synchronize NODES_E on QuadSurfMesh first");
  itr = (int)points_.size();
}

void
QuadSurfMesh::begin(QuadSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");
  itr = 0;
}

void
QuadSurfMesh::end(QuadSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");
  itr = static_cast<Edge::iterator>((int)edges_.size());
}

void
QuadSurfMesh::begin(QuadSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on QuadSurfMesh first");
  itr = 0;
}

void
QuadSurfMesh::end(QuadSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on QuadSurfMesh first");
  itr = static_cast<Face::iterator>((int)faces_.size() / 4);
}

void
QuadSurfMesh::begin(QuadSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on QuadSurfMesh first");
  itr = 0;
}

void
QuadSurfMesh::end(QuadSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on QuadSurfMesh first");
  itr = 0;
}


void
QuadSurfMesh::get_nodes(Node::array_type &array, Edge::index_type eidx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");
  static int table[8][2] =
  {
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 0},
  };

  const int idx = edges_[eidx];
  const int off = idx % 4;
  const int node = idx - off;
  array.clear();
  array.push_back(faces_[node + table[off][0]]);
  array.push_back(faces_[node + table[off][1]]);
}


void
QuadSurfMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on QuadSurfMesh first");
  array.clear();
  array.push_back(faces_[idx * 4 + 0]);
  array.push_back(faces_[idx * 4 + 1]);
  array.push_back(faces_[idx * 4 + 2]);
  array.push_back(faces_[idx * 4 + 3]);
}


void
QuadSurfMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");

  array.clear();

  unsigned int i;
  for (i=0; i < 4; i++)
  {
    const int a = idx * 4 + i;
    const int b = a - a % 4 + (a+1) % 4;
    int j = (int)edges_.size()-1;
    for (; j >= 0; j--)
    {
      const int c = edges_[j];
      const int d = c - c % 4 + (c+1) % 4;
      if (faces_[a] == faces_[c] && faces_[b] == faces_[d] ||
	  faces_[a] == faces_[d] && faces_[b] == faces_[c])
      {
	array.push_back(j);
	break;
      }
    }
  }
}


bool
QuadSurfMesh::get_neighbor(Face::index_type &neighbor,
			   Face::index_type from,
			   Edge::index_type edge) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
	    "Must call synchronize EDGE_NEIGHBORS_E on QuadSurfMesh first");
  unsigned int n = edge_neighbors_[edges_[edge]];
  if (n != MESH_NO_NEIGHBOR && (n / 4) == from)
  {
    n = edge_neighbors_[n];
  }
  if (n != MESH_NO_NEIGHBOR)
  {
    neighbor = n / 4;
    return true;
  }
  return false;
}

void
QuadSurfMesh::get_neighbors(Face::array_type &neighbor,
			    Face::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
	    "Must call synchronize EDGE_NEIGHBORS_E on QuadSurfMesh first");
  Edge::array_type edges;
  get_edges(edges, idx);

  neighbor.clear();
  Edge::array_type::iterator iter = edges.begin();
  while (iter != edges.end()) {
    Face::index_type f;
    if (get_neighbor(f, idx, *iter)) {
      neighbor.push_back(f);
    }
    ++iter;
  }
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
  Node::iterator bi, ei;
  begin(bi);
  end(ei);
  loc = 0;
  
  bool found = false;
  double mindist = 0.0;
  while (bi != ei) 
  {
    const Point &center = point(*bi);
    const double dist = (p - center).length2();
    if (!found || dist < mindist) 
    {      
      loc = *bi;
      mindist = dist;
      found = true;
    }
    ++bi;
  }
  return found;
}


static double
distance_to_line2(const Point &p, const Point &a, const Point &b)
{
  Vector m = b - a;
  Vector n = p - a;
  if (m.length2() < 1e-6)
  {
    return n.length2();
  }
  else
  {
    const double t0 = Dot(m, n) / Dot(m, m);
    if (t0 <= 0) return (n).length2();
    else if (t0 >= 1.0) return (p - b).length2();
    else return (n - m * t0).length2();
  }
}


bool
QuadSurfMesh::locate(Edge::index_type &loc, const Point &p) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");

  Edge::iterator bi, ei;
  Node::array_type nodes;
  begin(bi);
  end(ei);
  loc = 0;
  
  bool found = false;
  double mindist = 0.0;
  while (bi != ei)
  {
    get_nodes(nodes,*bi);
    const double dist = distance_to_line2(p, points_[nodes[0]], points_[nodes[1]]);
    if (!found || dist < mindist)
    {
      loc = *bi;
      mindist = dist;
      found = true;
    }
    ++bi;
  }
  return found;
}


// TODO: Nearest cell center is incorrect if the quads are not a
// delanay triangulation of the cell centers which is most of the time.
bool
QuadSurfMesh::locate(Face::index_type &loc, const Point &p) const
{  
  Face::iterator bi, ei;
  Point center;
  begin(bi);
  end(ei);
  loc = 0;
  
  bool found = false;
  double mindist = 0.0;
  while (bi != ei) 
  {
    get_center(center, *bi);
    const double dist = (p - center).length2();
    if (!found || dist < mindist) 
    {      
      loc = *bi;
      mindist = dist;
      found = true;
    }
    ++bi;
  }
  return found;
}


bool
QuadSurfMesh::locate(Cell::index_type &loc, const Point &) const
{
  loc = 0;
  return false;
}


int
QuadSurfMesh::get_weights(const Point &p, Face::array_type &l, double *w)
{
  Face::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}


static double
tri_area(const Point &a, const Point &b, const Point &c)
{
  return Cross(b-a, c-a).length();
}


int
QuadSurfMesh::get_weights(const Point &p, Node::array_type &locs, double *w)
{
  Face::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(locs, idx);
    const Point &p0 = point(locs[0]);
    const Point &p1 = point(locs[1]);
    const Point &p2 = point(locs[2]);
    const Point &p3 = point(locs[3]);

    const double a0 = tri_area(p, p0, p1);
    if (a0 < 1.0e-6)
    {
      const Vector v0 = p0 - p1;
      const Vector v1 = p - p1;
      const double l2 = Dot(v0, v0);
      w[0] = (l2 < 1.0e-6) ? 0.5 : Dot(v0, v1) / l2;
      w[1] = 1.0 - w[0];
      w[2] = 0.0;
      w[3] = 0.0;
      return 4;
    }
    const double a1 = tri_area(p, p1, p2);
    if (a1 < 1.0e-6)
    {
      const Vector v0 = p1 - p2;
      const Vector v1 = p - p2;
      const double l2 = Dot(v0, v0);
      w[1] = (l2 < 1.0e-6) ? 0.5 : Dot(v0, v1) / l2;
      w[2] = 1.0 - w[1];
      w[3] = 0.0;
      w[0] = 0.0;
      return 4;
    }
    const double a2 = tri_area(p, p2, p3);
    if (a2 < 1.0e-6)
    {
      const Vector v0 = p2 - p3;
      const Vector v1 = p - p3;
      const double l2 = Dot(v0, v0);
      w[2] = (l2 < 1.0e-6) ? 0.5 : Dot(v0, v1) / l2;
      w[3] = 1.0 - w[2];
      w[0] = 0.0;
      w[1] = 0.0;
      return 4;
    }
    const double a3 = tri_area(p, p3, p0);
    if (a3 < 1.0e-6)
    {
      const Vector v0 = p3 - p0;
      const Vector v1 = p - p0;
      const double l2 = Dot(v0, v0);
      w[3] = (l2 < 1.0e-6) ? 0.5 : Dot(v0, v1) / l2;
      w[0] = 1.0 - w[3];
      w[1] = 0.0;
      w[2] = 0.0;
      return 4;
    }

    w[0] = tri_area(p0, p1, p2) / (a0 * a3);
    w[1] = tri_area(p1, p2, p0) / (a1 * a0);
    w[2] = tri_area(p2, p3, p1) / (a2 * a1);
    w[3] = tri_area(p3, p0, p2) / (a3 * a2);

    const double suminv = 1.0 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= suminv;
    w[1] *= suminv;
    w[2] *= suminv;
    w[3] *= suminv;
    return 4;
  }
  return 0;
}


void
QuadSurfMesh::get_center(Point &result, Edge::index_type idx) const
{
  Node::array_type arr;
  get_nodes(arr, idx);
  result = point(arr[0]);
  result.asVector() += point(arr[1]).asVector();

  result.asVector() *= 0.5;
}


void
QuadSurfMesh::get_center(Point &p, Face::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  Node::array_type::iterator nai = nodes.begin();
  get_center(p, *nai);
  ++nai;
  while (nai != nodes.end())
  {
    const Point &pp = point(*nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 4.0);
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

bool
QuadSurfMesh::synchronize(unsigned int tosync)
{
  if (tosync & EDGES_E && !(synchronized_ & EDGES_E))
    compute_edges();
  if (tosync & NORMALS_E && !(synchronized_ & NORMALS_E))
    compute_normals();
  if (tosync & EDGE_NEIGHBORS_E && !(synchronized_ & EDGE_NEIGHBORS_E)) 
    compute_edge_neighbors();
  return true;
}

void
QuadSurfMesh::compute_normals()
{
  normal_lock_.lock();
  if (synchronized_ & NORMALS_E) {
    normal_lock_.unlock();
    return;
  }
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
    ave.safe_normalize();
    normals_[i] = ave; ++i;
    ++nif_iter;
  }
  synchronized_ |= NORMALS_E;
  normal_lock_.unlock();
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
    point_lock_.lock();
    normal_lock_.lock();
    points_.push_back(p);
    if (synchronized_ & NORMALS_E) normals_.push_back(Vector());
    point_lock_.unlock();
    normal_lock_.unlock();
    return static_cast<Node::index_type>((int)points_.size() - 1);
  }
}


QuadSurfMesh::Elem::index_type
QuadSurfMesh::add_quad(Node::index_type a, Node::index_type b,
		       Node::index_type c, Node::index_type d)
{
  face_lock_.lock();
  faces_.push_back(a);
  faces_.push_back(b);
  faces_.push_back(c);
  faces_.push_back(d);
  face_lock_.unlock();
  synchronized_ &= ~NORMALS_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  return static_cast<Elem::index_type>(((int)faces_.size() - 1) >> 2);
}



QuadSurfMesh::Elem::index_type
QuadSurfMesh::add_elem(Node::array_type a)
{
  face_lock_.lock();
  faces_.push_back(a[0]);
  faces_.push_back(a[1]);
  faces_.push_back(a[2]);
  faces_.push_back(a[3]);
  face_lock_.unlock();
  synchronized_ &= ~NORMALS_E;
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  return static_cast<Elem::index_type>(((int)faces_.size() - 1) >> 2);
}



#ifdef HAVE_HASH_MAP

struct edgehash
{
  size_t operator()(const pair<int, int> &a) const
  {
    hash<int> hasher;
    return hasher((int)hasher(a.first) + a.second);
  }
#ifdef __ECC

  static const size_t bucket_size = 4;
  static const size_t min_buckets = 8;

  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first < b.first || a.first == b.first && a.second < b.second;
  }
#endif
};

struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first == b.first && a.second == b.second;
  }
};

#else

struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first < b.first || a.first == b.first && a.second < b.second;
  }
};

#endif

#ifdef HAVE_HASH_MAP

#ifdef __ECC
typedef hash_map<pair<int, int>, int, edgehash> EdgeMapType;
#else
typedef hash_map<pair<int, int>, int, edgehash, edgecompare> EdgeMapType;
#endif

#else

typedef map<pair<int, int>, int, edgecompare> EdgeMapType;

#endif

void
QuadSurfMesh::compute_edges()
{
  edge_lock_.lock();
  if (synchronized_ & EDGES_E) {
    edge_lock_.unlock();
    return;
  }

  EdgeMapType edge_map;
  
  for( int i=(int)faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 4 + (a+1) % 4;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    pair<int, int> nodes(n0, n1);
    edge_map[nodes] = i;
  }

  EdgeMapType::iterator itr;

  for (itr = edge_map.begin(); itr != edge_map.end(); ++itr)
  {
    edges_.push_back((*itr).second);
  }

  synchronized_ |= EDGES_E;
  edge_lock_.unlock();
}


void
QuadSurfMesh::compute_edge_neighbors()
{
  // TODO: This is probably broken with the new indexed edges.
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  edge_neighbor_lock_.lock();
  if (synchronized_ & EDGE_NEIGHBORS_E) {
    edge_neighbor_lock_.unlock();
    return;
  }

  EdgeMapType edge_map;
  
  edge_neighbors_.resize(faces_.size());
  for (unsigned int j = 0; j < edge_neighbors_.size(); j++)
  {
    edge_neighbors_[j] = MESH_NO_NEIGHBOR;
  }

  for(int i = (int)faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 4 + (a+1) % 4;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    pair<int, int> nodes(n0, n1);
   
    EdgeMapType::iterator maploc;

    maploc = edge_map.find(nodes);
    if (maploc != edge_map.end())
    {
      edge_neighbors_[(*maploc).second] = i;
      edge_neighbors_[i] = (*maploc).second;
    }
    edge_map[nodes] = i;
  }

  synchronized_ |= EDGE_NEIGHBORS_E;
  edge_neighbor_lock_.unlock();

}



QuadSurfMesh::Node::index_type
QuadSurfMesh::add_point(const Point &p)
{
  point_lock_.lock();
  normal_lock_.lock();
  points_.push_back(p);
  if (synchronized_ & NORMALS_E) normals_.push_back(Vector());
  point_lock_.unlock();
  normal_lock_.unlock();
  return static_cast<Node::index_type>((int)points_.size() - 1);
}



QuadSurfMesh::Elem::index_type
QuadSurfMesh::add_quad(const Point &p0, const Point &p1,
		       const Point &p2, const Point &p3)
{
  return add_quad(add_find_point(p0), add_find_point(p1),
		  add_find_point(p2), add_find_point(p3));
}


#define QUADSURFMESH_VERSION 2

void
QuadSurfMesh::io(Piostream &stream)
{
  const int version = stream.begin_class(type_name(-1), QUADSURFMESH_VERSION);

  Mesh::io(stream);

  Pio(stream, points_);
  Pio(stream, faces_);
  if (version == 1)
  {
    Pio(stream, edge_neighbors_);
  }

  stream.end_class();

  if (stream.reading())
  {
    synchronized_ = NODES_E | FACES_E | CELLS_E;
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
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on QuadSurfMesh first");
  s = edges_.size();
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
