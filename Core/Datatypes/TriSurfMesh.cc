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
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
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
  : points_(0),
    faces_(0),
    edge_neighbors_(0),
    node_neighbors_(0),
    synchronized_(NODES_E | FACES_E | CELLS_E)
{
}

TriSurfMesh::TriSurfMesh(const TriSurfMesh &copy)
  : points_(copy.points_),
    faces_(copy.faces_),
    edge_neighbors_(copy.edge_neighbors_),
    node_neighbors_(copy.node_neighbors_),
    synchronized_(copy.synchronized_)
{
}

TriSurfMesh::~TriSurfMesh()
{
}

/* To generate a random point inside of a triangle, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void
TriSurfMesh::get_random_point(Point &p, const Face::index_type &ei,
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
  ASSERTMSG(synchronized_ & NODES_E, 
	    "Must call synchronize NODES_E on TriSurfMesh first");
  itr = 0;
}


void
TriSurfMesh::end(TriSurfMesh::Node::iterator &itr) const
{
  ASSERTMSG(synchronized_ & NODES_E, 
	    "Must call synchronize NODES_E on TriSurfMesh first");
  itr = points_.size();
}

void
TriSurfMesh::begin(TriSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  itr = 0;
}

void
TriSurfMesh::end(TriSurfMesh::Edge::iterator &itr) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");
  itr = static_cast<Edge::iterator>(edges_.size());
}

void
TriSurfMesh::begin(TriSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on TriSurfMesh first");
  itr = 0;
}

void
TriSurfMesh::end(TriSurfMesh::Face::iterator &itr) const
{
  ASSERTMSG(synchronized_ & FACES_E,
	    "Must call synchronize FACES_E on TriSurfMesh first");
  itr = static_cast<Face::iterator>(faces_.size() / 3);
}

void
TriSurfMesh::begin(TriSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on TriSurfMesh first");
  itr = 0;
}

void
TriSurfMesh::end(TriSurfMesh::Cell::iterator &itr) const
{
  ASSERTMSG(synchronized_ & CELLS_E,
	    "Must call synchronize CELLS_E on TriSurfMesh first");
  itr = 0;
}


void
TriSurfMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  int a = edges_[idx];
  int b = a - a % 3 + (a+1) % 3;
  array.clear();
  array.push_back(faces_[a]);
  array.push_back(faces_[b]);
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
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");

  array.clear();

  unsigned int i;
  for (i=0; i < 3; i++)
  {
    const int a = idx * 3 + i;
    const int b = a - a % 3 + (a+1) % 3;
    int j = edges_.size()-1;
    for (; j >= 0; j--)
    {
      const int c = edges_[j];
      const int d = c - c % 3 + (c+1) % 3;
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
TriSurfMesh::get_neighbor(Face::index_type &neighbor,
			  Face::index_type face,
			  Edge::index_type edge) const
{
  ASSERTMSG(synchronized_ & EDGE_NEIGHBORS_E,
	    "Must call synchronize EDGE_NEIGHBORS_E on TriSurfMesh first");
  int n = edge_neighbors_[edges_[edge]];
  if (n != -1 && (n % 3) == face)
  {
    n = edge_neighbors_[n];
  }
  if (n != -1)
  {
    neighbor = n / 3;
    return true;
  }
  return false;
}


void
TriSurfMesh::compute_node_neighbors()
{
  node_neighbors_.clear();
  node_neighbors_.resize(points_.size(),set<int>());
  unsigned int nfaces = faces_.size();
  for (unsigned int f = 0; f < nfaces; ++f)
  {
    node_neighbors_[faces_[f]].insert(faces_[next(f)]);
    node_neighbors_[faces_[f]].insert(faces_[prev(f)]);
  }
  synchronized_ |= NODE_NEIGHBORS_E;
}
      

//! Returns all nodes that share an edge with this node 
void
TriSurfMesh::get_neighbors(Node::array_type &array, Node::index_type idx) const
{
  ASSERTMSG(synchronized_ & NODE_NEIGHBORS_E, 
	    "Must call synchronize NODE_NEIGHBORS_E on TriSurfMesh first"); 
  array.resize(node_neighbors_[idx].size());
  copy(node_neighbors_[idx].begin(), node_neighbors_[idx].end(), array.begin());
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
TriSurfMesh::locate(Edge::index_type &loc, const Point &p) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");

  Edge::iterator bi, ei;
  
  bool found = false;
  double mindist = 0.0;
  while (bi != ei)
  {
    int a = *bi;
    int b = a - a % 3 + (a+1) % 3;

    const Point &p0 = points_[faces_[a]];
    const Point &p1 = points_[faces_[b]];

    const double dist = distance_to_line2(p, p0, p1);
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
TriSurfMesh::locate(Face::index_type &loc, const Point &p) const
{
  Face::iterator fi, fie;
  begin(fi);
  end(fie);

  loc = *fi;
  if (fi == fie)
  {
    return false;
  }

  while (fi != fie) {
    if (inside3_p((*fi)*3, p)) {
      loc = *fi;
      return true;
    }
    ++fi;
  }
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

bool 
TriSurfMesh::inside3_p(int i, const Point &p) const
{
  const Point &p0 = points_[faces_[i+0]];
  const Point &p1 = points_[faces_[i+1]];
  const Point &p2 = points_[faces_[i+2]];
  Vector v01(p0-p1);
  Vector v02(p0-p2);
  Vector v0(p0-p);
  Vector v1(p1-p);
  Vector v2(p2-p);
  double a(Cross(v01, v02).length()); // area of the whole triangle (2x)
  double a0(Cross(v1, v2).length());  // area opposite p0
  double a1(Cross(v2, v0).length());  // area opposite p1
  double a2(Cross(v0, v1).length());  // area opposite p2
  double s=a0+a1+a2;
  double r = a/s;
  if (r < (1-1.e-6)) return false;
  return true;
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
TriSurfMesh::get_center(Point &result, Edge::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");

  Node::array_type arr;
  get_nodes(arr, idx);
  Point p0, p1;
  get_center(p0, arr[0]);
  get_center(p1, arr[1]);

  result = ((p0.asVector() + p1.asVector()) * 0.5).asPoint();
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
  v *= 1.0 / static_cast<double>(nodes.size());
  p = v.asPoint();
}


bool
TriSurfMesh::synchronize(unsigned int tosync)
{
  if (tosync & EDGES_E && !(synchronized_ & EDGES_E))
    compute_edges();
  if (tosync & NORMALS_E && !(synchronized_ & NORMALS_E)) 
    compute_normals();
  if (tosync & NODE_NEIGHBORS_E && !(synchronized_ & NODE_NEIGHBORS_E))
    compute_node_neighbors();
  if (tosync & EDGE_NEIGHBORS_E && !(synchronized_ & EDGE_NEIGHBORS_E))
    compute_edge_neighbors();
  return true;
}


void
TriSurfMesh::compute_normals()
{
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
  synchronized_ |= NORMALS_E;
}


struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first == b.first && a.second == b.second;
  }
};


struct edgehash
{
  size_t operator()(const pair<int, int> &a) const
  {
    hash<int> hasher;
    return hasher(hasher(a.first) + a.second);
  }
};

void
TriSurfMesh::compute_edges()
{
  hash_map<pair<int, int>, int, edgehash, edgecompare> edge_map;
  
  int i;
  for (i=faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 3 + (a+1) % 3;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    pair<int, int> nodes(n0, n1);
    edge_map[nodes] = i;
  }

  hash_map<pair<int, int>, int, edgehash, edgecompare>::iterator itr;

  for (itr = edge_map.begin(); itr != edge_map.end(); ++itr)
  {
    edges_.push_back((*itr).second);
  }

  synchronized_ |= EDGES_E;
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
    node_neighbors_.push_back(set<int>());
    return static_cast<Node::index_type>(points_.size() - 1);
  }
}


void
TriSurfMesh::add_triangle(Node::index_type a, Node::index_type b, Node::index_type c)
{
  faces_.push_back(a);
  faces_.push_back(b);
  faces_.push_back(c);
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
}


TriSurfMesh::Elem::index_type
TriSurfMesh::add_elem(Node::array_type a)
{
  faces_.push_back(a[0]);
  faces_.push_back(a[1]);
  faces_.push_back(a[2]);
  synchronized_ &= ~EDGE_NEIGHBORS_E;
  synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~NORMALS_E;
  return static_cast<Elem::index_type>((faces_.size() - 1) / 3);
}


void
TriSurfMesh::compute_edge_neighbors(double err)
{
  // TODO: This is probably broken with the new indexed edges.
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");

  hash_map<pair<int, int>, int, edgehash, edgecompare> edge_map;
  
  edge_neighbors_.resize(faces_.size());
  for (unsigned int j = 0; j < edge_neighbors_.size(); j++)
  {
    edge_neighbors_[j] = -1;
  }

  int i;
  for (i=faces_.size()-1; i >= 0; i--)
  {
    const int a = i;
    const int b = a - a % 3 + (a+1) % 3;

    int n0 = faces_[a];
    int n1 = faces_[b];
    int tmp;
    if (n0 > n1) { tmp = n0; n0 = n1; n1 = tmp; }

    pair<int, int> nodes(n0, n1);
    
    hash_map<pair<int, int>, int, edgehash, edgecompare>::iterator maploc;
    maploc = edge_map.find(nodes);
    if (maploc != edge_map.end())
    {
      edge_neighbors_[(*maploc).second] = i;
      edge_neighbors_[i] = (*maploc).second;
    }
    edge_map[nodes] = i;
  }

  synchronized_ |= EDGE_NEIGHBORS_E;
}



TriSurfMesh::Node::index_type
TriSurfMesh::add_point(const Point &p)
{
  points_.push_back(p);
  if (synchronized_ & NORMALS_E) normals_.push_back(Vector());
  if (synchronized_ & NODE_NEIGHBORS_E) node_neighbors_.push_back(set<int>());
  return static_cast<Node::index_type>(points_.size() - 1);
}



void
TriSurfMesh::add_triangle(const Point &p0, const Point &p1, const Point &p2)
{
  add_triangle(add_find_point(p0), add_find_point(p1), add_find_point(p2));
}


#define TRISURFMESH_VERSION 1

void
TriSurfMesh::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), TRISURFMESH_VERSION);

  Mesh::io(stream);

  Pio(stream, points_);
  Pio(stream, faces_);
  Pio(stream, edge_neighbors_);

  stream.end_class();

  if (stream.reading())
  {
    synchronized_ |= EDGE_NEIGHBORS_E;
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
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");

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
