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
#include <Core/Math/Trig.h>
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
    edges_(copy.edges_),
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
TriSurfMesh::get_random_point(Point &p, Face::index_type ei, int seed) const
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
TriSurfMesh::transform(const Transform &t)
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




bool
TriSurfMesh::locate(Node::index_type &loc, const Point &p) const
{
  Node::iterator ni, nie;
  begin(ni);
  end(nie);


  if (ni == nie)
  {
    return false;
  }

  double min_dist = (p - points_[*ni]).length2();
  loc = *ni;
  ++ni;

  while (ni != nie)
  {
    const double dist = (p - points_[*ni]).length2();
    if (dist < min_dist)
    {
      min_dist = dist;
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
  begin(bi);
  end(ei);
  
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

//! return the area of the triangle.
double
TriSurfMesh::get_gradient_basis(Face::index_type fi, Vector& g0, Vector& g1,
			       Vector& g2)
{
  Point& p1 = points_[faces_[fi * 3]];
  Point& p2 = points_[faces_[fi * 3+1]];
  Point& p3 = points_[faces_[fi * 3+2]];

  double x1=p1.x();
  double y1=p1.y();
  double z1=p1.z();

  // rotate these points into the xy (z=0) plane
  Transform trans1;
  Transform rot1;
  Transform rot2;
  Transform rot3;

  Point newp1;
  Point newp2;
  Point newp3;
  Point tempp1;
  Point tempp2;
  Point tempp3;
  newp1 = p1;
  newp2 = p2;
  newp3 = p3;

  // 1. Translate so that p1 is at the origin
  trans1.post_translate(Vector(-x1,-y1,-z1));
  tempp1 = trans1.project(newp1);
  tempp2 = trans1.project(newp2);
  tempp3 = trans1.project(newp3);
  newp1 = tempp1;
  newp2 = tempp2;
  newp3 = tempp3;

  // 2. Rotate about z axis - this will bring the p1-p2 edge into the xz 
  //    plane and make p2y=0;
  double phi = Atan(newp2.y()/newp2.x());
  rot1.post_rotate(-phi, Vector(0,0,1));
  tempp2 = rot1.project(newp2);
  tempp3 = rot1.project(newp3);
  newp2 = tempp2;
  newp3 = tempp3;

  // 3. Rotate about y-axis so that newp2.z() = 0 and p1-p2 edge is 
  //    coincident with the x-axis
  double theta = Atan(newp2.z()/newp2.x());
  rot2.post_rotate(theta, Vector(0,1,0));
  tempp2 = rot2.project(newp2);
  tempp3 = rot2.project(newp3);
  newp2 = tempp2;
  newp3 = tempp3;

  // 4. Rotate p3 about x axis so that pnew3.z() = 0.
  theta = Atan(newp3.z()/newp3.y());
  rot3.post_rotate(theta, Vector(1,0,0));
  tempp3 = rot3.project(newp3);
  newp3 = tempp3;

  // All nodes are now in the x-y plane. Compute basis vectors.

  // first compute iA2, 1/(2xArea) (with z=0)
  double x1prime=newp1.x();
  double y1prime=newp1.y();
  double x2prime=newp2.x();
  double y2prime=newp2.y();
  double x3prime=newp3.x();
  double y3prime=newp3.y();
  
  double a1=+(x2prime*y3prime-x3prime*y2prime);
  double a2=-(x1prime*y3prime-x3prime*y1prime);
  double a3=+(x1prime*y2prime-x2prime*y1prime);

  double iA2=1./(a1+a2+a3);

  // compute the gradient basis vectors (with z=0)
  double b0=y2prime-y3prime;
  double c0=x3prime-x2prime;

  Vector g0prime=Vector(b0*iA2, c0*iA2, 0);
  
  double b1=y3prime-y1prime;
  double c1=x1prime-x3prime;

  Vector g1prime=Vector(b1*iA2, c1*iA2, 0);

  double b2=y1prime-y2prime;
  double c2=x2prime-x1prime;

  Vector g2prime=Vector(b2*iA2, c2*iA2, 0);

  // apply the inverse rotations to the basis vectors
  Vector g0temp;
  Vector g1temp;
  Vector g2temp; 
  g0temp = rot3.unproject(g0prime);
  g0 = rot2.unproject(g0temp);
  g0temp = g0;
  g0 = rot1.unproject(g0temp);
  g0temp = g0;
  g1temp = rot3.unproject(g1prime);
  g1 = rot2.unproject(g1temp);
  g1temp = g1;
  g1 = rot1.unproject(g1temp);
  g1temp = g1;
  g2temp = rot3.unproject(g2prime);
  g2 = rot2.unproject(g2temp);
  g2temp = g2;
  g2 = rot1.unproject(g2temp);
  g2temp = g2;

  // apply the inverse translation
  g0 = trans1.unproject(g0temp);
  g1 = trans1.unproject(g1temp);
  g2 = trans1.unproject(g2temp);
  
  // return the area of the element
  double area=(1./iA2)/2.0;
  return(area);
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
  const double a = Cross(v01, v02).length(); // area of the whole triangle (2x)
  const double a0 = Cross(v1, v2).length();  // area opposite p0
  const double a1 = Cross(v2, v0).length();  // area opposite p1
  const double a2 = Cross(v0, v1).length();  // area opposite p2
  const double s = a0+a1+a2;
  return fabs(s - a) < 1.0e-12 && a > 1.0e-12;
}


void
TriSurfMesh::get_weights(const Point &p,
			 Node::array_type &l, vector<double> &w)
{
  Face::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    Point p0, p1, p2;
    get_point(p0,l[0]);
    get_point(p1,l[1]);
    get_point(p2,l[2]);
    w.resize(3);
    w[0] = (Cross(p1-p,p2-p)).length();
    w[1] = (Cross(p0-p,p2-p)).length();
    w[2] = (Cross(p0-p,p1-p)).length();
    const double area_sum_inv = 1.0 / (w[0] + w[1] + w[2]);
    w[0] *= area_sum_inv;
    w[1] *= area_sum_inv;
    w[2] *= area_sum_inv;
  }
}


void
TriSurfMesh::get_center(Point &result, Edge::index_type idx) const
{
  ASSERTMSG(synchronized_ & EDGES_E,
	    "Must call synchronize EDGES_E on TriSurfMesh first");

  Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);

  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


void
TriSurfMesh::get_center(Point &p, Face::index_type i) const
{
  Node::array_type nodes;
  get_nodes(nodes, i);
  Node::array_type::iterator nai = nodes.begin();
  ASSERT(nodes.size() == 3);
  get_point(p, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    p.asVector() += pp.asVector();
    ++nai;
  }
  p.asVector() *= (1.0 / 3.0);
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


void
TriSurfMesh::insert_node(Face::index_type face, const Point &p)
{
  const bool do_neighbors = synchronized_ & EDGE_NEIGHBORS_E;
  const bool do_normals = false; // synchronized_ & NORMALS_E;

  Node::index_type pi = add_point(p);
  const unsigned f0 = face*3;
  const unsigned f1 = faces_.size();
  const unsigned f2 = f1+3;
  
  faces_.push_back(faces_[f0+1]);
  faces_.push_back(faces_[f0+2]);
  faces_.push_back(pi);

  faces_.push_back(faces_[f0+2]);
  faces_.push_back(faces_[f0+0]);
  faces_.push_back(pi);

  // must do last
  faces_[f0+2] = pi;

  if (do_neighbors)
  {
    edge_neighbors_.push_back(edge_neighbors_[f0+1]);
    if (edge_neighbors_.back() != -1) 
      edge_neighbors_[edge_neighbors_.back()] = edge_neighbors_.size()-1;
    edge_neighbors_.push_back(f2+2);
    edge_neighbors_.push_back(f0+1);
    
    edge_neighbors_.push_back(edge_neighbors_[f0+2]);
    if (edge_neighbors_.back() != -1) 
      edge_neighbors_[edge_neighbors_.back()] = edge_neighbors_.size()-1;
    edge_neighbors_.push_back(f0+2);
    edge_neighbors_.push_back(f1+1);
    
    edge_neighbors_[f0+1] = f1+2;
    edge_neighbors_[f0+2] = f2+1;
  }

  if (do_normals)
  {
    Vector normal = Vector( (p.asVector() +
			     normals_[faces_[f0]] + 
			     normals_[faces_[f1]] + 
			     normals_[faces_[f2]]).normalize() );

    normals_.push_back(normals_[faces_[f1]]);
    normals_.push_back(normals_[faces_[f2]]);
    normals_.push_back(normal);

    normals_.push_back(normals_[faces_[f2]]);
    normals_.push_back(normals_[faces_[f0]]);
    normals_.push_back(normal);

    normals_[faces_[f0+2]] = normal;

  }

  if (!do_neighbors) synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~EDGES_E;
  if (!do_normals) synchronized_ &= ~NORMALS_E;

    
  
}

  
  


bool
TriSurfMesh::insert_node(const Point &p)
{
  Face::index_type face;
  if (!locate(face,p)) return false;
  insert_node(face,p);
  return true;
}

 
/*             2
//             ^
//            / \
//           /f3 \
//        5 /-----\ 4
//         / \fac/ \
//        /f1 \ /f2 \
//       /     V     \
//      <------------->
//     0       3       1
*/

#define DEBUGINFO(f) cerr << "Face #" << f/3 << " N1: " << faces_[f+0] << " N2: " << faces_[f+1] << " N3: " << faces_[f+2] << "  B1: " << edge_neighbors_[f] << " B2: " << edge_neighbors_[f+1] << "  B3: " << edge_neighbors_[f+2] << endl;
void
TriSurfMesh::bisect_element(const Face::index_type face)
{
  const bool do_neighbors = synchronized_ & EDGE_NEIGHBORS_E;
  const bool do_normals = false; //synchronized_ & NORMALS_E;
  
  const unsigned f0 = face*3;
  Node::array_type nodes;
  get_nodes(nodes,face);
  vector<Vector> normals(3);
  for (int edge = 0; edge < 3; ++edge)
  {
    Point p = ((points_[faces_[f0+edge]] + 
		points_[faces_[next(f0+edge)]]) / 2.0).asPoint();
    nodes.push_back(add_point(p));

    if (do_normals)
      normals[edge] = Vector((normals_[faces_[f0+edge]] + 
			     normals_[faces_[next(f0+edge)]]).normalize());

  }

  const unsigned f1 = faces_.size();
  faces_.push_back(nodes[0]);
  faces_.push_back(nodes[3]);
  faces_.push_back(nodes[5]);

  const unsigned f2 = faces_.size();
  faces_.push_back(nodes[1]);
  faces_.push_back(nodes[4]);
  faces_.push_back(nodes[3]);

  const unsigned f3 = faces_.size();
  faces_.push_back(nodes[2]);
  faces_.push_back(nodes[5]);
  faces_.push_back(nodes[4]);

  faces_[f0+0] = nodes[3];
  faces_[f0+1] = nodes[4];
  faces_[f0+2] = nodes[5];

  if (do_neighbors)
  {

    edge_neighbors_.push_back(edge_neighbors_[f0+0]);
    edge_neighbors_.push_back(f0+2);
    edge_neighbors_.push_back(-1);
    
    edge_neighbors_.push_back(edge_neighbors_[f0+1]);
    edge_neighbors_.push_back(f0+0);
    edge_neighbors_.push_back(-1);
    
    edge_neighbors_.push_back(edge_neighbors_[f0+2]);
    edge_neighbors_.push_back(f0+1);
    edge_neighbors_.push_back(-1);    

    // must do last
    edge_neighbors_[f0+0] = f2+1;
    edge_neighbors_[f0+1] = f3+1;
    edge_neighbors_[f0+2] = f1+1;

  }

  if (do_normals)
  {
    normals_.push_back(normals_[f0+0]);
    normals_.push_back(normals[0]);
    normals_.push_back(normals[2]);
    
    normals_.push_back(normals_[f0+1]);
    normals_.push_back(normals[1]);
    normals_.push_back(normals[0]);
    
    normals_.push_back(normals_[f0+2]);
    normals_.push_back(normals[2]);
    normals_.push_back(normals[1]);    

    normals_[f0+0] = normals[0];
    normals_[f0+1] = normals[1];
    normals_[f0+2] = normals[2];
  }
  

  if (do_neighbors && edge_neighbors_[f1] != -1)
  {
    const unsigned nbr = edge_neighbors_[f1];
    const unsigned pnbr = prev(nbr);
    const unsigned f4 = faces_.size();
    faces_.push_back(nodes[1]);
    faces_.push_back(nodes[3]);
    faces_.push_back(faces_[pnbr]);
    edge_neighbors_[f2+2] = f4;
    edge_neighbors_.push_back(f2+2);
    edge_neighbors_.push_back(pnbr);
    edge_neighbors_.push_back(edge_neighbors_[pnbr]);
    edge_neighbors_[edge_neighbors_.back()] = f4+2;
    faces_[nbr] = nodes[3];
    edge_neighbors_[pnbr] = f4+1;    
    if (do_normals)
    {
      normals_[nbr] = normals[0];
      normals_.push_back(normals_[f0+1]);
      normals_.push_back(normals[0]);
      normals_.push_back(normals_[pnbr]);
    }
    
  }

  if (do_neighbors && edge_neighbors_[f2] != -1)
  {
    const unsigned nbr = edge_neighbors_[f2];
    const unsigned pnbr = prev(nbr);
    const unsigned f5 = faces_.size();
    faces_.push_back(nodes[2]);
    faces_.push_back(nodes[4]);
    faces_.push_back(faces_[pnbr]);
    edge_neighbors_[f3+2] = f5;
    edge_neighbors_.push_back(f3+2);
    edge_neighbors_.push_back(pnbr);
    edge_neighbors_.push_back(edge_neighbors_[pnbr]);
    edge_neighbors_[edge_neighbors_.back()] = f5+2;
    faces_[nbr] = nodes[4];
    edge_neighbors_[pnbr] = f5+1;
    if (do_normals)
    {
      normals_[nbr] = normals[1];
      normals_.push_back(normals_[f0+2]);
      normals_.push_back(normals[1]);
      normals_.push_back(normals_[pnbr]);
    }
  }

  if (do_neighbors && edge_neighbors_[f3] != -1)
  {
    const unsigned nbr = edge_neighbors_[f3];
    const unsigned pnbr = prev(nbr);
    const unsigned f6 = faces_.size();
    faces_.push_back(nodes[0]);
    faces_.push_back(nodes[5]);
    faces_.push_back(faces_[pnbr]);
    edge_neighbors_[f1+2] = f6;
    edge_neighbors_.push_back(f1+2);
    edge_neighbors_.push_back(pnbr);
    edge_neighbors_.push_back(edge_neighbors_[pnbr]);
    edge_neighbors_[edge_neighbors_.back()] = f6+2;
    faces_[nbr] = nodes[5];
    edge_neighbors_[pnbr] = f6+1;
    if (do_normals)
    {
      normals_[nbr] = normals[2];
      normals_.push_back(normals_[f0+0]);
      normals_.push_back(normals[2]);
      normals_.push_back(normals_[pnbr]);
    }
  }   

  if (!do_neighbors) synchronized_ &= ~NODE_NEIGHBORS_E;
  synchronized_ &= ~EDGES_E;
  if (!do_normals) synchronized_ &= ~NORMALS_E;

}


struct edgecompare
{
  bool operator()(const pair<int, int> &a, const pair<int, int> &b) const
  {
    return a.first == b.first && a.second == b.second;
  }
};


#ifdef HAVE_HASH_MAP

struct edgehash
{
  size_t operator()(const pair<int, int> &a) const
  {
    hash<int> hasher;
    return hasher(hasher(a.first) + a.second);
  }
};

#endif

void
TriSurfMesh::compute_edges()
{

#ifdef HAVE_HASH_MAP

  hash_map<pair<int, int>, int, edgehash, edgecompare> edge_map;

#else

  map<pair<int, int>, int, edgecompare> edge_map;

#endif
  
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

#ifdef HAVE_HASH_MAP

  hash_map<pair<int, int>, int, edgehash, edgecompare>::iterator itr;

#else

  map<pair<int, int>, int, edgecompare>::iterator itr;

#endif

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
  if (locate(i, p) && (p - points_[i]).length2() < err)
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

#ifdef HAVE_HASH_MAP

  hash_map<pair<int, int>, int, edgehash, edgecompare> edge_map;

#else

  map<pair<int, int>, int, edgecompare> edge_map;

#endif
  
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
    
#ifdef HAVE_HASH_MAP

    hash_map<pair<int, int>, int, edgehash, edgecompare>::iterator maploc;

#else

    map<pair<int, int>, int, edgecompare>::iterator maploc;

#endif
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

  if (stream.reading() && edge_neighbors_.size())
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
