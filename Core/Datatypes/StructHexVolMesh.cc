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
 *  StructHexVolMesh.cc: Templated Mesh defined on a 3D Structured Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 *
 */

/*
  See StructHexVolMesh.h for field/mesh comments.
*/

#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Plane.h>
#include <Core/Math/MusilRNG.h>
#include <iostream>


namespace SCIRun {

using namespace std;

PersistentTypeID StructHexVolMesh::type_id("StructHexVolMesh", "Mesh", maker);


StructHexVolMesh::StructHexVolMesh():
  grid_lock_("StructHexVolMesh grid lock")
{}

StructHexVolMesh::StructHexVolMesh(unsigned int i,
				   unsigned int j,
				   unsigned int k) :
  LatVolMesh(i, j, k, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
  points_(i, j, k),
  grid_lock_("StructHexVolMesh grid lock")
{}

StructHexVolMesh::StructHexVolMesh(const StructHexVolMesh &copy) :
  LatVolMesh(copy),
  grid_lock_("StructHexVolMesh grid lock")
{
  points_.copy( copy.points_ );
}

bool
StructHexVolMesh::get_dim(vector<unsigned int> &array) const
{
  array.resize(3);
  array.clear();

  array.push_back(ni_);
  array.push_back(nj_);
  array.push_back(nk_);

  return true;
}

BBox
StructHexVolMesh::get_bounding_box() const
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
StructHexVolMesh::transform(const Transform &t)
{
  Node::iterator i, ie;
  begin(i);
  end(ie);

  while (i != ie) {
    points_((*i).i_,(*i).j_,(*i).k_) = t.project(points_((*i).i_,(*i).j_,(*i).k_));

    ++i;
  }
  
  // Recompute grid.
  grid_.detach();
  compute_grid();
}

void
StructHexVolMesh::get_center(Point &result, const Node::index_type &idx) const
{
  result = points_(idx.i_, idx.j_, idx.k_);
}


void
StructHexVolMesh::get_center(Point &result, const Edge::index_type &idx) const
{
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);
  
  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


void
StructHexVolMesh::get_center(Point &result, const Face::index_type &idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  Node::array_type::iterator nai = nodes.begin();
  get_point(result, *nai);
  ++nai;
  while (nai != nodes.end())
  {
    Point pp;
    result.asVector() += pp.asVector();
    ++nai;
  }
  result.asVector() *= (1.0 / 4.0);
}


void
StructHexVolMesh::get_center(Point &result, const Cell::index_type &idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 8);
  Node::array_type::iterator nai = nodes.begin();
  get_point(result, *nai);
  ++nai;
  while (nai != nodes.end())
  {
    Point pp;
    result.asVector() += pp.asVector();
    ++nai;
  }
  result.asVector() *= (1.0 / 8.0);
}


double
StructHexVolMesh::inside8_p(Cell::index_type idx, const Point &p) const
{
  static const int table[6][3][3] =
  {{{0, 0, 0},
    {0, 1, 0},
    {0, 0, 1}},

   {{0, 0, 0},
    {1, 0, 0},
    {0, 1, 0}},

   {{0, 0, 0},
    {0, 0, 1},
    {1, 0, 0}},

   {{1, 1, 1},
    {1, 1, 0},
    {1, 0, 1}},

   {{1, 1, 1},
    {1, 0, 1},
    {0, 1, 1}},

   {{1, 1, 1},
    {0, 1, 1},
    {1, 1, 0}}};
  
  Point center;
  get_center(center, idx);

  double minval = 1.0e6;
  for (int i = 0; i < 6; i++)
  {
    Node::index_type n0(this,
			idx.i_ + table[i][0][0],
			idx.j_ + table[i][0][1],
			idx.k_ + table[i][0][2]);
    Node::index_type n1(this,
			idx.i_ + table[i][1][0],
			idx.j_ + table[i][1][1],
			idx.k_ + table[i][1][2]);
    Node::index_type n2(this,
			idx.i_ + table[i][2][0],
			idx.j_ + table[i][2][1],
			idx.k_ + table[i][2][2]);

    Point p0, p1, p2;
    get_center(p0, n0);
    get_center(p1, n1);
    get_center(p2, n2);

    const Vector v0(p1 - p0), v1(p2 - p0);
    const Vector normal = Cross(v0, v1);
    const Vector off0(p0 - p);
    const Vector off1(p0 - center);
    const double tmp = Dot(normal, off0) * Dot(normal, off1);
    if (tmp < -1.0e-12)
    {
      return tmp;
    }
    if (tmp < minval)
    {
      minval = tmp;
    }
  }
  return minval;
}


bool
StructHexVolMesh::locate(Cell::index_type &cell, const Point &p)
{
  if (grid_.get_rep() == 0)
  {
    compute_grid();
  }
  cell.mesh_ = this;
  LatVolMeshHandle mesh = grid_->get_typed_mesh();
  LatVolMesh::Cell::index_type ci;
  if (!mesh->locate(ci, p)) { return false; }
  const vector<Cell::index_type> &v = grid_->value(ci);
  vector<Cell::index_type>::const_iterator iter = v.begin();
  double mindist = -1.0;
  while (iter != v.end()) {
    const double tmp = inside8_p(*iter, p);
    if (tmp > mindist)
    {
      cell = *iter;
      mindist = tmp;
    }
    ++iter;
  }
  if (mindist > -1.0e-12)
  {
    return true;
  }
  else
  {
    return false;
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
StructHexVolMesh::locate(Node::index_type &node, const Point &p)
{
  node.mesh_ = this;
  Cell::index_type ci;
  if (locate(ci, p)) { // first try the fast way.
    Node::array_type nodes;
    get_nodes(nodes, ci);

    double dmin = distance2(p, points_(nodes[0].i_, nodes[0].j_, nodes[0].k_));
    node = nodes[0];
    for (unsigned int i = 1; i < nodes.size(); i++)
    {
      const double d =
	distance2(p, points_(nodes[i].i_, nodes[i].j_, nodes[i].k_));
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

    double min_dist = distance2(p, points_((*ni).i_, (*ni).j_, (*ni).k_));
    node = *ni;
    ++ni;

    while (ni != nie)
    {
      const double dist = distance2(p, points_((*ni).i_, (*ni).j_, (*ni).k_));
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


void
StructHexVolMesh::get_weights(const Point &p,
			      Cell::array_type &l, vector<double> &w)
{
  Cell::index_type idx;
  if (locate(idx, p))
  {
    l.push_back(idx);
    w.push_back(1.0);
  }
}



//===================================================================

// area3D_Polygon(): computes the area of a 3D planar polygon
//    Input:  int n = the number of vertices in the polygon
//            Point* V = an array of n+2 vertices in a plane
//                       with V[n]=V[0] and V[n+1]=V[1]
//            Point N = unit normal vector of the polygon's plane
//    Return: the (float) area of the polygon

// Copyright 2000, softSurfer (www.softsurfer.com)
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.
 
double
StructHexVolMesh::polygon_area(const Node::array_type &ni, const Vector N) const
{
    double area = 0;
    double an, ax, ay, az;  // abs value of normal and its coords
    int   coord;           // coord to ignore: 1=x, 2=y, 3=z
    int   i, j, k;         // loop indices
    const unsigned int n = ni.size();

    // select largest abs coordinate to ignore for projection
    ax = (N.x()>0 ? N.x() : -N.x());     // abs x-coord
    ay = (N.y()>0 ? N.y() : -N.y());     // abs y-coord
    az = (N.z()>0 ? N.z() : -N.z());     // abs z-coord

    coord = 3;                     // ignore z-coord
    if (ax > ay) {
        if (ax > az) coord = 1;    // ignore x-coord
    }
    else if (ay > az) coord = 2;   // ignore y-coord

    // compute area of the 2D projection
    for (i=1, j=2, k=0; i<=n; i++, j++, k++)
        switch (coord) {
        case 1:
            area += (get_point(ni[i%n]).y() *
		     (get_point(ni[j%n]).z() - get_point(ni[k%n]).z()));
            continue;
        case 2:
            area += (get_point(ni[i%n]).x() * 
		     (get_point(ni[j%n]).z() - get_point(ni[k%n]).z()));
            continue;
        case 3:
            area += (get_point(ni[i%n]).x() * 
		     (get_point(ni[j%n]).y() - get_point(ni[k%n]).y()));
            continue;
        }

    // scale to get area before projection
    an = sqrt( ax*ax + ay*ay + az*az);  // length of normal vector
    switch (coord) {
    case 1:
        area *= (an / (2*ax));
        break;
    case 2:
        area *= (an / (2*ay));
        break;
    case 3:
        area *= (an / (2*az));
    }
    return area;
}

double
StructHexVolMesh::pyramid_volume(const Node::array_type &face, const Point &p) const
{
  Vector e1(get_point(face[1])-get_point(face[0]));
  Vector e2(get_point(face[1])-get_point(face[2]));
  if (Cross(e1,e2).length2()>0.0) {
    Plane plane(get_point(face[0]), get_point(face[1]), get_point(face[2]));
    double dist = plane.eval_point(p);
    return fabs(plane.eval_point(p)*polygon_area(face,plane.normal())*0.25);
  }
  Vector e3(get_point(face[3])-get_point(face[2]));
  if (Cross(e2,e3).length2()>0.0) {
    Plane plane(get_point(face[1]), get_point(face[2]), get_point(face[3]));
    double dist = plane.eval_point(p);
    return fabs(plane.eval_point(p)*polygon_area(face,plane.normal())*0.25);
  }
  return 0.0;
}
  
  

void
StructHexVolMesh::get_weights(const Point &p,
			Node::array_type &nodes, vector<double> &w)
{
  synchronize (Mesh::FACES_E);
  Cell::index_type cell;
  if (locate(cell, p))
  {
    get_nodes(nodes,cell);
    unsigned int nnodes = nodes.size();
    ASSERT(nnodes == 8);
    w.resize(nnodes);
      
    Face::array_type faces;
    get_faces(faces, cell);
    unsigned int nfaces = faces.size();
    vector<double> face_point_volume(nfaces);
    double total_volume = 0.0;
    map<Node::index_type, map<Face::index_type,bool> > attached;
    int f, n;
      
    for (f = 0; f < nfaces; f++)
    {
      Node::array_type face_nodes;
      get_nodes(face_nodes, faces[f]);
      face_point_volume[f] = pyramid_volume(face_nodes, p);
      total_volume += face_point_volume[f];
      unsigned int n_face_nodes = face_nodes.size();
      for (n = 0; n < n_face_nodes; n++)
	attached[face_nodes[n]][faces[f]] = true;
    }
    for (n = 0; n < nnodes; n++)
    {
      double unattached_volume = 0.0;
      for (f = 0; f < nfaces; f++)
	if (!attached[nodes[n]][faces[f]])
	  unattached_volume += face_point_volume[f];
      w[n] = unattached_volume / total_volume;
    }
  }
}


void
StructHexVolMesh::compute_grid()
{
  grid_lock_.lock();
  if (grid_.get_rep() != 0) {grid_lock_.unlock(); return;} // only create once.

  BBox bb = get_bounding_box();
  if (!bb.valid()) { grid_lock_.unlock(); return; }

  // Cubed root of number of cells to get a subdivision ballpark.
  const double one_third = 1.L/3.L;
  Cell::size_type csize;
  size(csize);
  const double dsize = csize.i_ * csize.j_ * csize.k_;
  const int s = ((int)ceil(pow(dsize , one_third))) / 2 + 2;
  const Vector cell_epsilon = bb.diagonal() * (0.01 / s);
  bb.extend(bb.min() - cell_epsilon*2);
  bb.extend(bb.max() + cell_epsilon*2);

  LatVolMeshHandle mesh(scinew LatVolMesh(s, s, s, bb.min(), bb.max()));
  grid_ = scinew LatVolField<vector<Cell::index_type> >(mesh, Field::CELL);
  LatVolField<vector<Cell::index_type> >::fdata_type &fd = grid_->fdata();

  BBox box;
  Node::array_type nodes;
  Cell::iterator ci, cie;
  begin(ci); end(cie);
  while(ci != cie)
  {
    get_nodes(nodes, *ci);

    box.reset();
    for (unsigned int i = 0; i < nodes.size(); i++)
    {
      box.extend(points_(nodes[i].i_, nodes[i].j_, nodes[i].k_));
    }
    const Point padmin(box.min() - cell_epsilon);
    const Point padmax(box.max() + cell_epsilon);
    box.extend(padmin);
    box.extend(padmax);

    // add this cell index to all overlapping cells in grid_
    LatVolMesh::Cell::array_type carr;
    mesh->get_cells(carr, box);
    LatVolMesh::Cell::array_type::iterator giter = carr.begin();
    while (giter != carr.end()) {
      // Would like to just get a reference to the vector at the cell
      // but can't from value. Bypass the interface.
      vector<Cell::index_type> &v = fd[*giter];
      v.push_back(*ci);
      ++giter;
    }
    ++ci;
  }
  grid_lock_.unlock();
}



double
StructHexVolMesh::get_size(Node::index_type idx) const
{
  return 0.0;
}


double
StructHexVolMesh::get_size(Edge::index_type idx) const
{
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p0, p1;
  get_center(p0, arr[0]);
  get_center(p1, arr[1]);

  return (p1.asVector() - p0.asVector()).length();
}
  

double
StructHexVolMesh::get_size(Face::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  Point p0, p1, p2;
  get_point(p0, nodes[0]);
  get_point(p1, nodes[1]);
  get_point(p2, nodes[2]);
  Vector v0 = p1 - p0;
  Vector v1 = p2 - p0;
  return (v0.length() * v1.length());
}


double
StructHexVolMesh::get_size(Cell::index_type idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
  Point p0, p1, p2, p3;
  get_point(p0, nodes[0]);
  get_point(p1, nodes[1]);
  get_point(p2, nodes[3]);
  get_point(p3, nodes[4]);
  Vector v0 = p1 - p0;
  Vector v1 = p2 - p0;
  Vector v2 = p3 - p0;
  return (v0.length() * v1.length() * v2.length());
}



void
StructHexVolMesh::set_point(const Point &p, const Node::index_type &i)
{
  points_(i.i_, i.j_, i.k_) = p;
}

#define STRUCT_HEX_VOL_MESH_VERSION 1

void
StructHexVolMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), STRUCT_HEX_VOL_MESH_VERSION);

  LatVolMesh::io(stream);

  // IO data members, in order
  Pio(stream, points_);

  stream.end_class();
}

const string
StructHexVolMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "StructHexVolMesh";
  return name;
}

const TypeDescription*
StructHexVolMesh::get_type_description() const
{
  return SCIRun::get_type_description((StructHexVolMesh *)0);
}

const TypeDescription*
get_type_description(StructHexVolMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("StructHexVolMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
