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
#include <Core/Math/MusilRNG.h>
#include <iostream>


namespace SCIRun {

using namespace std;

PersistentTypeID StructHexVolMesh::type_id("StructHexVolMesh", "Mesh", maker);


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
StructHexVolMesh::transform(Transform &t)
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
StructHexVolMesh::get_center(Point &result, const Cell::index_type &idx) const
{
  Node::array_type nodes;
  get_nodes(nodes, idx);
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
  result = v.asPoint();
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

// The volume x 6, used by get_weights to compute barycentric coordinates.
static double
tet_vol6(const Point &p1, const Point &p2, const Point &p3, const Point &p4)
{
  const double x1=p1.x();
  const double y1=p1.y();
  const double z1=p1.z();
  const double x2=p2.x();
  const double y2=p2.y();
  const double z2=p2.z();
  const double x3=p3.x();
  const double y3=p3.y();
  const double z3=p3.z();
  const double x4=p4.x();
  const double y4=p4.y();
  const double z4=p4.z();
  const double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
  const double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
  const double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
  const double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
  return fabs(a1+a2+a3+a4);
}


// Tet inside test, cut and pasted from TetVolMesh.cc
static bool
tet_inside_p(const Point &p, const Point &p0, const Point &p1,
	     const Point &p2, const Point &p3)
{
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
  if (s0 < -1.e-12)
    return false;

  const double b1 = + (y3*z0-y0*z3) + (y0*z2-y2*z0) + (y2*z3-y3*z2);
  const double c1 = - (x3*z0-x0*z3) - (x0*z2-x2*z0) - (x2*z3-x3*z2);
  const double d1 = + (x3*y0-x0*y3) + (x0*y2-x2*y0) + (x2*y3-x3*y2);
  const double s1 = iV6 * (a1 + b1*p.x() + c1*p.y() + d1*p.z());
  if (s1 < -1.e-12)
    return false;

  const double b2 = - (y0*z1-y1*z0) - (y1*z3-y3*z1) - (y3*z0-y0*z3);
  const double c2 = + (x0*z1-x1*z0) + (x1*z3-x3*z1) + (x3*z0-x0*z3);
  const double d2 = - (x0*y1-x1*y0) - (x1*y3-x3*y1) - (x3*y0-x0*y3);
  const double s2 = iV6 * (a2 + b2*p.x() + c2*p.y() + d2*p.z());
  if (s2 < -1.e-12)
    return false;

  const double b3 = +(y1*z2-y2*z1) + (y2*z0-y0*z2) + (y0*z1-y1*z0);
  const double c3 = -(x1*z2-x2*z1) - (x2*z0-x0*z2) - (x0*z1-x1*z0);
  const double d3 = +(x1*y2-x2*y1) + (x2*y0-x0*y2) + (x0*y1-x1*y0);
  const double s3 = iV6 * (a3 + b3*p.x() + c3*p.y() + d3*p.z());
  if (s3 < -1.e-12)
    return false;

  return true;
}

static void
tetinterp(const Point &p, Point nodes[8],
	  vector<double> &w, int a, int b, int c, int d)
{
  int i;
  w.resize(8);
  for (i=0; i < 8; i++)
  {
    w[i] = 0.0;
  }
  
  const double wa = tet_vol6(p, nodes[b], nodes[c], nodes[d]); 
  const double wb = tet_vol6(p, nodes[a], nodes[c], nodes[d]); 
  const double wc = tet_vol6(p, nodes[a], nodes[b], nodes[d]); 
  const double wd = tet_vol6(p, nodes[a], nodes[b], nodes[c]); 

  const double sum = 1.0 / (wa + wb + wc + wd);
  
  w[a] = wa * sum;
  w[b] = wb * sum;
  w[c] = wc * sum;
  w[d] = wd * sum;
}


void
StructHexVolMesh::get_weights(const Point &p,
			      Node::array_type &nodes, vector<double> &w)
{
  Cell::index_type cell;
  if (locate(cell, p))
  {
    get_nodes(nodes, cell);
    Point v[8];
    int i;
    for (i = 0; i < 8; i++)
    {
      get_point(v[i], nodes[i]);
    }
    
    if (tet_inside_p(p, v[0], v[4], v[3], v[1]))
    {
      tetinterp(p, v, w, 0, 4, 3, 1);
    }
    else if (tet_inside_p(p, v[2], v[6], v[1], v[3]))
    {
      tetinterp(p, v, w, 2, 6, 1, 3);
    }
    else if (tet_inside_p(p, v[5], v[1], v[6], v[4]))
    {
      tetinterp(p, v, w, 5, 1, 6, 4);
    }
    else if (tet_inside_p(p, v[7], v[3], v[4], v[6]))
    {
      tetinterp(p, v, w, 7, 3, 4, 6);
    }
    else
    {
      tetinterp(p, v, w, 1, 3, 4, 6);
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
StructHexVolMesh::set_point(const Node::index_type &i, const Point &p)
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
