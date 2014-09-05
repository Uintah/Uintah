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
 *  StructHexVolMesh.cc: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan &&
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Packages/Fusion/Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Geometry/BBox.h>
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID StructHexVolMesh::type_id("StructHexVolMesh", "Mesh", maker);


void
StructHexVolMesh::get_random_point(Point &p, const Elem::index_type &ei,
			     int seed) const
{
  // TODO :
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
#if 0
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
  
  // Recompute grid.
  grid_.detach();
  compute_grid();
#endif
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

void
StructHexVolMesh::get_weights(const Point &p,
			      Node::array_type &nodes, vector<double> &w)
{
  Cell::index_type cell;
  if (locate(cell, p))
  {
    get_nodes(nodes, cell);
    Vector v[8];
    double vol[8];

    int i;
    for (i = 0; i < 8; i++)
    {
      Point np;
      get_point(np, nodes[i]);
      v[i] = np - p;
    }
    vol[0] =
      Cross(Cross(v[1], v[2]), v[6]).length() +
      Cross(Cross(v[1], v[3]), v[2]).length() +
      Cross(Cross(v[1], v[4]), v[5]).length() +
      Cross(Cross(v[1], v[6]), v[5]).length() +
      Cross(Cross(v[2], v[5]), v[6]).length() +
      Cross(Cross(v[3], v[4]), v[7]).length() +
      Cross(Cross(v[3], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[6]).length() +
      Cross(Cross(v[4], v[6]), v[7]).length();

    vol[1] =
      Cross(Cross(v[0], v[2]), v[3]).length() +
      Cross(Cross(v[0], v[3]), v[7]).length() +
      Cross(Cross(v[0], v[4]), v[5]).length() +
      Cross(Cross(v[0], v[4]), v[7]).length() +
      Cross(Cross(v[2], v[3]), v[7]).length() +
      Cross(Cross(v[2], v[5]), v[6]).length() +
      Cross(Cross(v[2], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[7]).length() +
      Cross(Cross(v[5], v[6]), v[7]).length();

    vol[2] =
      Cross(Cross(v[0], v[1]), v[3]).length() +
      Cross(Cross(v[0], v[1]), v[4]).length() +
      Cross(Cross(v[0], v[3]), v[4]).length() +
      Cross(Cross(v[1], v[4]), v[5]).length() +
      Cross(Cross(v[1], v[5]), v[6]).length() +
      Cross(Cross(v[3], v[4]), v[7]).length() +
      Cross(Cross(v[3], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[6]).length() +
      Cross(Cross(v[4], v[6]), v[7]).length();

    vol[3] =
      Cross(Cross(v[0], v[1]), v[2]).length() +
      Cross(Cross(v[0], v[1]), v[4]).length() +
      Cross(Cross(v[0], v[4]), v[7]).length() +
      Cross(Cross(v[1], v[2]), v[5]).length() +
      Cross(Cross(v[1], v[4]), v[5]).length() +
      Cross(Cross(v[2], v[5]), v[6]).length() +
      Cross(Cross(v[2], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[7]).length() +
      Cross(Cross(v[5], v[6]), v[7]).length();

    vol[4] =
      Cross(Cross(v[0], v[1]), v[2]).length() +
      Cross(Cross(v[0], v[1]), v[5]).length() +
      Cross(Cross(v[0], v[2]), v[3]).length() +
      Cross(Cross(v[0], v[3]), v[7]).length() +
      Cross(Cross(v[1], v[2]), v[5]).length() +
      Cross(Cross(v[2], v[3]), v[6]).length() +
      Cross(Cross(v[2], v[5]), v[6]).length() +
      Cross(Cross(v[3], v[6]), v[7]).length() +
      Cross(Cross(v[5], v[6]), v[7]).length();

    vol[5] =
      Cross(Cross(v[0], v[1]), v[3]).length() +
      Cross(Cross(v[0], v[1]), v[4]).length() +
      Cross(Cross(v[0], v[3]), v[4]).length() +
      Cross(Cross(v[1], v[2]), v[3]).length() +
      Cross(Cross(v[1], v[2]), v[6]).length() +
      Cross(Cross(v[2], v[3]), v[6]).length() +
      Cross(Cross(v[3], v[4]), v[7]).length() +
      Cross(Cross(v[3], v[6]), v[7]).length() +
      Cross(Cross(v[4], v[6]), v[7]).length();

    vol[6] =
      Cross(Cross(v[0], v[1]), v[2]).length() +
      Cross(Cross(v[0], v[1]), v[5]).length() +
      Cross(Cross(v[0], v[2]), v[3]).length() +
      Cross(Cross(v[0], v[3]), v[4]).length() +
      Cross(Cross(v[0], v[4]), v[5]).length() +
      Cross(Cross(v[1], v[2]), v[5]).length() +
      Cross(Cross(v[2], v[3]), v[7]).length() +
      Cross(Cross(v[3], v[4]), v[7]).length() +
      Cross(Cross(v[4], v[5]), v[7]).length();

    vol[7] =
      Cross(Cross(v[0], v[1]), v[3]).length() +
      Cross(Cross(v[0], v[1]), v[4]).length() +
      Cross(Cross(v[0], v[3]), v[4]).length() +
      Cross(Cross(v[1], v[2]), v[3]).length() +
      Cross(Cross(v[1], v[2]), v[6]).length() +
      Cross(Cross(v[1], v[4]), v[5]).length() +
      Cross(Cross(v[1], v[5]), v[6]).length() +
      Cross(Cross(v[2], v[3]), v[6]).length() +
      Cross(Cross(v[4], v[5]), v[6]).length();

    const double suminv = 1.0 / (vol[0] + vol[1] + vol[2] + vol[3] +
				 vol[4] + vol[5] + vol[6] + vol[7]);

    for (i = 0; i < 8; i++)
    {
      const double value = vol[i] * suminv;
      w.push_back(value);
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


void
StructHexVolMesh::set_point(const Node::index_type &i, const Point &p)
{
  points_(i.i_, i.j_, i.k_) = p;
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

#define LATVOLMESH_VERSION 1

void
StructHexVolMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), LATVOLMESH_VERSION);

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


} // namespace SCIRun
